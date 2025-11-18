from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

class StrategicProducer:
    def __init__(self, alpha_min, alpha_max, Pmin, Pmax, cost, demand, SOS1=False):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.Pmin = Pmin
        self.Pmax = Pmax
        self.cost = cost
        self.demand = demand
        self.SOS1 = SOS1

        self.num_generators = len(Pmin)

        self._build_model(SOS1,)
        self.solve()

    def _build_model(self, SOS1 = False):
        self.model = ConcreteModel()

        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.strategic_index = Set(initialize=[0])  # Index of the strategic producer

        self._build_variables(SOS1)
        self._build_objective()
        self._build_constraints(SOS1)

    def _build_variables(self, SOS1 = False):
        self.model.P_G = Var(self.model.n_gen, domain=Reals)
        self.model.alpha = Var(domain=Reals)
        self.model.lambda_dual = Var(domain=Reals)
        self.model.mu_min = Var(self.model.n_gen, domain=Reals)
        self.model.mu_max = Var(self.model.n_gen, domain=Reals)
        self.model.tau = Var(self.model.n_gen, domain=Binary)

        if SOS1 is False:
            self.model.z_min = Var(self.model.n_gen, domain=Binary)
            self.model.z_max = Var(self.model.n_gen, domain=Binary)
        else:
            # Slack variables (instead of z_min/z_max)
            self.model.slack_min = Var(self.model.n_gen, domain=NonNegativeReals)
            self.model.slack_max = Var(self.model.n_gen, domain=NonNegativeReals)

    def _build_objective(self, lin = True):
        if lin == False:
            self.model.objective = Objective(
                expr=sum(
                    -self.model.lambda_dual * self.model.P_G[i] + self.cost[i] * self.model.P_G[i]
                    for i in self.model.strategic_index
                ),
            sense=minimize
        )
        else:
            # Strong duality substitution
            dual_costs = (
                self.model.lambda_dual * self.demand
                + sum(self.model.mu_min[i] * self.Pmin[i] for i in self.model.n_gen)
                - sum(self.model.mu_max[i] * self.Pmax[i] for i in self.model.n_gen)
            )

            non_strat_costs = sum(
                self.cost[i] * self.model.P_G[i]
                for i in self.model.n_gen - self.model.strategic_index
            )

            strat_term1 = sum(-self.model.mu_min[i] * self.Pmin[i] 
                              + self.model.mu_max[i] * self.Pmax[i] for i in self.model.strategic_index)

            strat_term2 = sum(
                self.cost[i] * self.model.P_G[i]
                for i in self.model.strategic_index
            )

            self.model.objective = Objective(
                expr= - (dual_costs - non_strat_costs + strat_term1) + strat_term2,
                sense=minimize
            )


    def _build_constraints(self, SOS1 = False):
        
        # Alpha constraints
        self.model.alpha_constraint_min = Constraint(expr=self.model.alpha >= self.alpha_min)
        self.model.alpha_constraint_max = Constraint(expr=self.model.alpha <= self.alpha_max)

        # Power balance constraint
        self.model.power_balance = Constraint(expr=sum(self.model.P_G[i] for i in range(self.num_generators)) == self.demand)

        def stationarity_rule(m, i):
            return m.alpha - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0

        self.model.stationarity = Constraint(self.model.strategic_index, rule=stationarity_rule)

        def stationarity_non_strategic_rule(m, i):
            return self.cost[i] - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0
        
        self.model.stationarity_non_strategic = Constraint(
            self.model.n_gen - self.model.strategic_index, rule=stationarity_non_strategic_rule
        )
        if SOS1 is False:
            # ------------------------
            # Big-M + binary formulation
            # ------------------------
            M = 10000

            def tau_rule_lower(m, i):
                return m.alpha <= self.cost[i] * 0.999 + M * m.tau[i]

            def tau_rule_upper(m, i):
                return m.alpha >= self.cost[i] * 1.001 - M * (1 - m.tau[i])

            self.model.tau_lower = Constraint(self.model.n_gen - self.model.strategic_index, rule=tau_rule_lower)
            self.model.tau_upper = Constraint(self.model.n_gen - self.model.strategic_index, rule=tau_rule_upper)

            #self.model.tau_sum = Constraint(expr=sum(self.model.tau[i] for i in self.model.n_gen - self.model.strategic_index) <= len(self.cost) - 1)

            # min bound
            def gen_min_lower_rule(m, i):
                return m.P_G[i] - self.Pmin[i] >= 0
            self.model.gen_min_lower = Constraint(self.model.n_gen, rule=gen_min_lower_rule)

            def gen_min_upper_rule(m, i):
                return m.P_G[i] - self.Pmin[i] <= M * m.z_min[i]
            self.model.gen_min_upper = Constraint(self.model.n_gen, rule=gen_min_upper_rule)

            def mu_min_lower_rule(m, i):
                return m.mu_min[i] >= 0
            self.model.mu_min_lower = Constraint(self.model.n_gen, rule=mu_min_lower_rule)

            def mu_min_upper_rule(m, i):
                return m.mu_min[i] <= M * (1 - m.z_min[i])
            self.model.mu_min_upper = Constraint(self.model.n_gen, rule=mu_min_upper_rule)

            # max bound
            def gen_max_lower_rule(m, i):
                return self.Pmax[i] - m.P_G[i] >= 0
            self.model.gen_max_lower = Constraint(self.model.n_gen, rule=gen_max_lower_rule)

            def gen_max_upper_rule(m, i):
                return self.Pmax[i] - m.P_G[i] <= M * m.z_max[i]
            self.model.gen_max_upper = Constraint(self.model.n_gen, rule=gen_max_upper_rule)

            def mu_max_upper_rule(m, i):
                return m.mu_max[i] <= M * (1 - m.z_max[i])
            self.model.mu_max_upper = Constraint(self.model.n_gen, rule=mu_max_upper_rule)

            def mu_max_lower_rule(m, i):
                return m.mu_max[i] >= 0
            self.model.mu_max_lower = Constraint(self.model.n_gen, rule=mu_max_lower_rule)
        
        else:
           # Link primal bounds
            def gen_min_rule(m, i):
                return m.P_G[i] - self.Pmin[i] == m.slack_min[i]
            self.model.gen_min = Constraint(self.model.n_gen, rule=gen_min_rule)

            def gen_max_rule(m, i):
                return self.Pmax[i] - m.P_G[i] == m.slack_max[i]
            self.model.gen_max = Constraint(self.model.n_gen, rule=gen_max_rule)

            # Dual feasibility
            def mu_min_nonneg(m, i): return m.mu_min[i] >= 0
            def mu_max_nonneg(m, i): return m.mu_max[i] >= 0
            self.model.mu_min_nonneg = Constraint(self.model.n_gen, rule=mu_min_nonneg)
            self.model.mu_max_nonneg = Constraint(self.model.n_gen, rule=mu_max_nonneg)

            # SOS1 complementarity: build tiny indexed Var per generator
            for i in self.model.n_gen:
                # Create a 2-element Var to hold [slack, mu]
                vpair_min = Var([0, 1])
                vpair_max = Var([0, 1])
                self.model.add_component(f"sos1pair_min_{i}", vpair_min)
                self.model.add_component(f"sos1pair_max_{i}", vpair_max)

                # Link to real variables
                self.model.add_component(f"link_min_slack_{i}",
                    Constraint(expr=vpair_min[0] == self.model.slack_min[i]))
                self.model.add_component(f"link_min_mu_{i}",
                    Constraint(expr=vpair_min[1] == self.model.mu_min[i]))

                self.model.add_component(f"link_max_slack_{i}",
                    Constraint(expr=vpair_max[0] == self.model.slack_max[i]))
                self.model.add_component(f"link_max_mu_{i}",
                    Constraint(expr=vpair_max[1] == self.model.mu_max[i]))

                # Add SOS1
                self.model.add_component(f"sos1_min_{i}", SOSConstraint(var=vpair_min, sos=1))
                self.model.add_component(f"sos1_max_{i}", SOSConstraint(var=vpair_max, sos=1))

    def solve(self, solver_name="gurobi"):
        """
        Solve the optimization model.

        Parameters
        ----------
        solver_name : str, optional
            Name of the solver to use (default: "gurobi").
        tee : bool, optional
            If True, prints solver log output.
        """

        # Build model components if not already built
        if not hasattr(self, "model"):
            self._build_model()
            self._build_variables()
            self._build_constraints()

        # Create solver
        solver = SolverFactory(solver_name)

        # Solve
        results = solver.solve(self.model, tee=False)

        # Check solver status
        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            print("Optimal solution found for strategic producer problem.")
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

        # Display results
        # self._display_results()
    
    def _display_results(self):
        print("\nOptimal Generation and Prices:")
        for i in self.model.n_gen:
            print(f"Generator {i}: P_G = {self.model.P_G[i].value:.2f}, mu_min = {self.model.mu_min[i].value:.2f}, mu_max = {self.model.mu_max[i].value:.2f}")
        print(f"Market Price (lambda): {self.model.lambda_dual.value:.2f}")
        print(f"Strategic Producer's Bid (alpha): {self.model.alpha.value:.2f}")
        print(f"Objective Value (Profit): {-self.model.objective():.2f}")

    def economic_dispatch(self):
        """
        Solve the economic dispatch problem (non-strategic).
        """
        model = ConcreteModel()
        model.n_gen = Set(initialize=range(self.num_generators))
        model.P_G = Var(model.n_gen, domain=NonNegativeReals)
        model.objective = Objective(
            expr=sum(self.cost[i] * model.P_G[i] for i in model.n_gen),
            sense=minimize
        )
        model.power_balance = Constraint(expr=self.demand - sum(model.P_G[i] for i in model.n_gen) == 0)

        model.gen_min = Constraint(model.n_gen, rule=lambda m, i: model.P_G[i] >= self.Pmin[i])
        model.gen_max = Constraint(model.n_gen, rule=lambda m, i: model.P_G[i] <= self.Pmax[i])

        # Attach suffix to capture duals
        model.dual = Suffix(direction=Suffix.IMPORT)

        solver = SolverFactory("gurobi")
        results = solver.solve(model, tee=False)
        if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
            print("Optimal solution found for economic dispatch.")
            dispatch = {i: model.P_G[i].value for i in model.n_gen}
            clearing_price = -model.dual[model.power_balance]
            return dispatch, clearing_price
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    def plot_merit_order(self, dispatch, clearing_price):
        cost_array = np.array(self.cost)
        pmax_array = np.array(self.Pmax)

        # Sort generators by cost
        gen_sorted_idx = np.argsort(cost_array)
        gen_sorted_costs = cost_array[gen_sorted_idx]
        gen_sorted_caps = pmax_array[gen_sorted_idx]

        plt.figure(figsize=(10, 6))


        # Build stepwise supply curve
        gen_curve_x = [0]
        gen_curve_y = [0]
        cum_cap = 0
        
        for idx, (c, cap) in zip(gen_sorted_idx, zip(gen_sorted_costs, gen_sorted_caps)):
            # horizontal start
            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)
            # vertical end
            cum_cap += cap
            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)

            # --- annotate generator index ---
            midpoint = cum_cap - cap / 2
            plt.text(midpoint, c - 0.1, f"G{idx}", ha='center', va='top', fontsize=9, color="blue")

        # --- Demand (inelastic vertical line) ---
        demand = self.demand   # e.g. 200
        demand_curve_x = [demand, demand]
        demand_curve_y = [0, max(gen_curve_y) * 1.2]

        # --- Plot ---
        
        plt.step(gen_curve_x, gen_curve_y, where='post', label='Supply (Generators)', color='blue')
        plt.plot(demand_curve_x, demand_curve_y, 'r--', label=f'Demand = {demand}')

        # Mark clearing price
        plt.axhline(y=-clearing_price, color='green', linestyle='--',
                    label=f'Clearing Price = {-clearing_price:.2f}')
        plt.scatter([demand], [-clearing_price], color='black', zorder=5, label='Market Clearing Point')

        plt.ylim(bottom=0, top=max(gen_curve_y) * 1.2)
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.title('Merit Order Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_merit_order_with_strategic(self, dispatch_ED, clearing_price_ED,
                                    dispatch_SP, clearing_price_SP):
        cost_array = np.array(self.cost)
        pmax_array = np.array(self.Pmax)

        # --- Economic Dispatch (baseline merit order) ---
        gen_sorted_idx = np.argsort(cost_array)
        gen_sorted_costs = cost_array[gen_sorted_idx]
        gen_sorted_caps = pmax_array[gen_sorted_idx]

        plt.figure(figsize=(10, 6))

        gen_curve_x = [0]
        gen_curve_y = [0]
        cum_cap = 0
        for idx, (c, cap) in zip(gen_sorted_idx, zip(gen_sorted_costs, gen_sorted_caps)):
            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)
            cum_cap += cap
            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)

            # Label under ED supply line
            midpoint = cum_cap - cap / 2
            plt.text(midpoint, c - 0.2, f"G{idx}", ha='center', va='top', fontsize=8, color="blue")

        # Plot ED supply curve
        plt.step(gen_curve_x, gen_curve_y, where='post', color='blue', label='Supply (ED)')

        # --- Strategic Producer Case ---
        sp_costs = cost_array.copy()
        # Replace strategic producer's cost with its chosen bid (alpha)
        for i in self.model.strategic_index:
            sp_costs[i] = self.model.alpha.value 
            print(self.model.alpha.value)
        print(sp_costs)
        gen_sorted_idx_SP = np.argsort(sp_costs)
        print(gen_sorted_idx_SP)
        gen_sorted_costs_SP = sp_costs[gen_sorted_idx_SP]
        gen_sorted_caps_SP = pmax_array[gen_sorted_idx_SP]

        gen_curve_x_SP = [0]
        gen_curve_y_SP = [0]
        cum_cap_SP = 0
        for idx, (c, cap) in zip(gen_sorted_idx_SP, zip(gen_sorted_costs_SP, gen_sorted_caps_SP)):
            gen_curve_x_SP.append(cum_cap_SP)
            gen_curve_y_SP.append(c)
            cum_cap_SP += cap
            gen_curve_x_SP.append(cum_cap_SP)
            gen_curve_y_SP.append(c)

            # Label under SP supply line
            midpoint_SP = cum_cap_SP - cap / 2
            plt.text(midpoint_SP, c - 0.2, f"G{idx}", ha='center', va='top', fontsize=8, color="purple")

        # Plot SP supply curve
        plt.step(gen_curve_x_SP, gen_curve_y_SP, where='post', color='purple', linestyle='--', label='Supply (SP)')

        # --- Demand ---
        demand = self.demand
        plt.axvline(demand, color='red', linestyle='--', label=f'Demand = {demand}')

        # --- Clearing prices ---
        plt.scatter([demand], [clearing_price_ED], color='green', zorder=5, marker='o', label=f'ED Price = {clearing_price_ED:.2f}', s = 100)
        plt.scatter([demand], [clearing_price_SP], color='magenta', zorder=5, marker='x', label=f'SP Price = {clearing_price_SP:.2f}', s = 100)

        # --- Formatting ---
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.title('Merit Order Curve: ED vs Strategic Producer')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Print comparison
        print("Dispatch Comparison (ED vs SP):")
        print("  Gen   Cost(ED)  Cost(SP)   ED [MW]   SP [MW]")
        print("  ---  --------- ---------  --------  --------")

        for i in range(len(cost_array)):
            # In SP case, replace strategic cost with alpha
            cost_sp = self.model.alpha.value if i in self.model.strategic_index else self.cost[i]

            line = (
                f"  {i:2d}   "
                f"{self.cost[i]:9.2f} "
                f"{cost_sp:9.2f} "
                f"{dispatch_ED[i]:8.2f}  "
                f"{dispatch_SP[i]:8.2f}"
            )
            print(line)

        print()
        print(f"  Strategic producer bid (alpha): {self.model.alpha.value:8.2f}")
        print(f"  Profit from strategic bidding : {(clearing_price_SP - self.cost[0]) * dispatch_SP[0]:8.2f}")
        print(f"  Clearing price (ED)           : {clearing_price_ED:8.2f}")
        print(f"  Clearing price (SP)           : {clearing_price_SP:8.2f}")
        print()



alpha_min = -400
alpha_max = 5000
Pmin = [0, 0, 0,0]
Pmax = [60,80,55,75]
#cost = [2.0, 3.0, 4.0,5.0 ]
cost = [2, 3*1.5, 4*1.5,5*1.5 ]
demand = 150
SOS1 = False  # Set to True to use SOS1 formulation, False for Big-M

problem = StrategicProducer(alpha_min, alpha_max, Pmin, Pmax, cost, demand, SOS1)

# Baseline competitive economic dispatch
dispatch_ED, clearing_price_ED = problem.economic_dispatch()

# Strategic producer problem
dispatch_SP = {i: problem.model.P_G[i].value for i in problem.model.n_gen}
clearing_price_SP = problem.model.lambda_dual.value

problem.plot_merit_order_with_strategic(dispatch_ED, clearing_price_ED,
                                        dispatch_SP, clearing_price_SP)