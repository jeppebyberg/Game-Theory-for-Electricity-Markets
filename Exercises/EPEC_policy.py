from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
import itertools

class EPEC:
    def __init__(self, 
                 alpha_min, alpha_max, 
                 Pmin, Pmax, 
                 demand,
                 cost_min, cost_max, segments,
                 max_iter = 100, convergence_tol = 0.01):

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.Pmin = Pmin
        self.Pmax = Pmax
        
        self.demand = demand

        self.cost_min = cost_min
        self.cost_max = cost_max
        self.segments = segments
            
        self.max_iter = max_iter
        self.convergence_tol = convergence_tol

        self.num_generators = len(Pmin)

        self.cost = np.array([
            np.linspace(self.cost_min[i], self.cost_max[i], self.segments)
            for i in range(self.num_generators)
        ])
        
        self.results = {}

    def iterate_cost_combinations(self):
        # all combinations of cost vectors (Cartesian product)
        all_combinations = list(itertools.product(*self.cost))

        print(f"Total combinations to evaluate: {len(all_combinations)}")

        for run_id, init_cost_vector in enumerate(all_combinations):
            init_cost_vector = np.array(init_cost_vector)
            profits, alphas, dispatches, iterations, PoA, dispatch_ED, clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, clearing_price_history, weight_history = self.run_best_response(init_cost_vector, run_id)

            self.results[run_id] = {
                "init_cost_vector": init_cost_vector,
                "profit_history": profits,
                "alpha_history": alphas,
                "dispatch_history": dispatches,
                "iterations": iterations,
                "PoA": PoA,
                "final_dispatch": final_dispatch,
                "final_bid": final_bid,
                "clearing_price": clearing_price_SP,
                "clearing_price_history": clearing_price_history,
                "dispatch_ED": dispatch_ED,
                "clearing_price_ED": clearing_price_ED,
                "weight_history": weight_history

            }
        worst_id, worst_poa = max(
            ((id, res['PoA']) for id, res in epec.results.items()),
            key=lambda x: x[1]
        )

        print(f"Worst PoA: {worst_poa:.2f} (from run {worst_id})")

    def run_best_response(self, init_cost_vector, run_id):
        # reset histories for this run
        profit_history = []
        alpha_history = []
        dispatch_history = []
        convergence_check = []
        clearing_price_history = []
        weight_history = []

        dispatch_ED, clearing_price_ED, minimum_cost_ED = epec.economic_dispatch(init_cost_vector)

        iter = 0
        cost_vector = init_cost_vector.copy()

        # Best-response iterations
        while iter <= self.max_iter:
            profit_history.append([None] * self.num_generators)
            alpha_history.append([None] * self.num_generators)
            dispatch_history.append([None] * self.num_generators)
            convergence_check.append([False] * self.num_generators)
            weight_history.append([None] * self.num_generators)

            for i in range(self.num_generators):
                self._build_model(i, cost_vector)
                self.solve()
                cost_vector[i] = self.model.alpha.value
                profit_history[iter][i] = -self.model.objective()
                alpha_history[iter][i] = self.model.alpha.value
                dispatch_history[iter] = [self.model.P_G[ii].value for ii in self.model.n_gen]
                clearing_price_SP = self.model.lambda_dual.value
                weight_history[iter][i] = [self.model.omega[ii].value for ii in self.model.n_gen - self.model.strategic_index]

                if iter > 0:
                    if profit_history[iter][i] >= (1 - self.convergence_tol) * profit_history[iter - 1][i] and profit_history[iter][i] <= (1 + self.convergence_tol) * profit_history[iter - 1][i]:
                        convergence_check[iter][i] = True

            clearing_price_history.append(clearing_price_SP)

            if all(convergence_check[iter]):
                print(f"Run id: {run_id} - Converged after {iter} iterations.")
                PoA = clearing_price_SP * self.demand / minimum_cost_ED
                final_bid = cost_vector.copy()
                final_dispatch = dispatch_history[iter]
                break
            if iter == self.max_iter:
                print(f"Run id: {run_id} - Reached max iterations - {self.max_iter}.")
                PoA = clearing_price_SP * self.demand / minimum_cost_ED
                
                # Compute mean of last 10 dispatches 
                dispatch_array = np.array(dispatch_history[-10:])
                mean_dispatch = np.mean(dispatch_array, axis=0)
                final_dispatch = mean_dispatch

                bid_array = np.array(alpha_history[-10:])
                mean_bid = np.mean(bid_array, axis=0)
                final_bid = mean_bid
                break
            iter += 1
        return profit_history, alpha_history, dispatch_history, iter, PoA, dispatch_ED, clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, clearing_price_history, weight_history

    def _build_model(self, index_strategic, cost_vector):
        self.model = ConcreteModel()

        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.strategic_index = Set(initialize=[index_strategic])  # Index of the strategic producer

        self._build_variables()
        self._build_objective(cost_vector)
        self._build_constraints(cost_vector)
        self._build_policy_constraints(cost_vector)

    def _build_variables(self):
        self.model.P_G = Var(self.model.n_gen, domain=Reals)
        self.model.alpha = Var(domain=Reals)
        self.model.lambda_dual = Var(domain=Reals)
        self.model.mu_min = Var(self.model.n_gen, domain=Reals)
        self.model.mu_max = Var(self.model.n_gen, domain=Reals)
        self.model.z_min = Var(self.model.n_gen, domain=Binary)
        self.model.z_max = Var(self.model.n_gen, domain=Binary)
        self.model.tau = Var(self.model.n_gen, domain=Binary)
        self.model.omega = Var(self.model.n_gen - self.model.strategic_index, domain=Reals)

    def _build_objective(self, cost_vector, lin = False):
        if lin == False:
            self.model.objective = Objective(
                expr=sum(
                    -self.model.lambda_dual * self.model.P_G[i] + cost_vector[i] * self.model.P_G[i]
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

            non_strat_costs = sum(cost_vector[i] * self.model.P_G[i]
                                for i in self.model.n_gen - self.model.strategic_index)

            strat_bounds = sum(self.model.mu_min[i] * self.Pmin[i] + self.model.mu_max[i] * self.Pmax[i]
                            for i in self.model.strategic_index)

            strat_costs = sum(cost_vector[i] * self.model.P_G[i]
                            for i in self.model.strategic_index)

            self.model.objective = Objective(
                expr= - (dual_costs - non_strat_costs - strat_bounds) + strat_costs,
                sense=minimize
            )

    def _build_constraints(self, cost_vector):
        # Alpha constraints
        self.model.alpha_constraint_min = Constraint(expr=self.model.alpha >= self.alpha_min)
        self.model.alpha_constraint_max = Constraint(expr=self.model.alpha <= self.alpha_max)

        # Power balance constraint
        self.model.power_balance = Constraint(expr=sum(self.model.P_G[i] for i in range(self.num_generators)) == self.demand)

        def stationarity_rule(m, i):
            return m.alpha - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0

        self.model.stationarity = Constraint(self.model.strategic_index, rule=stationarity_rule)

        def stationarity_non_strategic_rule(m, i):
            return cost_vector[i] - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0

        self.model.stationarity_non_strategic = Constraint(
            self.model.n_gen - self.model.strategic_index, rule=stationarity_non_strategic_rule
        )
        # ------------------------
        # Big-M + binary formulation
        # ------------------------
        M = 1000

        def tau_rule_lower(m, i):
            return m.alpha <= cost_vector[i] * 0.999 + M * m.tau[i]

        def tau_rule_upper(m, i):
            return m.alpha >= cost_vector[i] * 1.001 - M * (1 - m.tau[i])

        self.model.tau_lower = Constraint(self.model.n_gen - self.model.strategic_index, rule=tau_rule_lower)
        self.model.tau_upper = Constraint(self.model.n_gen - self.model.strategic_index, rule=tau_rule_upper)

        self.model.tau_sum = Constraint(expr=sum(self.model.tau[i] for i in self.model.n_gen - self.model.strategic_index) <= len(cost_vector) - 1)

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
    
    def _build_policy_constraints(self, cost_vector):
        # Create arbitrary policy constraints
        def policy_rule_1(m):
            return m.alpha == sum(m.omega[i] * cost_vector[i] * self.Pmax[i] for i in self.model.n_gen - self.model.strategic_index)

        self.model.policy_1 = Constraint(rule=policy_rule_1)

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

        # Create solver
        solver = SolverFactory(solver_name)

        # Solve
        results = solver.solve(self.model, tee=False)

        # Check solver status
        if not (results.solver.status == 'ok') and not (results.solver.termination_condition == 'optimal'):
            # print("Optimal solution found for strategic producer problem.")
        # else:
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

    def economic_dispatch(self, init_cost_vector):
        """
        Solve the economic dispatch problem (non-strategic).
        """
        model = ConcreteModel()
        model.n_gen = Set(initialize=range(self.num_generators))
        model.P_G = Var(model.n_gen, domain=NonNegativeReals)
        model.objective = Objective(
            expr=sum(init_cost_vector[i] * model.P_G[i] for i in model.n_gen),
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
            # print("Optimal solution found for economic dispatch.")
            dispatch = [model.P_G[i].value for i in model.n_gen]
            clearing_price = -model.dual[model.power_balance]
            minimum_cost = sum(init_cost_vector[i] * dispatch[i] for i in model.n_gen)
            return dispatch, clearing_price, minimum_cost
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    def plot_merit_order_curve(self, run_id):

        init_cost_vector = self.results[run_id]['init_cost_vector']
        cost_vector = self.results[run_id]['final_bid']
        dispatch_ED = self.results[run_id]['dispatch_ED']
        clearing_price_ED = self.results[run_id]['clearing_price_ED']
        dispatch_SP = self.results[run_id]['final_dispatch']
        clearing_price_SP = self.results[run_id]['clearing_price']

        cost_array = np.array(init_cost_vector)
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
        sp_costs = np.array(cost_vector)

        gen_sorted_idx_SP = np.argsort(sp_costs)
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
            cost_sp = sp_costs[i]

            line = (
                f"  {i:2d}   "
                f"{init_cost_vector[i]:9.2f} "
                f"{cost_sp:9.2f} "
                f"{dispatch_ED[i]:8.2f}  "
                f"{dispatch_SP[i]:8.2f}"
            )
            print(line)

        print()
        print(f"  Clearing price (ED)           : {clearing_price_ED:8.2f}")
        print(f"  Clearing price (SP)           : {clearing_price_SP:8.2f}")
        print()

    def plot_alpha_over_iterations(self, run_id):
        alpha_history = self.results[run_id]['alpha_history']

        init_cost_vector = self.results[run_id]['init_cost_vector']

        alpha_history = np.array(alpha_history)
        plt.figure(figsize=(10, 6))
        for i in range(self.num_generators):
            plt.plot(alpha_history[:, i], marker='o', label=f'Generator {i} - Init Cost {init_cost_vector[i]:.0f}')
        plt.xlabel('Iteration')
        plt.ylabel('Alpha (Bid)')
        plt.title('Alpha Evolution Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_clearing_price_over_iterations(self, run_id):
        clearing_price_history = self.results[run_id]['clearing_price_history']

        clearing_price_history = np.array(clearing_price_history)
        plt.figure(figsize=(10, 6))
        plt.plot(clearing_price_history, marker='o', label=f'Clearing Price')
        plt.xlabel('Iteration')
        plt.ylabel('Clearing Price')
        plt.title('Clearing Price Evolution Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_dispatch_over_iterations(self, run_id):
        dispatch_history = self.results[run_id]['dispatch_history']
        economic_dispatch = self.results[run_id]['dispatch_ED']

        init_cost_vector = self.results[run_id]['init_cost_vector']

        economic_dispatch = np.array([economic_dispatch] * len(dispatch_history))
        dispatch_history = np.array(dispatch_history)
        
        plt.figure(figsize=(10, 6))
        for i in range(self.num_generators):
            plt.plot(dispatch_history[:, i], marker='o', label=f'Generator {i} - Init Cost {init_cost_vector[i]:.2f}')
            if i == 0:
                plt.plot(economic_dispatch[:, i], linestyle='--', color='black', label='Economic Dispatch')
            else:
                plt.plot(economic_dispatch[:, i], linestyle='--', color='black')
        plt.xlabel('Iteration')
        plt.ylabel('Dispatch (MW)')
        plt.title('Dispatch Evolution Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_PoA(self):
        PoA_values = [self.results[run_id]['PoA'] for run_id in self.results]
        plt.figure(figsize=(8, 5))
        plt.hist(PoA_values, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Price of Anarchy (PoA)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Price of Anarchy Across Runs')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_weights(self, run_id):

        weight_history = self.results[run_id]['weight_history']

        n_iter = len(weight_history)                     # total iterations
        n_players = len(self.model.n_gen)                # total generators (all are strategic in EPEC)

        fig, axes = plt.subplots(1, n_players, figsize=(5*n_players, 4), sharey=True)

        if n_players == 1:
            axes = [axes]  # make iterable

        for sp_idx, ax in enumerate(axes):
            # Collect weights from history for this strategic player
            weights_over_time = [weight_history[it][sp_idx] for it in range(n_iter)]
            weights_over_time = np.array(weights_over_time)  # shape: (iterations, n_price_takers)

            # Identify which players are the price takers in this problem
            price_takers = [p for p in range(n_players) if p != sp_idx]

            # Plot each price taker’s weight trajectory
            for pt_idx, pt in enumerate(price_takers):
                ax.plot(range(n_iter), weights_over_time[:, pt_idx], label=f"Player {pt}")

            ax.set_title(f"Strategic player {sp_idx}")
            ax.set_xlabel("Iteration")
            if sp_idx == 0:
                ax.set_ylabel("Weight value")
            ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    
    alpha_min = -100
    alpha_max = 100
    
    Pmin = [0, 0, 0]
    Pmax = [50, 55, 60]

    cost_min = [20, 30, 40]
    cost_max = [60, 70, 80]

    segments = 2 

    max_iter = 50   
    demand = 95

    convergence_tol = 0.01

    epec = EPEC(alpha_min, alpha_max, 
                Pmin, Pmax, demand, 
                cost_min, cost_max, 
                segments, 
                max_iter, convergence_tol)

    epec.iterate_cost_combinations()
    # epec.plot_clearing_price_over_iterations(run_id = 0)
    # epec.plot_alpha_over_iterations(run_id = 0)
    # epec.plot_dispatch_over_iterations(run_id = 0)
    epec.plot_merit_order_curve(run_id = 0)
    # epec.plot_weights(run_id = 0)
    # epec.plot_PoA()

    # print("Omega[0]:", epec.model.omega[0].value)
    # print("Omega[1]:", epec.model.omega[1].value)
    # print("Final Bid[0]:", epec.results[26]['final_bid'][0])
    # print("Final Bid[1]:", epec.results[26]['final_bid'][1])
    # print("Pmax[0]:", epec.Pmax[0])
    # print("Pmax[1]:", epec.Pmax[1])
    # print("Policy Check:", epec.model.alpha.value, "==", epec.model.omega[0].value * epec.results[26]['final_bid'][0] * epec.Pmax[0] + epec.model.omega[1].value * epec.results[26]['final_bid'][1] * epec.Pmax[1])
