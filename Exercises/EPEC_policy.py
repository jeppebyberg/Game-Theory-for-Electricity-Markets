from pyomo.environ import *
from pyomo.core.base.var import IndexedVar
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

class EPEC:
    def __init__(self, 
                 alpha_min, alpha_max, 
                 Pmin, Pmax, 
                 demand,
                 cost_min, cost_max, segments,
                 cost_ownership,
                 max_iter = 100, convergence_tol = 0.01):

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.Pmin = Pmin
        self.Pmax = Pmax
        
        self.demand = demand

        self.cost_min = cost_min
        self.cost_max = cost_max
        self.segments = segments
        self.cost_ownership = cost_ownership

        self.max_iter = max_iter
        self.convergence_tol = convergence_tol

        self.num_generators = len(Pmin)

        self.cost = np.array([
            np.linspace(self.cost_min[i], self.cost_max[i], self.segments)
            for i in range(self.num_generators)
        ])
        
        self.results = {}

    def iterate_cost_combinations(self):
        self.ownership_mode = False
        # all combinations of cost vectors (Cartesian product)
        all_combinations = list(itertools.product(*self.cost))

        print(f"Total combinations to evaluate: {len(all_combinations)}")
        total_runs = len(all_combinations)
        converged_runs = 0

        for run_id, init_cost_vector in enumerate(all_combinations):
            init_cost_vector = np.array(init_cost_vector)
            (profits, alphas, dispatches, iterations, PoA, dispatch_ED, 
             clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, 
             clearing_price_history, weight_history, converged, not_converged
             ) = self.run_best_response(init_cost_vector, run_id)

            if converged:
                converged_runs += 1

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
                "weight_history": weight_history,
                "converged": converged,
                "not_converged": not_converged,
            }

        share_converged = 100 * converged_runs / total_runs if total_runs > 0 else 0.0
        print(f"Converged in {converged_runs}/{total_runs} runs ({share_converged:.1f}%)")
    
        worst_id, worst_poa = max(
            ((id, res['PoA']) for id, res in self.results.items()),
            key=lambda x: x[1]
        )

        print(f"Worst PoA: {worst_poa:.2f} (from run id {worst_id})")
        return share_converged, worst_poa

    def iterate_ownership_combinations(self, ownership_size = 2):
        # Ownership combinations
        all_combinations = list(itertools.combinations(range(self.num_generators), ownership_size))

        print(all_combinations)
        print(f"Total combinations to evaluate: {len(all_combinations)}")
        total_runs = len(all_combinations)
        converged_runs = 0

        for run_id, owner_indexes in enumerate(all_combinations):
            (profits, alphas, dispatches, iterations, PoA, dispatch_ED, 
             clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, 
             clearing_price_history, weight_history, converged, not_converged
             ) = self.run_best_response_ownership(owner_indexes, run_id)
            if converged:
                converged_runs += 1

            self.results[run_id] = {
                "init_cost_vector": self.cost_ownership,
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
                "weight_history": weight_history,
                "converged": converged,
                "not_converged": not_converged,
                "owner_indexes": owner_indexes,
            }

        share_converged = 100 * converged_runs / total_runs if total_runs > 0 else 0.0
        print(f"Converged in {converged_runs}/{total_runs} runs ({share_converged:.1f}%)")
    
        worst_id, worst_poa = max(
            ((id, res['PoA']) for id, res in self.results.items()),
            key=lambda x: x[1]
        )

        print(f"Worst PoA: {worst_poa:.2f} (from run id {worst_id})")
        return share_converged, worst_poa

    def run_best_response(self, init_cost_vector, run_id):
        profit_history, alpha_history, dispatch_history = [], [], []
        convergence_check, clearing_price_history, weight_history = [], [], []

        dispatch_ED, clearing_price_ED, minimum_cost_ED = self.economic_dispatch(init_cost_vector)
        iter = 0
        cost_vector = init_cost_vector.copy()
        converged = False
        
        random.seed(run_id)
        
        while iter < self.max_iter:
            profit_history.append([None] * self.num_generators)
            alpha_history.append([None] * self.num_generators)
            dispatch_history.append([None] * self.num_generators)
            convergence_check.append([False] * self.num_generators)
            weight_history.append([None] * self.num_generators)


            ######## Jacobi Method ########
            # # --- Compute new best responses ---
            # new_cost_vector = cost_vector.copy()
            # for i in range(self.num_generators):
            #     self._build_model(i, cost_vector, init_cost_vector)
            #     self.solve()
            #     alpha_new = self.model.alpha.value

            #     # --- Enforce rational bidding: never below marginal cost ---
            #     alpha_new = max(alpha_new, init_cost_vector[i])
            #     new_cost_vector[i] = alpha_new

            # # --- Apply relaxation (damping) ---
            # lambda_relax = min(0.3 + 0.01*iter, 1.0)
            # print(f"Iteration {iter}: Relaxation factor = {lambda_relax:.2f}")
            # cost_vector = (1 - lambda_relax) * np.array(cost_vector) + lambda_relax * np.array(new_cost_vector)
            # cost_vector = cost_vector.tolist()

            players = list(range(self.num_generators))
            random.shuffle(players)                    # randomize update order each iteration

            for p in players:
                self._build_model(p, cost_vector, init_cost_vector)
                self.solve()
                for g in (p if isinstance(p, (list, tuple, set)) else [p]):
                    alpha_new = self.model.alpha[g].value
                    alpha_new = max(alpha_new, init_cost_vector[g])   # rational floor
                    cost_vector[g] = alpha_new
                    alpha_history[iter][g] = alpha_new

            # --- Market clearing (global consistency) ---
            dispatch_round, clearing_price_round, _ = self.economic_dispatch(cost_vector)
            dispatch_history[iter] = dispatch_round
            clearing_price_history.append(clearing_price_round)

            for j in range(self.num_generators):
                profit_history[iter][j] = (
                    clearing_price_round * dispatch_round[j]
                    - init_cost_vector[j] * dispatch_round[j]
                )

            # --- Convergence check ---
            if iter > 5:
                for j in range(self.num_generators):
                    prev_profit = profit_history[iter - 1][j]
                    curr_profit = profit_history[iter][j]
                    # curr_clearing_price = clearing_price_history[-1]
                    # prev_clearing_price = clearing_price_history[-2]
                    curr_bid = alpha_history[iter][j]
                    prev_bid = alpha_history[iter - 1][j]
                    if (
                        curr_profit >= prev_profit * (1 - self.convergence_tol)
                        and curr_profit <= prev_profit * (1 + self.convergence_tol)
                        # and curr_clearing_price >= prev_clearing_price * (1 - self.convergence_tol)
                        # and curr_clearing_price <= prev_clearing_price * (1 + self.convergence_tol)
                        and curr_bid >= prev_bid * (1 - self.convergence_tol)
                        and curr_bid <= prev_bid * (1 + self.convergence_tol)
                    ):
                        convergence_check[iter][j] = True
                    else:
                        convergence_check[iter][j] = False

                if all(convergence_check[iter]):
                    print(f"Run id: {run_id} - Converged after {iter} full rounds.")
                    converged = True
                    break
            if iter == self.max_iter:
                print(f"Run id: {run_id} - Reached maximum iterations {self.max_iter} without convergence.")

            iter += 1

        # --- Final consistent results ---
        PoA = clearing_price_round * self.demand / minimum_cost_ED
        final_bid = cost_vector.copy()
        final_dispatch = dispatch_round

        not_converged = not converged

        return (
            profit_history,
            alpha_history,
            dispatch_history,
            iter,
            PoA,
            dispatch_ED,
            clearing_price_ED,
            final_dispatch,
            final_bid,
            clearing_price_round,
            clearing_price_history,
            weight_history,
            converged,
            not_converged,
        )

    def run_best_response_ownership(self, owner_indexes, run_id):
        profit_history, alpha_history, dispatch_history = [], [], []
        convergence_check, clearing_price_history, weight_history = [], [], []

        init_cost_vector = self.cost_ownership

        dispatch_ED, clearing_price_ED, minimum_cost_ED = self.economic_dispatch(init_cost_vector)
        iter = 0
        cost_vector = init_cost_vector.copy()
        converged = False
        
        random.seed(run_id)
        
        while iter <= self.max_iter:
            profit_history.append([None] * self.num_generators)
            alpha_history.append([None] * self.num_generators)
            dispatch_history.append([None] * self.num_generators)
            convergence_check.append([False] * self.num_generators)
            weight_history.append([None] * self.num_generators)


            ######## Jacobi Method ########
            # # --- Compute new best responses ---
            # new_cost_vector = cost_vector.copy()
            # for i in range(self.num_generators):
            #     self._build_model(i, cost_vector, init_cost_vector)
            #     self.solve()
            #     alpha_new = self.model.alpha.value

            #     # --- Enforce rational bidding: never below marginal cost ---
            #     alpha_new = max(alpha_new, init_cost_vector[i])
            #     new_cost_vector[i] = alpha_new

            # # --- Apply relaxation (damping) ---
            # lambda_relax = min(0.3 + 0.01*iter, 1.0)
            # print(f"Iteration {iter}: Relaxation factor = {lambda_relax:.2f}")
            # cost_vector = (1 - lambda_relax) * np.array(cost_vector) + lambda_relax * np.array(new_cost_vector)
            # cost_vector = cost_vector.tolist()

            # competitors = all generators not owned
            competitors = [i for i in range(self.num_generators) if i not in owner_indexes]

            # Create player update list
            players = [owner_indexes] + competitors

            random.shuffle(players)                    # randomize update order each iteration

            for p in players:
                self._build_model(p, cost_vector, init_cost_vector)
                self.solve()
                # # --- Enforce rational bidding floor and cap ---
                for g in (p if isinstance(p, (list, tuple, set)) else [p]):
                    alpha_new = self.model.alpha[g].value
                    alpha_new = max(alpha_new, init_cost_vector[g])   # rational floor
                    cost_vector[g] = alpha_new
                    alpha_history[iter][g] = alpha_new

            # --- Market clearing (global consistency) ---
            dispatch_round, clearing_price_round, _ = self.economic_dispatch(cost_vector)
            dispatch_history[iter] = dispatch_round
            clearing_price_history.append(clearing_price_round)

            for j in range(self.num_generators):
                profit_history[iter][j] = (
                    clearing_price_round * dispatch_round[j]
                    - init_cost_vector[j] * dispatch_round[j]
                )

            # --- Convergence check ---
            if iter > 5:
                for j in range(self.num_generators):
                    prev_profit = profit_history[iter - 1][j]
                    curr_profit = profit_history[iter][j]
                    # curr_clearing_price = clearing_price_history[-1]
                    # prev_clearing_price = clearing_price_history[-2]
                    curr_bid = alpha_history[iter][j]
                    prev_bid = alpha_history[iter - 1][j]
                    if (
                        curr_profit >= prev_profit * (1 - self.convergence_tol)
                        and curr_profit <= prev_profit * (1 + self.convergence_tol)
                        # and curr_clearing_price >= prev_clearing_price * (1 - self.convergence_tol)
                        # and curr_clearing_price <= prev_clearing_price * (1 + self.convergence_tol)
                        and curr_bid >= prev_bid * (1 - self.convergence_tol)
                        and curr_bid <= prev_bid * (1 + self.convergence_tol)
                    ):
                        convergence_check[iter][j] = True
                    else:
                        convergence_check[iter][j] = False

                if all(convergence_check[iter]):
                    print(f"Run id: {run_id} - Converged after {iter} full rounds.")
                    converged = True
                    break
            if iter == self.max_iter:
                print(f"Run id: {run_id} - Reached maximum iterations {self.max_iter} without convergence.")

            iter += 1

        # --- Final consistent results ---
        PoA = clearing_price_round * self.demand / minimum_cost_ED
        final_bid = cost_vector.copy()
        final_dispatch = dispatch_round

        not_converged = not converged

        return (
            profit_history,
            alpha_history,
            dispatch_history,
            iter,
            PoA,
            dispatch_ED,
            clearing_price_ED,
            final_dispatch,
            final_bid,
            clearing_price_round,
            clearing_price_history,
            weight_history,
            converged,
            not_converged,
        )

    def _build_model(self, index_strategic, cost_vector, init_cost_vector):
        self.model = ConcreteModel()
        if isinstance(index_strategic, int):
            # Single-generator competitor
            strategic_set = [index_strategic]
        else:
            # Multi-generator owner (tuple/list)
            strategic_set = list(index_strategic)

        self.model.n_gen = Set(initialize=range(self.num_generators))
        self.model.strategic_index = Set(initialize=strategic_set)  # Index of the strategic producer

        self._build_variables()
        self._build_objective(cost_vector, init_cost_vector)
        self._build_constraints(cost_vector)
        # self._build_policy_constraints(cost_vector)

    def _build_variables(self):
        self.model.P_G = Var(self.model.n_gen, domain=Reals)
        self.model.alpha = Var(self.model.strategic_index, domain=Reals)
        self.model.lambda_dual = Var(domain=Reals)
        self.model.mu_min = Var(self.model.n_gen, domain=Reals)
        self.model.mu_max = Var(self.model.n_gen, domain=Reals)
        self.model.z_min = Var(self.model.n_gen, domain=Binary)
        self.model.z_max = Var(self.model.n_gen, domain=Binary)
        # self.model.tau = Var(self.model.n_gen, domain=Binary)
        self.model.tau = Var(self.model.strategic_index, 
                     self.model.n_gen - self.model.strategic_index, 
                     domain=Binary)

        self.model.omega = Var(self.model.n_gen - self.model.strategic_index, domain=Reals)

    def _build_objective(self, cost_vector, init_cost_vector):
        # Strong duality substitution
        dual_costs = (
            self.model.lambda_dual * self.demand
            + sum(self.model.mu_min[i] * self.Pmin[i] for i in self.model.n_gen)
            - sum(self.model.mu_max[i] * self.Pmax[i] for i in self.model.n_gen)
        )

        non_strat_costs = sum(
            cost_vector[i] * self.model.P_G[i]
            for i in self.model.n_gen - self.model.strategic_index
        )

        strat_term1 = sum(-self.model.mu_min[i] * self.Pmin[i] 
                            + self.model.mu_max[i] * self.Pmax[i] for i in self.model.strategic_index)

        strat_term2 = sum(
            init_cost_vector[i] * self.model.P_G[i]
            for i in self.model.strategic_index
        )

        self.model.objective = Objective(
            expr= - (dual_costs - non_strat_costs + strat_term1) + strat_term2,
            sense=minimize
        )

    def _build_constraints(self, cost_vector):
        # Alpha constraints
        def alpha_min_rule(m, i):
            return m.alpha[i] >= self.alpha_min

        def alpha_max_rule(m, i):
            return m.alpha[i] <= self.alpha_max

        self.model.alpha_min_constr = Constraint(self.model.strategic_index, rule=alpha_min_rule)
        self.model.alpha_max_constr = Constraint(self.model.strategic_index, rule=alpha_max_rule)


        # Power balance constraint
        self.model.power_balance = Constraint(expr=sum(self.model.P_G[i] for i in range(self.num_generators)) == self.demand)

        def stationarity_rule(m, i):
            return m.alpha[i] - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0

        self.model.stationarity = Constraint(self.model.strategic_index, rule=stationarity_rule)

        def stationarity_non_strategic_rule(m, i):
            return cost_vector[i] - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0

        self.model.stationarity_non_strategic = Constraint(
            self.model.n_gen - self.model.strategic_index, rule=stationarity_non_strategic_rule
        )
        # ------------------------
        # Big-M + binary formulation
        # ------------------------
        M = 10000

        def tau_rule_lower(m, k, i):
            return m.alpha[k] <= cost_vector[i] * 0.999 + M * m.tau[k, i]


        def tau_rule_upper(m, k, i):
            return m.alpha[k] >= cost_vector[i] * 1.001 - M * (1 - m.tau[k, i])

        self.model.tau_lower = Constraint(self.model.strategic_index, self.model.n_gen - self.model.strategic_index, rule=tau_rule_lower)
        self.model.tau_upper = Constraint(self.model.strategic_index, self.model.n_gen - self.model.strategic_index, rule=tau_rule_upper)

        # self.model.tau_sum = Constraint(expr=sum(self.model.tau[i] for i in self.model.n_gen - self.model.strategic_index) <= len(cost_vector) - 1)

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
        if "owner_indexes" not in self.results[run_id]:
            owner_indexes = set()
        else:
            owner_indexes = set(self.results[run_id]['owner_indexes'])

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
            # Add bold / color highlight for owned generators
            if idx in owner_indexes:
                label_color = "red"
                font_weight = "bold"
            else:
                label_color = "blue"
                font_weight = "normal"

            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)

            cum_cap += cap

            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)

            midpoint = cum_cap - cap / 2
            
            # Highlight ownership
            if idx in owner_indexes:
                y_offset = -2
                label_color = "red"
                font_weight = "bold"
                label_text = "Generator owned"   
            else:
                y_offset = -0.2
                label_color = "blue"
                font_weight = "normal"
                label_text = f"G{idx}"           

            plt.text(
                midpoint,
                c + y_offset,
                label_text,                     
                ha="center",
                va="bottom" if y_offset > 0 else "top",
                fontsize=9,
                color=label_color,
                fontweight=font_weight,
            )

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
        plt.title(f'Merit Order Curve: ED vs Strategic Producer. Run ID: {run_id}')
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
                f"{cost_sp:9.5f} "
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

    def plot_profits(self, run_id):
        profit_history = self.results[run_id]['profit_history']

        profit_history = np.array(profit_history)
        plt.figure(figsize=(10, 6))
        for i in range(self.num_generators):
            plt.plot(profit_history[:, i], marker='o', label=f'Generator {i}')
        plt.xlabel('Iteration')
        plt.ylabel('Profit')
        plt.title('Profit Evolution Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def generate_scaled_setup(n_players: int, base_demand=100):
    """
    Generates ordered generator parameters for n_players,
    preserving the relative pattern from [30, 35, 40, 45].
    Keeps total system capacity constant for comparable social welfare.
    """

    # --- Reference pattern from 4-player base case ---
    base_pattern = np.array([30, 35, 40, 45])
    base_pattern_sum = base_pattern.sum()
    base_pattern = base_pattern / base_pattern.sum()  # normalize

    # --- Interpolate to match n_players ---
    x_base = np.linspace(0, 1, len(base_pattern))
    x_new = np.linspace(0, 1, n_players)
    pattern_scaled = np.interp(x_new, x_base, base_pattern)

    # Normalize so total capacity is constant (≈165)
    Pmax = pattern_scaled / pattern_scaled.sum() * base_pattern_sum
    Pmin = np.zeros(n_players)

    # --- Costs: follow same increasing pattern ---
    base_cost_min = np.array([200, 300, 400, 500])
    base_cost_max = np.array([400, 600, 800, 1000])

    cost_min_pattern = np.interp(x_new, x_base, base_cost_min)
    cost_max_pattern = np.interp(x_new, x_base, base_cost_max)

    cost_min = np.round(cost_min_pattern, 1)
    cost_max = np.round(cost_max_pattern, 1)

    demand = base_demand

    return (
        Pmin.tolist(),
        Pmax.round(1).tolist(),
        cost_min.tolist(),
        cost_max.tolist(),
        demand,
    )

def run_multiple_player_setups(max_players: int = 10):
    alpha_min = 0
    alpha_max = 1200
    
    Pmin = [0, 0, 0, 0]
    Pmax = [30, 35, 40, 45]

    cost_min = [200, 250, 300, 350]
    cost_max = [c * 2 for c in cost_min]
    cost_ownership = None
    segments = 2

    max_iter = 150
    demand = 100

    convergence_tol = 0.001
   
    epec_results = {}

    players_list = range(4, max_players + 1)
    convergence_rate = []
    worst_poa_list   = []

    for n_players in players_list:
        print(f"\n--- Running EPEC for {n_players} players ---")
        Pmin, Pmax, cost_min, cost_max, demand = generate_scaled_setup(n_players=n_players)
        print("Pmin:", Pmin)
        print("Pmax:", Pmax)
        print("Cost min:", cost_min)
        print("Cost max:", cost_max)
        print("Demand:", demand)

        epec = EPEC(alpha_min = alpha_min, alpha_max = alpha_max, 
                    Pmin = Pmin, Pmax = Pmax, demand = demand, 
                    cost_min = cost_min, cost_max = cost_max, 
                    segments = segments, 
                    cost_ownership = cost_ownership,
                    max_iter = max_iter, convergence_tol = convergence_tol)
        
        share_converged, worst_poa = epec.iterate_cost_combinations()
        convergence_rate.append(share_converged)
        worst_poa_list.append(worst_poa)
        epec_results[n_players] = epec
    
    # # --- Plot convergence rate vs number of players ---
    plt.figure(figsize=(8, 5))
    plt.plot(players_list, convergence_rate, marker='o')
    plt.xlabel('Number of Players')
    plt.ylabel('Convergence Rate (%)')
    plt.title('EPEC Convergence Rate vs Number of Players')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # # --- Plot worst PoA vs number of players ---
    plt.figure(figsize=(8, 5))
    plt.plot(players_list, worst_poa_list, marker='o', color='orange')
    plt.xlabel('Number of Players')
    plt.ylabel('Worst Price of Anarchy (PoA)')
    plt.title('Worst PoA vs Number of Players')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return epec_results
    
if __name__ == "__main__":
    
    alpha_min = 0
    alpha_max = 1200
    
    Pmin = [0, 0, 0, 0, 0]
    Pmax = [30, 30, 30, 30, 30]

    cost_min = [200, 250, 300, 350, 400]
    cost_max = [c * 2 for c in cost_min]
    segments = 2

    cost_ownership = [250, 300, 350, 400, 450]  # example ownership costs

    max_iter = 150
    demand = 75

    convergence_tol = 0.001

    epec = EPEC(alpha_min = alpha_min, alpha_max = alpha_max, 
                Pmin = Pmin, Pmax = Pmax, demand = demand, 
                cost_min = cost_min, cost_max = cost_max, 
                segments = segments, 
                cost_ownership = cost_ownership,
                max_iter = max_iter, convergence_tol = convergence_tol)
    
    epec.iterate_ownership_combinations(2)
    for run_id in epec.results:
        epec.plot_merit_order_curve(run_id = run_id)
        epec.plot_profits(run_id = run_id)

    # # # epec.plot_clearing_price_over_iterations(run_id = 0)
    # # # epec.plot_alpha_over_iterations(run_id = 0)
    # # # epec.plot_dispatch_over_iterations(run_id = 0)
    # for run_id in epec.results:
    #     epec.plot_merit_order_curve(run_id = run_id)
    #     epec.plot_profits(run_id = run_id)
    # epec.plot_weights(run_id = 0)
    # epec.plot_PoA()

    # multiplayer_results = run_multiple_player_setups(max_players=6)


    # print("Omega[0]:", epec.model.omega[0].value)
    # print("Omega[1]:", epec.model.omega[1].value)
    # print("Final Bid[0]:", epec.results[26]['final_bid'][0])
    # print("Final Bid[1]:", epec.results[26]['final_bid'][1])
    # print("Pmax[0]:", epec.Pmax[0])
    # print("Pmax[1]:", epec.Pmax[1])
    # print("Policy Check:", epec.model.alpha.value, "==", epec.model.omega[0].value * epec.results[26]['final_bid'][0] * epec.Pmax[0] + epec.model.omega[1].value * epec.results[26]['final_bid'][1] * epec.Pmax[1])
