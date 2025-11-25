from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import os
import csv

class EPEC:
    def __init__(self, 
                 alpha_min, alpha_max, 
                 Pmin, Pmax, 
                 demand,
                 cost_min, cost_max, segments,
                 max_iter = 100, convergence_tol = 0.01,
                 heuristic: bool = True):

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

        self.heuristic = heuristic  # whether to use heuristic (randomized update order and rational bidding floor)

        self.num_generators = len(Pmin)

        self.cost = np.array([
            np.linspace(self.cost_min[i], self.cost_max[i], self.segments)
            for i in range(self.num_generators)
        ])
        
        self.results = {}

    def run_single_experiment(self):
        run_id = 0
        init_cost_vector = np.array([self.cost_min[i] for i in range(self.num_generators)])
        (profits, alphas, dispatches, iterations, PoA, dispatch_ED, 
             clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, 
             clearing_price_history, weight_history, converged, not_converged,
             player_order_history, round_by_round_bids, round_by_round_dispatch, 
             round_by_round_prices, round_by_round_profits
             ) = self.run_best_response(init_cost_vector, run_id)
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

    def iterate_cost_combinations(self):
        # all combinations of cost vectors (Cartesian product)
        all_combinations = list(itertools.product(*self.cost))

        print(f"Total combinations to evaluate: {len(all_combinations)}")
        total_runs = len(all_combinations)
        converged_runs = 0

        for run_id, init_cost_vector in enumerate(all_combinations):
            init_cost_vector = np.array(init_cost_vector)
            (profits, alphas, dispatches, iterations, PoA, dispatch_ED, 
             clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, 
             clearing_price_history, weight_history, converged, not_converged,
             player_order_history, round_by_round_bids, round_by_round_dispatch, 
             round_by_round_prices, round_by_round_profits
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
             clearing_price_history, weight_history, converged, not_converged,
             actor_profit_history
             ) = self.run_best_response_ownership(owner_indexes, run_id)
            if converged:
                converged_runs += 1

            self.results[run_id] = {
                "init_cost_vector": self.cost_min,
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
        
        #convert results to csv
        file_path = os.path.join(os.path.dirname(__file__), f"results_{self.demand}_{self.num_generators}_{self.alpha_min}.csv")
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Key", "Value"])
            for key, value in self.results.items():
                writer.writerow([key, value])


        print(f"Worst PoA: {worst_poa:.2f} (from run id {worst_id})")
        return share_converged, worst_poa

    def run_best_response(self, init_cost_vector, run_id):
        profit_history, alpha_history, dispatch_history = [], [], []
        convergence_check, clearing_price_history, weight_history = [], [], []
        player_order_history = []
        round_by_round_bids = []
        round_by_round_dispatch = []
        round_by_round_prices = []
        round_by_round_profits = []

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
            if self.heuristic:
                random.shuffle(players)                    # randomize update order each iteration
                player_order_history.append(players.copy())
            else:
                player_order_history.append(players.copy())  # fixed order

            iter_round_bids = []
            iter_round_dispatch = []
            iter_round_prices = []
            iter_round_profits = []

            for p in players:
                self._build_model(p, cost_vector, init_cost_vector)
                self.solve()
                for g in (p if isinstance(p, (list, tuple, set)) else [p]):
                    alpha_new = self.model.alpha[g].value
                    # Add heuristic: enforce rational bidding floor and cap
                    if self.heuristic:
                        alpha_new = max(alpha_new, init_cost_vector[g])   

                    cost_vector[g] = alpha_new
                    alpha_history[iter][g] = alpha_new
                # --- Capture state after this player/actor acts ---
                dispatch_after_round, clearing_price_after_round, _ = self.economic_dispatch(cost_vector)
                iter_round_bids.append(cost_vector.copy())
                iter_round_dispatch.append(dispatch_after_round)
                iter_round_prices.append(clearing_price_after_round)
                
                # Calculate profit for the strategic player/actor
                # If p is a single generator (int), calculate its profit
                # If p is multiple generators (list/tuple/set), calculate total profit for the actor
                if isinstance(p, (list, tuple, set)):
                    strategic_profit = sum(
                        clearing_price_after_round * dispatch_after_round[g]
                        - init_cost_vector[g] * dispatch_after_round[g]
                        for g in p
                    )
                else:
                    strategic_profit = (
                        clearing_price_after_round * dispatch_after_round[p]
                        - init_cost_vector[p] * dispatch_after_round[p]
                    )
                iter_round_profits.append(strategic_profit)

            # --- Market clearing (global consistency) ---
            dispatch_round, clearing_price_round, _ = self.economic_dispatch(cost_vector)
            dispatch_history[iter] = dispatch_round
            clearing_price_history.append(clearing_price_round)

            for j in range(self.num_generators):
                profit_history[iter][j] = (
                    clearing_price_round * dispatch_round[j]
                    - init_cost_vector[j] * dispatch_round[j]
                )

            round_by_round_bids.append(iter_round_bids)
            round_by_round_dispatch.append(iter_round_dispatch)
            round_by_round_prices.append(iter_round_prices)
            round_by_round_profits.append(iter_round_profits)

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
            iter += 1
            if iter == self.max_iter:
                print(f"Run id: {run_id} - Reached maximum iterations {self.max_iter} without convergence.")

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
            player_order_history,
            round_by_round_bids,
            round_by_round_dispatch,
            round_by_round_prices,
            round_by_round_profits
        )

    def run_best_response_ownership(self, owner_indexes, run_id):
        profit_history, alpha_history, dispatch_history = [], [], []
        convergence_check, clearing_price_history, weight_history = [], [], []
        actor_profit_history = []

        init_cost_vector = self.cost_min

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

            # Track profit for each actor: owner + individual competitors
            num_actors = 1 + len([i for i in range(self.num_generators) if i not in owner_indexes])
            actor_profit_history.append([None] * num_actors)

            # competitors = all generators not owned
            competitors = [i for i in range(self.num_generators) if i not in owner_indexes]

            # Create player update list
            players = [owner_indexes] + competitors

            # If the heuristic is enabled, randomize the update order each iteration
            if self.heuristic:
                random.shuffle(players)                    # randomize update order each iteration

            for p in players:
                self._build_model(p, cost_vector, init_cost_vector)
                self.solve()
                # # --- Enforce rational bidding floor and cap ---
                for g in (p if isinstance(p, (list, tuple, set)) else [p]):
                    alpha_new = self.model.alpha[g].value
                    # Add heuristic: enforce rational bidding floo
                    if self.heuristic:
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

            # --- Calculate actor-level profits ---
            # Actor 0: the owner (combines all owned generators)
            owner_profit = sum(profit_history[iter][g] for g in owner_indexes)
            actor_profit_history[iter][0] = owner_profit

            # Actors 1+: individual competitors
            actor_idx = 1
            for j in range(self.num_generators):
                if j not in owner_indexes:
                    actor_profit_history[iter][actor_idx] = profit_history[iter][j]
                    actor_idx += 1

            # --- Convergence check ---
            if iter > 5:
                num_actors = len(actor_profit_history[iter])
                actor_converged = [False] * num_actors
                
                for actor_idx in range(num_actors):
                    prev_profit = actor_profit_history[iter - 1][actor_idx]
                    curr_profit = actor_profit_history[iter][actor_idx]
                    
                    # Check profit convergence for this actor
                    profit_converged = (
                        curr_profit >= prev_profit * (1 - self.convergence_tol) and
                        curr_profit <= prev_profit * (1 + self.convergence_tol)
                    )
                    
                    # Check bid convergence for this actor's generators
                    if actor_idx == 0:
                        # Owner: check all owned generators' bids
                        bids_converged = all(
                            alpha_history[iter][g] >= alpha_history[iter-1][g] * (1 - self.convergence_tol) and
                            alpha_history[iter][g] <= alpha_history[iter-1][g] * (1 + self.convergence_tol)
                            for g in owner_indexes
                        )
                    else:
                        # Competitor: find which generator this actor owns
                        competitors = [i for i in range(self.num_generators) if i not in owner_indexes]
                        gen_idx = competitors[actor_idx - 1]
                        bids_converged = (
                            alpha_history[iter][gen_idx] >= alpha_history[iter-1][gen_idx] * (1 - self.convergence_tol) and
                            alpha_history[iter][gen_idx] <= alpha_history[iter-1][gen_idx] * (1 + self.convergence_tol)
                        )
                    
                    actor_converged[actor_idx] = profit_converged and bids_converged
                
                # Check if all actors have converged
                if all(actor_converged):
                    print(f"Run id: {run_id} - Converged after {iter} full rounds.")
                    print(f"  Owner profit: ${actor_profit_history[iter][0]:.2f}")
                    print(f"  Competitor profits: {[actor_profit_history[iter][i] for i in range(1, num_actors)]}")
                    print(f"  Owner share of total profit: {100 * actor_profit_history[iter][0] / sum(actor_profit_history[iter]):.2f}%")
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
            actor_profit_history
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
    
    # def _build_policy_constraints(self, cost_vector):
    #     # Create arbitrary policy constraints
    #     def policy_rule_1(m):
    #         return m.alpha == sum(m.omega[i] * cost_vector[i] * self.Pmax[i] for i in self.model.n_gen - self.model.strategic_index)

    #     self.model.policy_1 = Constraint(rule=policy_rule_1)

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

        plt.figure(figsize=(12, 7))

        gen_curve_x = [0]
        gen_curve_y = [0]
        cum_cap = 0
        
        for idx, (c, cap) in zip(gen_sorted_idx, zip(gen_sorted_costs, gen_sorted_caps)):
            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)
            cum_cap += cap
            gen_curve_x.append(cum_cap)
            gen_curve_y.append(c)

            midpoint = cum_cap - cap / 2
            
            # ED labels - positioned ABOVE the curve
            if idx in owner_indexes:
                label_color = "darkblue"
                font_weight = "bold"
                label_text = f"G{idx}*"  # Add asterisk for owned generators
            else:
                label_color = "darkblue"
                font_weight = "normal"
                label_text = f"G{idx}"

            plt.text(
                midpoint,
                c + 5,  # Position above the curve
                label_text,
                ha="center",
                va="bottom",
                fontsize=10,
                color=label_color,
                fontweight=font_weight,
            )

        # Plot ED supply curve
        plt.step(gen_curve_x, gen_curve_y, where='post', color='blue', linewidth=2, label='Supply (ED)')

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

            # SP labels - positioned BELOW the curve
            midpoint_SP = cum_cap_SP - cap / 2
            
            if idx in owner_indexes:
                label_color_SP = "darkred"
                font_weight_SP = "bold"
                label_text_SP = f"G{idx}*"
            else:
                label_color_SP = "purple"
                font_weight_SP = "normal"
                label_text_SP = f"G{idx}"
            
            plt.text(
                midpoint_SP, 
                c - 8,  # Position below the curve
                label_text_SP, 
                ha='center', 
                va='top', 
                fontsize=10, 
                color=label_color_SP,
                fontweight=font_weight_SP
            )

        # Plot SP supply curve
        plt.step(gen_curve_x_SP, gen_curve_y_SP, where='post', color='purple', linestyle='--', linewidth=2, label='Supply (SP)')

        # --- Demand ---
        demand = self.demand
        plt.axvline(demand, color='red', linestyle='--', linewidth=2, label=f'Demand = {demand}')

        # --- Clearing prices ---
        plt.scatter([demand], [clearing_price_ED], color='green', zorder=5, marker='o', 
                    label=f'ED Price = {clearing_price_ED:.2f}', s=150, edgecolors='black', linewidths=2)
        plt.scatter([demand], [clearing_price_SP], color='magenta', zorder=5, marker='x', 
                    label=f'SP Price = {clearing_price_SP:.2f}', s=150, linewidths=3)

        # --- Formatting ---
        plt.xlabel('Quantity (MW)', fontsize=12, fontweight='bold')
        plt.ylabel('Price ($/MWh)', fontsize=12, fontweight='bold')
        # Add ownership info to title if applicable
        if owner_indexes:
            owned_gens = ', '.join([f'G{i}' for i in sorted(owner_indexes)])
            plt.title(f'Merit Order Curve: ED vs Strategic Producer. Run ID: {run_id}\n'
                    f'Owned generators: {owned_gens} (marked with *)', 
                    fontsize=14, fontweight='bold')
        else:
            plt.title(f'Merit Order Curve: ED vs Strategic Producer. Run ID: {run_id}', 
                    fontsize=14, fontweight='bold')

        plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                fontsize=10, ncol=3, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        plt.show()

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
        # Filter only converged runs
        PoA_values = [
            self.results[run_id]['PoA']
            for run_id in self.results
            if self.results[run_id]['converged']  # <--- ONLY converged runs
        ]

        if len(PoA_values) == 0:
            print("No converged runs — cannot plot PoA distribution.")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(PoA_values, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Price of Anarchy (PoA)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Price of Anarchy (Only Converged Runs)')
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

        # Check if this is an ownership scenario
        if "owner_indexes" in self.results[run_id]:
            owner_indexes = set(self.results[run_id]['owner_indexes'])
        else:
            owner_indexes = set()

        profit_history = np.array(profit_history)
        plt.figure(figsize=(10, 6))
        
        for i in range(self.num_generators):
            # Add asterisk for owned generators
            label = f'Generator {i}*' if i in owner_indexes else f'Generator {i}'
            plt.plot(profit_history[:, i], marker='o', label=label)
        
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Profit ($)', fontsize=12, fontweight='bold')
        
        # Add ownership info to title
        if owner_indexes:
            owned_gens = ', '.join([f'G{i}' for i in sorted(owner_indexes)])
            plt.title(f'Profit Evolution Over Iterations - Run ID: {run_id}\n'
                    f'Generators marked with * are owned together: {owned_gens}', 
                    fontsize=13, fontweight='bold')
        else:
            plt.title(f'Profit Evolution Over Iterations - Run ID: {run_id}', 
                    fontsize=13, fontweight='bold')
        
        plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                ncol=4, framealpha=0.9, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make room for legend below
        plt.show()

    def plot_actor_profits(self, run_id):
        """
        Plot actor-level profits over iterations for ownership scenarios.
        Shows the owner's combined profit and each competitor's individual profit.
        """
        if "owner_indexes" not in self.results[run_id]:
            print(f"Run {run_id} is not an ownership scenario. Use plot_profits() instead.")
            return
        
        owner_indexes = self.results[run_id]['owner_indexes']
        profit_history = self.results[run_id]['profit_history']
        iterations = self.results[run_id]['iterations']
        
        # Calculate actor profits for each iteration
        actor_profit_history = []
        
        for iter_profits in profit_history:
            # Owner profit: sum of all owned generators
            owner_profit = sum(iter_profits[g] for g in owner_indexes if iter_profits[g] is not None)
            
            # Competitor profits: individual generators not owned
            competitor_profits = [
                iter_profits[g] for g in range(self.num_generators) 
                if g not in owner_indexes and iter_profits[g] is not None
            ]
            
            actor_profit_history.append([owner_profit] + competitor_profits)
        
        # Transpose to get profit trajectories per actor
        num_iters = len(actor_profit_history)
        num_actors = len(actor_profit_history[0])
        
        actor_trajectories = []
        for actor_idx in range(num_actors):
            trajectory = [actor_profit_history[iter_idx][actor_idx] 
                        for iter_idx in range(num_iters)]
            actor_trajectories.append(trajectory)
        
        # Create the plot
        plt.figure(figsize=(12, 7))
        
        # Plot owner profit (Actor 0)
        owned_gens = ', '.join([f'G{i}' for i in sorted(owner_indexes)])
        plt.plot(range(1, num_iters + 1), actor_trajectories[0], 
                marker='o', linewidth=2.5, markersize=8, 
                label=f'Owner (owns {owned_gens})', 
                color='red', linestyle='-', alpha=0.8)
        
        # Plot competitor profits (Actors 1+)
        competitors = [i for i in range(self.num_generators) if i not in owner_indexes]
        colors = plt.cm.tab10(range(len(competitors)))
        
        for actor_idx in range(1, num_actors):
            gen_idx = competitors[actor_idx - 1]
            plt.plot(range(1, num_iters + 1), actor_trajectories[actor_idx], 
                    marker='s', linewidth=2, markersize=6, 
                    label=f'Competitor G{gen_idx}', 
                    color=colors[actor_idx - 1], linestyle='--', alpha=0.7)
        
        conv_iter = iterations if self.results[run_id]['converged'] else None
        
        # Formatting
        plt.xlabel('Iteration', fontsize=13, fontweight='bold')
        plt.ylabel('Profit ($)', fontsize=13, fontweight='bold')
        conv_text = f' (Converged at iteration {conv_iter})' if conv_iter else ' (Did not converge)'
        plt.title(f'Actor Profits Over Iterations - Run ID: {run_id}{conv_text}\n'
                f'Owner controls: {owned_gens}', 
                fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
          fontsize=10, framealpha=0.9, ncol=3)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add summary statistics box
        final_owner_profit = actor_trajectories[0][-1]
        total_competitor_profit = sum(traj[-1] for traj in actor_trajectories[1:])
        
        textstr = f'Final Profits:\n'
        textstr += f'Owner: ${final_owner_profit:.2f}\n'
        textstr += f'All Competitors: ${total_competitor_profit:.2f}\n'
        textstr += f'Owner Share: {final_owner_profit/(final_owner_profit+total_competitor_profit)*100:.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
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
    
    # Basic setup
    alpha_min = 0
    alpha_max = 1200
    convergence_tol = 0.001
    heuristic = True
    max_iter = 150

    # Exercise 3 and 4 setup
    Pmin = [ 0,  0,  0,  0]
    Pmax = [30, 35, 40, 45]

    demand = 100

    cost_min = [200, 250, 300, 350]
    cost_max = [c * 2 for c in cost_min]

    segments = 2

    # Run single experiment - exercise 3
    epec = EPEC(alpha_min = alpha_min, alpha_max = alpha_max, 
                Pmin = Pmin, Pmax = Pmax, demand = demand, 
                cost_min = cost_min, cost_max = cost_max, 
                segments = segments, 
                max_iter = max_iter, convergence_tol = convergence_tol,
                heuristic=heuristic)
    
    epec.run_single_experiment()

    epec.plot_merit_order_curve(run_id = 0)
    epec.plot_profits(run_id = 0)
    epec.plot_alpha_over_iterations(run_id = 0)
    epec.plot_clearing_price_over_iterations(run_id = 0)
    epec.plot_dispatch_over_iterations(run_id = 0)

    # Exercise 4

    epec.iterate_cost_combinations()
    for run_id in epec.results:
        epec.plot_merit_order_curve(run_id = run_id)
        epec.plot_profits(run_id = run_id)
    epec.plot_PoA()


    # multiplayer_results = run_multiple_player_setups(max_players=6)



    # Changed such that cost_ownership is equivalent to cost_min in program
    # Pmin = [ 0,  0,  0,  0,  0,  0,  0]
    # Pmax = [30, 30, 30, 30, 30, 30, 30]

    # cost_min = [1,1.5,2.5,24,25,27.5,29]  # example ownership costs

    # demand = 175
    

    # epec = EPEC(alpha_min = alpha_min, alpha_max = alpha_max, 
    #             Pmin = Pmin, Pmax = Pmax, demand = demand, 
    #             cost_min = cost_min, cost_max = cost_max, 
    #             segments = segments, 
    #             max_iter = max_iter, convergence_tol = convergence_tol,
    #             heuristic=heuristic)
    
    # epec.iterate_ownership_combinations(2)


    #for run_id in epec.results:
     #   epec.plot_merit_order_curve(run_id = run_id)
     #   epec.plot_profits(run_id = run_id)
     #   epec.plot_actor_profits(run_id=run_id)

    # # # epec.plot_clearing_price_over_iterations(run_id = 0)
    # # # epec.plot_alpha_over_iterations(run_id = 0)
    # # # epec.plot_dispatch_over_iterations(run_id = 0)
    # for run_id in epec.results:
    #     epec.plot_merit_order_curve(run_id = run_id)
    #     epec.plot_profits(run_id = run_id)
    # epec.plot_weights(run_id = 0)
    # epec.plot_PoA()

