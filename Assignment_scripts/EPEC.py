from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import os
from typing import List, Dict
import csv
import pandas as pd
from Assignment_scripts.config_loader import load_defaults


class EPEC:
    def __init__(self,  
                 Pmin: List[float], Pmax: List[float], 
                 demand: float,
                 cost: List[float], 
                 segments: int = None,
                 exercise: str = "3"
                ):

        # Load and assign algorithm parameters from defaults.yaml
        d = load_defaults("Assignment_scripts/defaults.yaml")

        self.alpha_min       = d["alpha_min"]
        self.alpha_max       = d["alpha_max"]
        self.max_iter        = d["max_iter"]
        self.convergence_tol = d["convergence_tol"]
        self.heuristic       = d["heuristic"]

        self.exercise = exercise

        # ----- Problem data -----
        self.Pmin = [float(val) for val in Pmin]
        self.Pmax = [float(val) for val in Pmax]

        # Define number of generators
        self.num_generators = len(Pmin) # Number of generators defined by length of Pmin/Pmax/cost

        self.demand = float(demand) # System demand

        self.segments = segments # number of segments in the cost discretization

        self.true_cost = [float(c) for c in cost] # True costs of each generator
        if segments is not None:
            cost_max = [c * 2 for c in self.true_cost] # Define maximum cost as double the true cost
            # Create cost discretization for each generator
            self.cost = np.array([
                np.linspace(self.true_cost[i], cost_max[i], self.segments)
                for i in range(self.num_generators)
            ])

        assert len(Pmax) == self.num_generators, "Pmax length must match number of generators"
        assert len(cost) == self.num_generators, "Cost length must match number of generators"

        # Results storage
        self.results = {}
    
    def run_single_experiment(self) -> None: 
        """
        Function used for exercise 3 - Single experiment with the cost vector inserted. 
        Each player is a single generator, the Best Response algorithm is used to compute the Nash Equilibrium.
        Saves the results in the results dictionary. 
        """
        run_id = 0
        cost_vector = np.array([self.true_cost[i] for i in range(self.num_generators)])
        self.run_best_response(cost_vector, run_id)
        
    def iterate_cost_combinations(self) -> tuple[float, float]:
        """
        Function used for exercise 4
        - Iterate over all cost vector combinations. 
        Each player is a single generator, the Best Response algorithm is used to compute the Nash Equilibrium.
        Saves the results for each run_id ie. setup used in the results dictionary.   
        The returned values are used when increasing the number of players

        Return
        -------
        share_converged : float
            Percentage of runs that converged.
        -------
        worst_poa : float
            Worst Price of Anarchy (PoA) observed among all runs.
        """

        # all combinations of cost vectors (Cartesian product)
        all_combinations = list(itertools.product(*self.cost))

        print(f"Total combinations to evaluate: {len(all_combinations)}")
        
        #The total number of runs is the length of all_combinations
        total_runs = len(all_combinations)

        # Initialize counter for converged runs
        converged_runs = 0

        # For all the cost vector combinations, run the best response algorithm
        for run_id, cost_vector in enumerate(all_combinations):

            # Get the cost vector from the combination tuple
            cost_vector = np.array(cost_vector)

            # Run the best response algorithm for this cost vector and run_id
            self.run_best_response(cost_vector, run_id)

            # If the run converged, increment the counter
            if self.results[run_id]["converged"]:
                converged_runs += 1

        # Calculate share of converged runs
        share_converged = 100 * converged_runs / total_runs if total_runs > 0 else 0.0
        print(f"Converged in {converged_runs}/{total_runs} runs ({share_converged:.1f}%)")
    
        # Find the worst PoA among all runs
        worst_id, worst_poa = max(
            ((id, res['PoA']) for id, res in self.results.items()),
            key=lambda x: x[1]
        )

        print(f"Worst PoA: {worst_poa:.2f} (from run id {worst_id})")
        return share_converged, worst_poa

    def run_best_response(self, cost_vector: List[float], run_id: int) -> None:
        """
        The best response algorithm implementation as presented in the assignment and class. This algorithm also stores the internal histories for analysis

        Saves the results in the results dictionary

        Parameters
        -------
        cost_vector : List[float]
            Array of cost values for each generator.
        run_id : int
            Identifier for the current run/experiment.
        """

        # Initialize histories from global perspective
        profit_history, alpha_history, dispatch_history, clearing_price_history = [], [], [], []

        # Initialize the internal histories from each player's perspective
        internal_profit_history, internal_bid_history, internal_dispatch_history, internal_clearing_price_history = [], [], [], []
        # NOTE the internal_profit_history is the only list needed for the algorithm to work - The others are stored for analysis purposes

        # Initialize player order history
        player_order_history = []

        # Initialize convergence checks       
        convergence_check_1, convergence_check_2 = [], []
        
        # Calculate the economic dispatch for the true cost vector
        dispatch_ED, clearing_price_ED = self.economic_dispatch(cost_vector)

        # Initialize the iteration counter, bid vector and convergence flag
        iter = 0
        bid_vector = cost_vector.copy()
        converged = False
        
        # Main loop for the best response algorithm
        while iter < self.max_iter:

            # Initialize global histories for this iteration
            profit_history.append([None] * self.num_generators)
            alpha_history.append([None] * self.num_generators)
            dispatch_history.append([None] * self.num_generators)
            # NOTE the clearing price is only one value and is appended when calculated - does not need to be initialized here

            # Initialize internal histories for this iteration 
            internal_profit_history.append([None] * self.num_generators)
            internal_clearing_price_history.append([None] * self.num_generators)
            # NOTE the internal_bid_history and internal_dispatch_history are appended after each player's action - does not need to be initialized here
 
            # Initialize convergence checks for this iteration
            convergence_check_1.append([False] * self.num_generators)
            convergence_check_2.append([False] * self.num_generators)

            # Create player update list
            players = list(range(self.num_generators))

            # If the heuristic is enabled, randomize the update order each iteration
            if self.heuristic:
                random.seed(iter + run_id)
                random.shuffle(players)
            
            # Store the player order for this iteration
            player_order_history.append(players.copy())

            # Each player updates their bid in sequence
            for p in players:
                self._build_model(p, bid_vector, cost_vector)
                self.solve()

                # Robustly handle both single generator players and multi-generator players
                for g in (p if isinstance(p, (list, tuple, set)) else [p]):
                    
                    # Retrieve the bid from the MPEC
                    alpha_new = self.model.alpha[g].value
                    # Add heuristic: enforce rational bidding floor and cap
                    if self.heuristic:
                        alpha_new = max(alpha_new, cost_vector[g])   

                    # Update the bid vector and store the new bid
                    bid_vector[g] = alpha_new
                    alpha_history[iter][g] = alpha_new

                # Store internal histories after player's action
                internal_profit_history[iter][p] = -value(self.model.objective)
                internal_dispatch_history.append([self.model.P_G[i].value for i in range(self.num_generators)])
                internal_clearing_price_history[iter][p] = self.model.lambda_dual.value
                internal_bid_history.append(bid_vector.copy())

            # Store clearing price for this iteration after all players have acted
            clearing_price_history.append(internal_clearing_price_history[iter][p])

            # After the first 5 iterations, check for convergence
            if iter > 4:
                # First convergence check - after each player has acted
                for p in players:
                    # Get the previous and current profit for player p
                    prev_profit = internal_profit_history[iter - 1][p]
                    curr_profit = internal_profit_history[iter][p]   

                    # Check if the profit has converged within the tolerance
                    if (curr_profit + self.convergence_tol  >= prev_profit * (1 - self.convergence_tol) and 
                        curr_profit <= prev_profit * (1 + self.convergence_tol) + self.convergence_tol): 

                        # If True, mark player p as converged for this check
                        convergence_check_1[iter][p] = True

                # If all players passed the first convergence check the second check is performed
                if all(convergence_check_1[iter]):                      
                    # Perform economic dispatch based on current bids
                    dispatch_round, clearing_price_round = self.economic_dispatch(bid_vector)

                    # Store the dispatch results for this iteration
                    dispatch_history[iter] = dispatch_round

                    # For each player, calculate and store profit after market clearing
                    for p in players:
                        # Calculate profit for player p based on market clearing - This is done in cases with both single and multi-generator players
                        if isinstance(p, (list, tuple, set)):
                            profit_market_clearing = sum(
                                clearing_price_round * dispatch_round[g]
                                - cost_vector[g] * dispatch_round[g]
                                for g in p
                            )
                        else:
                            profit_market_clearing = (
                                clearing_price_round * dispatch_round[p]
                                - cost_vector[p] * dispatch_round[p]
                            )
                        
                        # Store the profit after market clearing
                        profit_history[iter][p] = profit_market_clearing

                        # Debugging prints
                        # print("Player", p)
                        # print("Curr profit", internal_profit_history[iter][p] + self.convergence_tol, ">=Threshold",profit_history[iter][p] * (1 - self.convergence_tol))
                        # print("Curr profit", internal_profit_history[iter][p], "<=Threshold",profit_history[iter][p] * (1 + self.convergence_tol) + self.convergence_tol)

                        # Second convergence check - after market clearing - Added the convergence tolerance as well to avoid issues when profit is 0
                        if (internal_profit_history[iter][p] + self.convergence_tol >= profit_history[iter][p] * (1 - self.convergence_tol) and 
                            internal_profit_history[iter][p] <= profit_history[iter][p] * (1 + self.convergence_tol) + self.convergence_tol):
                            convergence_check_2[iter][p] = True

                    # If all players passed the second convergence check, the algorithm has converged
                    if all (convergence_check_2[iter]):
                        print(f"Run id: {run_id} - Converged after {iter} full rounds.")
                        converged = True
                        break
                
                # If the algorithm has not converged pass to the next iteration
                else:
                    # If iter is the last iteration, perform final economic dispatch and store results
                    if iter == self.max_iter - 1:
                        dispatch_round, clearing_price_round = self.economic_dispatch(bid_vector)
                        dispatch_history[iter] = dispatch_round
                        for j in range(self.num_generators):
                            profit_history[iter][j] = (
                                clearing_price_round * dispatch_round[j]
                                - cost_vector[j] * dispatch_round[j]
                            )
            # Update iteration counter
            iter += 1
            
            # If maximum iterations reached without convergence, store last dispatch
            if iter == self.max_iter:
                print(f"Run id: {run_id} - Reached maximum iterations {self.max_iter} without convergence.")
                dispatch_history.append(dispatch_history[iter - 1])

        # Calculate Price of Anarchy (PoA)
        PoA = sum(cost_vector[g] * dispatch_history[iter][g] for g in range(self.num_generators)) / (sum(cost_vector[g] * dispatch_ED[g] for g in range(self.num_generators)))

        # Calculate alternative PoA definition based on bids
        PoA_ish = sum(bid_vector[g] * dispatch_history[iter][g] for g in range(self.num_generators)) / (sum(cost_vector[g] * dispatch_ED[g] for g in range(self.num_generators)))

        final_bid = bid_vector.copy()
        final_dispatch = dispatch_round

        # Store results in the results dictionary
        self.results[run_id] = {
                        "cost_vector": cost_vector,
                        "internal_profit_history": internal_profit_history,
                        "internal_bid_history": internal_bid_history,
                        "internal_dispatch_history": internal_dispatch_history,
                        "internal_clearing_price_history": internal_clearing_price_history,
                        "player_order_history": player_order_history,
                        "profit_history": profit_history,
                        "alpha_history": alpha_history,
                        "dispatch_history": dispatch_history,
                        "clearing_price_history": clearing_price_history,
                        "iterations": iter,
                        "final_dispatch": final_dispatch,
                        "final_bid": final_bid,
                        "final_clearing_price": clearing_price_history[-1],
                        "dispatch_ED": dispatch_ED,
                        "clearing_price_ED": clearing_price_ED,
                        "PoA": PoA,
                        "PoA_ish": PoA_ish,
                        "converged": converged,
                    }

    def iterate_ownership_combinations(self, ownership_size: int = 2) -> tuple[float, float]:
        """
        Function used for exercise 5
        - Iterate over all ownership combinations where one player "ownership_size" generators.
        The Best Response algorithm is then triggered to compute the Nash Equilibrium.
        Saves the results for each run_id ie. setup used in the results dictionary.
        
        Parameters
        -------
        ownership_size : int
            Number of generators owned by the main actor.
        
        Return
        -------
        share_converged : float
            Percentage of runs that converged.
        worst_poa : float
            Worst Price of Anarchy (PoA) observed among all runs.

        """

        # Ownership combinations
        all_combinations = list(itertools.combinations(range(self.num_generators), ownership_size))

        # The different ownership combinations to evaluate
        print(all_combinations)
        print(f"Total combinations to evaluate: {len(all_combinations)}")

        # Initialize counters for total runs and converged runs
        total_runs = len(all_combinations)
        converged_runs = 0

        # For all the ownership combinations, run the best response algorithm
        for run_id, owner_indexes in enumerate(all_combinations):
            print(owner_indexes)
            self.run_best_response_ownership(owner_indexes, run_id)
            if self.results[run_id]["converged"]:
                converged_runs += 1

        # Calculate share of converged runs
        share_converged = 100 * converged_runs / total_runs if total_runs > 0 else 0.0
        print(f"Converged in {converged_runs}/{total_runs} runs ({share_converged:.1f}%)")
    
        # Find the worst PoA among all runs
        worst_id, worst_poa = max(
            ((id, res['PoA']) for id, res in self.results.items()),
            key=lambda x: x[1]
        )
        
        # #convert results to csv
        # os.makedirs("outputs/ownership_results", exist_ok=True)
        # file_path = os.path.join("outputs/ownership_results", f"results_{self.demand}_{self.num_generators}_{self.alpha_min}.csv")
        # with open(file_path, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Key", "Value"])
        #     for key, value in self.results.items():
        #         writer.writerow([key, value])

        print(f"Worst PoA: {worst_poa:.2f} (from run id {worst_id})")
        return share_converged, worst_poa

    def run_best_response_ownership(self, owner_indexes: tuple[int], run_id: int) -> None:
        """
        The best response algorithm implementation for exercise 5 - Ownership of multiple generators. 

        NOTE: This algorithm does not store the internal histories for analysis, like the run_best_response
        
        Saves the results in the results dictionary.

        Parameters
        -------
        owner_indexes : tuple[int]
            List of generator indexes owned by the main actor.
        run_id : int
            Identifier for the current run/experiment.
        """

        # Initialize histories from global perspective
        actor_profit_history, alpha_history, dispatch_history, clearing_price_history = [], [], [], []

        # Initialize the internal_profit_history from each actor's perspective
        internal_actor_profit_history = []

        # Initialize convergence checks
        convergence_check_1, convergence_check_2,  = [], []

        # Calculate the economic dispatch for the true cost vector
        cost_vector = self.true_cost.copy()
        dispatch_ED, clearing_price_ED = self.economic_dispatch(cost_vector)

        # Initialize the iteration counter, bid vector and convergence flag
        iter = 0
        bid_vector = cost_vector.copy()
        converged = False
        
        # Main loop for the best response algorithm
        while iter < self.max_iter:
            # Initialize global histories for this iteration
            alpha_history.append([None] * self.num_generators)
            dispatch_history.append([None] * self.num_generators)

            # Track profit for each actor: owner + individual competitors
            num_actors = 1 + len([i for i in range(self.num_generators) if i not in owner_indexes])

            # Initialize internal histories for this iteration
            internal_actor_profit_history.append([None] * num_actors)
            actor_profit_history.append([None] * num_actors)

            # Initialize convergence checks for this iteration
            convergence_check_1.append([False] * num_actors)
            convergence_check_2.append([False] * num_actors)

            # competitors = all generators not owned
            competitors = [i for i in range(self.num_generators) if i not in owner_indexes]

            # List of actors. First one is the owner coalition
            actors = [tuple(owner_indexes)] + competitors

            # Map actors -> list index
            actor_to_index = {actor: a_idx for a_idx, actor in enumerate(actors)}

            # Create player update list
            players = [owner_indexes] + competitors

            # If the heuristic is enabled, randomize the update order each iteration
            if self.heuristic:
                random.seed(iter + run_id)
                random.shuffle(players)

            # Each player updates their bid in sequence
            for p in players:
                self._build_model(p, bid_vector, cost_vector)
                self.solve()
                # # --- Enforce rational bidding floor and cap ---
                for g in (p if isinstance(p, (list, tuple, set)) else [p]):
                    alpha_new = self.model.alpha[g].value
                    # Add heuristic: enforce rational bidding floo
                    if self.heuristic:
                        alpha_new = max(alpha_new, cost_vector[g])   # rational floor
    
                    # Update the bid vector and store the new bid
                    bid_vector[g] = alpha_new
                    alpha_history[iter][g] = alpha_new

                # Store internal histories after player's action
                internal_actor_profit_history[iter][actor_to_index[p]] = -value(self.model.objective)
                dispatch_history[iter][g] = self.model.P_G[g].value

            # After the first 5 iterations, check for convergence
            if iter > 4:
                # First convergence check - after each player has acted
                for p in players:
                    prev_profit = internal_actor_profit_history[iter - 1][actor_to_index[p]]
                    curr_profit = internal_actor_profit_history[iter][actor_to_index[p]]

                    # Check if the profit has converged within the tolerance
                    if (curr_profit >= prev_profit * (1 - self.convergence_tol) and curr_profit <= prev_profit * (1 + self.convergence_tol)): 

                        # If True, mark player p as converged for this check
                        convergence_check_1[iter][actor_to_index[p]] = True
                
                # If all players passed the first convergence check the second check is performed
                if all(convergence_check_1[iter]):
                    # Perform economic dispatch based on current bids
                    dispatch_round, clearing_price_round = self.economic_dispatch(bid_vector)
                    dispatch_history[iter] = dispatch_round

                    # For each player, calculate and store profit after market clearing
                    for p in players:
                        # Calculate profit for the actor
                        if isinstance(p, (list, tuple, set)):
                            profit_market_clearing = sum(
                                clearing_price_round * dispatch_round[g]
                                - cost_vector[g] * dispatch_round[g]
                                for g in p
                            )
                        else:
                            profit_market_clearing = (
                                clearing_price_round * dispatch_round[p]
                                - cost_vector[p] * dispatch_round[p]
                            )
                        
                        # Store the profit after market clearing
                        actor_profit_history[iter][actor_to_index[p]] = profit_market_clearing

                        # Second convergence check - after market clearing
                        if (internal_actor_profit_history[iter][actor_to_index[p]] + self.convergence_tol >= actor_profit_history[iter][actor_to_index[p]] * (1 - self.convergence_tol) and 
                            internal_actor_profit_history[iter][actor_to_index[p]] <= actor_profit_history[iter][actor_to_index[p]] * (1 + self.convergence_tol) + self.convergence_tol):
                            convergence_check_2[iter][actor_to_index[p]] = True

                    # Store clearing price for this iteration after all players have acted
                    clearing_price_history.append(clearing_price_round)

                    # If all players passed the second convergence check, the algorithm has converged
                    if all(convergence_check_2[iter]):
                        print(f"Run id: {run_id} - Converged after {iter} full rounds.")
                        converged = True
                        print(f"  Owner profit: ${actor_profit_history[iter][actor_to_index[tuple(owner_indexes)]]:.2f}")
                        print(f"  Competitor profits: {[actor_profit_history[iter][i] for i in range(1, num_actors)]}")
                        print(f"  Owner share of total profit: {100 * actor_profit_history[iter][actor_to_index[tuple(owner_indexes)]] / sum(actor_profit_history[iter]):.2f}%")
                        break
                # If the algorithm has not converged pass to the next iteration
                else:
                    # If iter is the last iteration, perform final economic dispatch and store results
                    if iter == self.max_iter - 1:
                        dispatch_round, clearing_price_round = self.economic_dispatch(bid_vector)
                        dispatch_history[iter] = dispatch_round
                        clearing_price_history.append(clearing_price_round)
                        for j in range(self.num_generators):
                            actor_profit_history[iter][j] = (
                                clearing_price_round * dispatch_round[j]
                                - cost_vector[j] * dispatch_round[j]
                            )
            # Update iteration counter
            iter += 1
            # If maximum iterations reached without convergence, store last dispatch
            if iter == self.max_iter:
                print(f"Run id: {run_id} - Reached maximum iterations {self.max_iter} without convergence.")
                dispatch_history.append(dispatch_history[iter - 1])

        # Calculate Price of Anarchy (PoA)
        PoA = sum(cost_vector[g] * dispatch_history[iter][g] for g in range(self.num_generators)) / (sum(cost_vector[g] * dispatch_ED[g] for g in range(self.num_generators)))
        # Calculate alternative PoA definition based on bids
        PoA_ish = sum(bid_vector[g] * dispatch_history[iter][g] for g in range(self.num_generators)) / (sum(cost_vector[g] * dispatch_ED[g] for g in range(self.num_generators)))

        # Store final bids and dispatch
        final_bid = bid_vector.copy()
        final_dispatch = dispatch_round

        # Store results in the results dictionary
        self.results[run_id] = {
                        "cost_vector": cost_vector,
                        "profit_history": internal_actor_profit_history,
                        "alpha_history": alpha_history,
                        "dispatch_history": dispatch_history,
                        "iterations": iter,
                        "PoA": PoA,
                        "PoA_ish": PoA_ish,
                        "final_dispatch": final_dispatch,
                        "final_bid": final_bid,
                        "final_clearing_price": clearing_price_history[-1],
                        "clearing_price_history": clearing_price_history,
                        "dispatch_ED": dispatch_ED,
                        "clearing_price_ED": clearing_price_ED,
                        "converged": converged,
                    }

    def _build_model(self, index_strategic: int | tuple[int], bid_vector: List[float], cost_vector: List[float]) -> None:
        """
        Function to build the MPEC model for the strategic player/actor. 
        This function calls sub-functions to build variables, objective and constraints, and set the index of which participant is strategic.
        The function is defined such, that either a single generator index (int) or multiple generator indexes (list/tuple) can be passed as the strategic player.

        Parameters
        ----------
        index_strategic : int or tuple[int]
            Index or indexes of the strategic generator(s).
        bid_vector : List[float]
            Array of bid values for each generator - not using the strategic players previous bid.
        cost_vector : List[float]
            Array of cost values for each generator. (Only used in the objective to calculate the profit of the strategic player).
        """

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
        self._build_objective(bid_vector, cost_vector)
        self._build_constraints(bid_vector)

    def _build_variables(self) -> None:
        """
        Function to build the Pyomo variables for the MPEC model. 
        """
        self.model.P_G = Var(self.model.n_gen, domain=Reals)
        self.model.alpha = Var(self.model.strategic_index, domain=Reals)
        self.model.lambda_dual = Var(domain=Reals)
        self.model.mu_min = Var(self.model.n_gen, domain=NonNegativeReals)
        self.model.mu_max = Var(self.model.n_gen, domain=NonNegativeReals)
        self.model.z_min = Var(self.model.n_gen, domain=Binary)
        self.model.z_max = Var(self.model.n_gen, domain=Binary)
        self.model.tau = Var(self.model.strategic_index, self.model.n_gen - self.model.strategic_index, domain=Binary)

    def _build_objective(self, bid_vector: List[float], cost_vector: List[float]) -> None:
        """
        Function to build the Pyomo objective for the MPEC model. 

        Parameters
        ----------
        bid_vector : List[float]
            Array of bid values for each generator.
        cost_vector : List[float]
            Array of cost values for each generator. (Only used in the objective to calculate the profit of the strategic player).
        """

        # Strong duality substitution
        term_lambda = self.model.lambda_dual * self.demand

        term_duals = sum(
            self.model.mu_min[i] * self.Pmin[i] 
            - self.model.mu_max[i] * self.Pmax[i]
            for i in self.model.n_gen
        )

        term_non_strat = sum(
            bid_vector[i] * self.model.P_G[i]
            for i in (self.model.n_gen - self.model.strategic_index)
        )

        term_duals_strat_1 = sum(
            self.model.mu_min[i] * self.Pmin[i]
            for i in self.model.strategic_index
        )

        term_duals_strat_2 = sum(
            self.model.mu_max[i] * self.Pmax[i] 
            for i in self.model.strategic_index
        )

        term_costs_strat = sum(
            cost_vector[i] * self.model.P_G[i]
            for i in self.model.strategic_index
        )

        self.model.objective = Objective(
            expr = -(term_lambda + term_duals - term_non_strat - term_duals_strat_1 + term_duals_strat_2) + term_costs_strat,
            sense = minimize
        )

    def _build_constraints(self, bid_vector: List[float]) -> None:
        """
        Function to build the Pyomo constraints for the MPEC model. STILL NEED COMMENTS ON EACH OF THE CONSTRAINTS
        """

        # Alpha constraints
        def alpha_min_rule(m, i):
            return m.alpha[i] >= self.alpha_min

        def alpha_max_rule(m, i):
            return m.alpha[i] <= self.alpha_max

        self.model.alpha_min_constr = Constraint(self.model.strategic_index, rule=alpha_min_rule)
        self.model.alpha_max_constr = Constraint(self.model.strategic_index, rule=alpha_max_rule)

        # Power balance constraint
        self.model.power_balance = Constraint(expr=sum(self.model.P_G[i] for i in range(self.num_generators)) == self.demand)


        # Stationarity constraints are writen in two constraints
        def stationarity_rule(m, i):
            return m.alpha[i] - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0

        self.model.stationarity = Constraint(self.model.strategic_index, rule=stationarity_rule)

        def stationarity_non_strategic_rule(m, i):
            return bid_vector[i] - m.lambda_dual - m.mu_min[i] + m.mu_max[i] == 0

        self.model.stationarity_non_strategic = Constraint(
            self.model.n_gen - self.model.strategic_index, rule=stationarity_non_strategic_rule
        )

        # ------------------------
        # Big-M + binary formulation
        # ------------------------
        M = 10000
         # Small positive constant to enforce strict inequality

        epsilon = 0.01

        def alpha_not_equal_lower(m, p, i):
            return m.alpha[p] - bid_vector[i] >= epsilon - M*(1 - m.tau[p, i])

        def alpha_not_equal_upper(m, p, i):
            return bid_vector[i] - m.alpha[p] >= epsilon - M*(m.tau[p, i])

        self.model.alpha_not_equal_lower = Constraint(
            self.model.strategic_index, self.model.n_gen - self.model.strategic_index, rule=alpha_not_equal_lower
        )

        self.model.alpha_not_equal_upper = Constraint(
            self.model.strategic_index, self.model.n_gen - self.model.strategic_index, rule=alpha_not_equal_upper
        )

        def min_primal_feasibility(m, i):
            return  0 <= m.P_G[i] - self.Pmin[i]
        self.model.min_primal_feasibility = Constraint(self.model.n_gen, rule=min_primal_feasibility)
        
        def min_complementarity_1(m, i):
            # if z_min[i] = 0 -> P_G[i] = Pmin[i]
            return m.P_G[i] - self.Pmin[i] <= M * m.z_min[i]
        self.model.min_complementarity_1 = Constraint(self.model.n_gen, rule=min_complementarity_1)
        
        def mu_min_lower_rule(m, i):
            return 0 <= m.mu_min[i]
        self.model.mu_min_lower = Constraint(self.model.n_gen, rule=mu_min_lower_rule)

        def min_complementarity_2(m, i):
            # if z_min[i] = 1 -> mu_min[i] = 0
            return m.mu_min[i] <= M * (1 - m.z_min[i])
        self.model.min_complementarity_2 = Constraint(self.model.n_gen, rule=min_complementarity_2)

        def max_primal_feasibility(m, i):
            return  0 <= self.Pmax[i] - m.P_G[i]
        self.model.max_primal_feasibility = Constraint(self.model.n_gen, rule=max_primal_feasibility)

        def max_complementarity_1(m, i):
            # if z_max[i] = 0 -> P_G[i] = Pmax[i]
            return self.Pmax[i] - m.P_G[i] <= M * m.z_max[i]
        self.model.max_complementarity_1 = Constraint(self.model.n_gen, rule=max_complementarity_1)        
        
        def max_complementarity_2(m, i):
            # if z_max[i] = 1 -> mu_max[i] = 0
            return m.mu_max[i] <= M * (1 - m.z_max[i])
        self.model.max_complementarity_2 = Constraint(self.model.n_gen, rule=max_complementarity_2)       

    def solve(self, solver_name: str = "gurobi") -> None:
        """
        Solve the optimization model.

        Parameters
        ----------
        solver_name : str, optional
            Name of the solver to use (default: "gurobi").
        """

        # Create solver
        solver = SolverFactory(solver_name)

        # Solve
        results = solver.solve(self.model, tee=False)

        # Check solver status
        if not (results.solver.status == 'ok') and not (results.solver.termination_condition == 'optimal'):
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)

    def MPEC(self, index_strategic: int | tuple[int], cost_vector: List[float]) -> tuple[List[float], float, dict]:
        """
        Solve the MPEC problem for the strategic player/actor.

        Parameters
        ----------
        index_strategic : int or tuple[int]
            Index or indexes of the strategic generator(s).
        cost_vector : List[float]
            Array of cost values for each generator. (Only used in the objective to calculate the profit of the strategic player).
        
        Returns
        -------
        dispatch : List[float]
            Optimal dispatch for each generator.
        clearing_price : float
            Market clearing price.
        alpha_values : Dict[float]
            Dictionary of alpha values for each strategic generator.
        """
        self._build_model(index_strategic = index_strategic, bid_vector = cost_vector, cost_vector = cost_vector)
        self.solve()
        dispatch = [self.model.P_G[i].value for i in range(self.num_generators)]
        clearing_price = self.model.lambda_dual.value
        alpha_values = {i: self.model.alpha[i].value for i in self.model.strategic_index}
        return dispatch, clearing_price, alpha_values

    def economic_dispatch(self, cost_vector: List[float]) -> tuple[List[float], float]:
        """
        Solve the economic dispatch problem with the bids placed.

        Parameters
        ----------
        cost_vector : List[float]
            Array of cost/bids values for each generator.
        
        Returns
        -------
        dispatch : List[float]
            Optimal dispatch for each generator.
        clearing_price : float
            Market clearing price.
        """
        model = ConcreteModel()
        model.n_gen = Set(initialize=range(self.num_generators))
        model.P_G = Var(model.n_gen, domain=NonNegativeReals)
        model.objective = Objective(
            expr=sum(cost_vector[i] * model.P_G[i] for i in model.n_gen),
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
            return dispatch, clearing_price
        else:
            print("Solver status:", results.solver.status)
            print("Termination condition:", results.solver.termination_condition)
    
    def plot_merit_order_curve(self, run_id: int) -> None:
        """
        Plot the merit order curve for both the economic dispatch and the strategic producer case for a given run_id.
        Saves the plot as a PNG file in the outputs/{exercise}/merit_order_curves/ directory.

        Parameters
        ----------
        run_id : int
            Identifier for the run/experiment to plot.
        """

        if "owner_indexes" not in self.results[run_id]:
            owner_indexes = set()
        else:
            owner_indexes = set(self.results[run_id]['owner_indexes'])

        cost_vector = self.results[run_id]['cost_vector']
        bid_vector = self.results[run_id]['final_bid']
        dispatch_ED = self.results[run_id]['dispatch_ED']
        clearing_price_ED = self.results[run_id]['clearing_price_ED']
        dispatch_SP = self.results[run_id]['final_dispatch']
        clearing_price_SP = self.results[run_id]['final_clearing_price']

        cost_array = np.array(cost_vector)
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
        sp_costs = np.array(bid_vector)
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
        # if owner_indexes:
        #     owned_gens = ', '.join([f'G{i}' for i in sorted(owner_indexes)])
        #     plt.title(f'Merit Order Curve: ED vs Strategic Producer. Run ID: {run_id}\n'
        #             f'Owned generators: {owned_gens} (marked with *)', 
        #             fontsize=14, fontweight='bold')
        # else:
        #     plt.title(f'Merit Order Curve: ED vs Strategic Producer. Run ID: {run_id}', 
        #             fontsize=14, fontweight='bold')

        plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                fontsize=10, ncol=3, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/merit_order", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/merit_order/merit_order_curve_run_{run_id}.png', dpi=300)
        # plt.show()
        plt.close()

    def plot_alpha_over_iterations(self, run_id: int) -> None:
        """
        Plot the evolution of alpha (bids) over iterations for a given run_id.
        Saves the plot as a PNG file in the outputs/{exercise}/alpha/ directory.

        Parameters
        ----------
        run_id : int
            Identifier for the run/experiment to plot.
        """
        
        alpha_history = self.results[run_id]['alpha_history']

        cost_vector = self.results[run_id]['cost_vector']

        alpha_history = np.array(alpha_history)
        plt.figure(figsize=(10, 6))
        for i in range(self.num_generators):
            plt.plot(alpha_history[:, i], marker='o', label=f'Generator {i} - Init Cost {cost_vector[i]:.0f}')
        plt.xlabel('Iteration')
        plt.ylabel('Alpha (Bid)')
        # plt.title('Alpha Evolution Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/alpha", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/alpha/alpha_evolution_run_{run_id}.png', dpi=300)
        plt.close()

    def plot_clearing_price_over_iterations(self, run_id: int) -> None:
        """
        Plot the evolution of clearing price over iterations for a given run_id.
        Saves the plot as a PNG file in the outputs/{exercise}/clearing_price/ directory.
        
        Parameters
        ----------
        run_id : int
            Identifier for the run/experiment to plot.
        """

        clearing_price_history = self.results[run_id]['clearing_price_history']

        clearing_price_history = np.array(clearing_price_history)
        plt.figure(figsize=(10, 6))
        plt.plot(clearing_price_history, marker='o', label=f'Clearing Price')
        plt.xlabel('Iteration')
        plt.ylabel('Clearing Price')
        # plt.title('Clearing Price Evolution Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/clearing_price", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/clearing_price/clearing_price_evolution_run_{run_id}.png', dpi=300)
        plt.close()

    def plot_dispatch_over_iterations(self, run_id: int) -> None:
        """
        Plot the evolution of dispatch over iterations for a given run_id.
        Saves the plot as a PNG file in the outputs/{exercise}/dispatch/ directory.
        
        Parameters
        ----------
        run_id : int
            Identifier for the run/experiment to plot.
        """

        dispatch_history = self.results[run_id]['dispatch_history']
        economic_dispatch = self.results[run_id]['dispatch_ED']

        cost_vector = self.results[run_id]['cost_vector']

        economic_dispatch = np.array([economic_dispatch] * len(dispatch_history))
        dispatch_history = np.array(dispatch_history)
        
        plt.figure(figsize=(10, 6))
        for i in range(self.num_generators):
            plt.plot(dispatch_history[:, i], marker='o', label=f'Generator {i} - Init Cost {cost_vector[i]:.2f}')
            if i == 0:
                plt.plot(economic_dispatch[:, i], linestyle='--', label='Economic Dispatch')
            else:
                plt.plot(economic_dispatch[:, i], linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Dispatch (MW)')
        # plt.title('Dispatch Evolution Over Iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/dispatch", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/dispatch/dispatch_evolution_run_{run_id}.png', dpi=300)
        plt.close()

    def plot_dispatch_comparison(self, run_id: int) -> None:
        """
        Creates a bar plot comparing the Economic Dispatch (ED) with the Strategic Equilibrium Dispatch (SP) for one run.
        Saves the plot as a PNG file in the outputs/{exercise}/dispatch/ directory.
        
        Parameters
        ----------
        run_id : int
            Identifier for the run/experiment to plot.
        """

        result = self.results[run_id]

        dispatch_ED = result["dispatch_ED"]
        dispatch_SP = result["final_dispatch"]

        num_g = self.num_generators
        generators = [f"G{i}" for i in range(num_g)]

        x = np.arange(num_g)
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(x - width/2, dispatch_ED, width, label="Economic Dispatch (ED)", color="seagreen")
        plt.bar(x + width/2, dispatch_SP, width, label="Equilibrium Dispatch (SP)", color="purple")

        plt.xlabel("Generator")
        plt.ylabel("Dispatch (MW)")
        plt.title("Dispatch Comparison: Economic vs Equilibrium")

        plt.xticks(x, generators)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/dispatch", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/dispatch/dispatch_comparison_run_{run_id}.png', dpi=300)
        plt.close()

    def plot_PoA(self) -> None:
        """
        Plot the distributions of Price of Anarchy (PoA) and Improved PoA (PoA_ish) across all runs.
        Saves the plot as a PNG file in the outputs/{exercise}/PoA/ directory.
        """

        # Extract values
        PoA_values_converged = [
            self.results[run_id]['PoA']
            for run_id in self.results
            if self.results[run_id]['converged']
        ]
        PoA_values = [
            self.results[run_id]['PoA']
            for run_id in self.results
        ]

        PoA_ish_values_converged = [
            self.results[run_id]['PoA_ish']
            for run_id in self.results
            if self.results[run_id]['converged']
        ]

        PoA_ish_values = [
            self.results[run_id]['PoA_ish']
            for run_id in self.results
        ]

        # --- Compute equal-width global bins ---
        all_min = min(min(PoA_values), min(PoA_ish_values))
        all_max = max(max(PoA_values), max(PoA_ish_values))
        n_bins = 20

        bins = np.linspace(all_min, all_max, n_bins + 1)

        # --- Create subplots ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # -------------------------------------------------------
        # Left subplot: PoA
        # -------------------------------------------------------
        ax = axes[0]
        ax.hist(PoA_values, bins=bins, color='lightgray', edgecolor='black', label='All Runs')
        ax.hist(PoA_values_converged, bins=bins, color='skyblue', edgecolor='black', alpha=0.5, label='Converged Runs')

        ax.set_xlabel('Price of Anarchy (PoA)')
        ax.set_ylabel('Frequency')
        # ax.set_title('PoA Distribution')
        ax.grid(True)
        ax.legend()

        # -------------------------------------------------------
        # Right subplot: PoA-ish
        # -------------------------------------------------------
        ax = axes[1]
        ax.hist(PoA_ish_values, bins=bins, color='lightgray', edgecolor='black', label='All Runs')
        ax.hist(PoA_ish_values_converged, bins=bins, color='skyblue', edgecolor='black', alpha=0.5, label='Converged Runs')

        ax.set_xlabel('Improved PoA (PoA_ish)')
        # ax.set_title('Improved PoA Distribution')
        ax.grid(True)
        ax.legend()

        # --- Layout ---
        fig.tight_layout()
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/PoA", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/PoA/PoA_distributions.png', dpi=300)
        plt.close()

    def plot_profits(self, run_id: int) -> None:
        """
        Plot generator-level profits over iterations for a given run_id.
        Saves the plot as a PNG file in the outputs/{exercise}/profits/ directory.
        
        Parameters
        ----------
        run_id : int
            Identifier for the run/experiment to plot.
        """

        profit_history = self.results[run_id]['internal_profit_history']

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
        # if owner_indexes:
        #     owned_gens = ', '.join([f'G{i}' for i in sorted(owner_indexes)])
        #     plt.title(f'Profit Evolution Over Iterations - Run ID: {run_id}\n'
        #             f'Generators marked with * are owned together: {owned_gens}', 
        #             fontsize=13, fontweight='bold')
        # else:
        #     plt.title(f'Profit Evolution Over Iterations - Run ID: {run_id}', 
        #             fontsize=13, fontweight='bold')
        
        plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                ncol=4, framealpha=0.9, fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make room for legend below
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/profits", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/profits/profit_evolution_run_{run_id}.png', dpi=300)
        plt.close()
        # plt.show()

    def plot_actor_profits(self, run_id: int) -> None:
        """
        Plot actor-level profits over iterations for ownership scenarios.
        Shows the owner's combined profit and each competitor's individual profit.
        Saves the plot as a PNG file in the outputs/{exercise}/actor_profits/ directory.

        Parameters
        ----------
        run_id : int
            Identifier for the run/experiment to plot.
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
        # plt.title(f'Actor Profits Over Iterations - Run ID: {run_id}{conv_text}\n'
        #         f'Owner controls: {owned_gens}', 
        #         fontsize=14, fontweight='bold')
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
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/actor_profits", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/actor_profits/actor_profits_run_{run_id}.png', dpi=300)
        plt.close()

    def plot_merit_order_from_MPEC(self, strategic_index: int, cost_vector: List[float]) -> None:
        """
        Plot the merit order curves from both Economic Dispatch (ED) and the Strategic Producer (SP) MPEC solution.
        Saves the plot as a PNG file in the outputs/{exercise}/merit_order_MPEC/ directory.

        Parameters
        ----------
        strategic_index : int
            Index of the strategic generator.
        cost_vector : List[float]
            Array of cost values for each generator.
        """

        # --- ED ---
        _, clearing_price_ED = self.economic_dispatch(cost_vector)
        cost_ED = np.array(cost_vector)

        # --- SP ---
        dispatch_SP, clearing_price_SP, alpha_values = self.MPEC(
            index_strategic=strategic_index,
            cost_vector=cost_vector
        )

        alpha_SP = np.array([alpha_values.get(i, cost_vector[i]) for i in range(self.num_generators)])
        pmax_array = np.array(self.Pmax)

        # --- SORT ---
        idx_ED = np.argsort(cost_ED)
        idx_SP = np.argsort(alpha_SP)

        sorted_cost_ED = cost_ED[idx_ED]
        sorted_cost_SP = alpha_SP[idx_SP]

        sorted_caps_ED = pmax_array[idx_ED]
        sorted_caps_SP = pmax_array[idx_SP]

        # --- Build curves ---
        def build_curve(costs, caps):
            x_vals, y_vals = [0], [0]
            cum = 0
            for c, cap in zip(costs, caps):
                x_vals.append(cum)
                y_vals.append(c)
                cum += cap
                x_vals.append(cum)
                y_vals.append(c)
            return x_vals, y_vals

        x_ED, y_ED = build_curve(sorted_cost_ED, sorted_caps_ED)
        x_SP, y_SP = build_curve(sorted_cost_SP, sorted_caps_SP)

        # --- Plot ---
        plt.figure(figsize=(14, 6))

        plt.step(x_ED, y_ED, where='post', color='blue', linewidth=2, label="ED Supply Curve")
        plt.step(x_SP, y_SP, where='post', color='purple', linestyle='--', linewidth=2, label="SP Supply Curve")

        demand = self.demand
        plt.axvline(demand, color='red', linestyle='--', linewidth=2, label=f"Demand = {demand}")

        plt.scatter([demand], [clearing_price_ED], color='green', s=130, marker='o',
                    edgecolors='black', label=f"ED Clearing Price = {clearing_price_ED:.2f}")

        plt.scatter([demand], [clearing_price_SP], color='black', s=150, marker='X',
                    label=f"SP Clearing Price = {clearing_price_SP:.2f}")

        # --- Adaptive label offsets (the important fix!) ---
        y_offset = max(max(sorted_cost_ED), max(sorted_cost_SP)) * 0.05

        # ED labels (above)
        cum = 0
        for idx in idx_ED:
            cap = pmax_array[idx]
            midpoint = cum + cap / 2
            plt.text(midpoint, cost_ED[idx] + y_offset,
                    f"G{idx}", ha="center", va="bottom", fontsize=9)
            cum += cap

        # SP labels (below)
        cum_SP = 0
        for idx in idx_SP:
            cap = pmax_array[idx]
            midpoint = cum_SP + cap / 2
            plt.text(midpoint, alpha_SP[idx] - y_offset,
                    f"G{idx}", ha="center", va="top", fontsize=9, color="purple")
            cum_SP += cap

        # plt.title(title, fontsize=15, fontweight='bold')
        plt.xlabel("Quantity (MW)", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center', 
                fontsize=10, ncol=3, framealpha=0.9)
        plt.tight_layout(rect=[0, 0, 1, 0.95])   # extra space if needed
        os.makedirs(f"Assignment_scripts/figures/{self.exercise}/merit_order_MPEC", exist_ok=True)
        plt.savefig(f'Assignment_scripts/figures/{self.exercise}/merit_order_MPEC/merit_order_MPEC_demand_{self.demand}.png', dpi=300)
        plt.close()

    def build_iteration_table(self, run_id) -> pd.DataFrame:
        """
        Build a DataFrame summarizing each iteration's bids, dispatches,
        market prices, and strategic profits for all generators for a given run_id.
        """

        result = self.results[run_id]
        num_g = self.num_generators

        internal_bids = result["internal_bid_history"]
        internal_dispatch = result["internal_dispatch_history"]
        internal_profits = result["internal_profit_history"]
        internal_prices = result["internal_clearing_price_history"]
        player_order_hist = result["player_order_history"]

        rows = []

        round_counter = 0   # counts rounds globally across iterations

        for iter_idx, update_order in enumerate(player_order_hist):

            for round_idx, p in enumerate(update_order):

                # Round-level bid vector
                bids = internal_bids[round_counter]
                dispatches = internal_dispatch[round_counter]

                # Clearing price for this round = price the model produced when p acted
                market_price = internal_prices[iter_idx][p]

                # Strategic profit for this round
                strat_profit = internal_profits[iter_idx][p]

                # Build row
                row = {
                    "Iter": iter_idx + 1,
                    "Rnd": round_idx + 1,
                    "Strategic Player": f"G{p}",
                    "Market Price": round(market_price, 3) if market_price is not None else None,
                    "Strategic Profit": round(strat_profit, 3) if strat_profit is not None else None,
                }

                # Add bids for all generators
                for g in range(num_g):
                    row[f"Bid G{g}"] = round(bids[g], 3) if bids[g] is not None else None

                # Add dispatch for all generators
                for g in range(num_g):
                    row[f"Dispatch G{g}"] = round(dispatches[g], 3) if dispatches[g] is not None else None

                rows.append(row)

                round_counter += 1

        return pd.DataFrame(rows)

if __name__ == "__main__":
    
    # Exercise 2

    Pmin = [ 0,  0,  0,  0]
    Pmax = [60, 80, 55, 70]

    demand = 150

    cost = [200, 300, 400, 500]

    # Run single experiment - exercise 3
    epec = EPEC(
                Pmin = Pmin, 
                Pmax = Pmax, 
                demand = demand, 
                cost = cost,
    )

    epec.plot_merit_order_from_MPEC(strategic_index=0, cost_vector=cost)

    demand = 190

    # Run single experiment - exercise 3
    epec = EPEC(
                Pmin = Pmin, 
                Pmax = Pmax, 
                demand = demand, 
                cost = cost,
    )

    epec.plot_merit_order_from_MPEC(strategic_index=0, cost_vector=cost)

