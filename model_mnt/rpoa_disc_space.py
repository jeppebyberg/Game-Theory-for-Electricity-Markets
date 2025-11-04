import pandas as pd
import itertools
from mpec_model import Generator, Actor
from br_convergence import run_until_convergence


class DiscretizedSpace:
    def __init__(self, s_space_df, p_max_dict=None, demand=1000):
        """
        Creates a discretized strategy space for generator costs.
        
        Parameters
        ----------
        s_space_df : pd.DataFrame
            DataFrame where each column represents a generator and each row 
            contains possible cost values for that generator.
        p_max_dict : dict, optional
            Dictionary mapping generator names to their p_max values.
            If None, defaults to 350 for all generators.
        demand : float
            Total demand in the market.
        """
        self.s_space = s_space_df
        self.generator_names = s_space_df.columns.tolist()
        self.p_max_dict = p_max_dict or {gen: 350 for gen in self.generator_names}
        self.demand = demand
        self.results_all = []
        
    def get_combinations(self):
        """
        Generate all combinations of generator costs.
        
        Returns
        -------
        list of dict
            List of dictionaries, each mapping generator names to costs.
        """
        cost_options = [self.s_space[gen].tolist() for gen in self.generator_names]
        combinations = list(itertools.product(*cost_options))
        
        return [dict(zip(self.generator_names, costs)) for costs in combinations]
    
    def run_all_combinations(self, solver_name="gurobi", tee=False, tol=0.05, max_iterations=50):
        """
        Run convergence analysis for all cost combinations.
        
        Parameters
        ----------
        solver_name : str
            Solver to use.
        tee : bool
            Show solver output if True.
        tol : float
            Convergence tolerance.
        max_iterations : int
            Maximum iterations for convergence.
            
        Returns
        -------
        list of dict
            Results for each combination with metadata.
        """
        combinations = self.get_combinations()
        total_combinations = len(combinations)
        
        print(f"Running {total_combinations} combinations...")
        print(f"Generators: {self.generator_names}\n")
        
        for idx, cost_dict in enumerate(combinations, 1):
            print(f"\n{'='*60}")
            print(f"Combination {idx}/{total_combinations}")
            print(f"Costs: {cost_dict}")
            print(f"{'='*60}")
            
            # Create generators with the cost combination
            generators = []
            for gen_name in self.generator_names:
                gen = Generator(
                    p_max=self.p_max_dict[gen_name],
                    cost=cost_dict[gen_name]
                )
                generators.append(gen)
            
            # Create actors (one actor per generator in this setup)
            actors = [
                Actor(actor_id=i, generators=[gen], is_strategic=False)
                for i, gen in enumerate(generators)
            ]
            
            # Run convergence
            results, profit_history, price_of_anarchy = run_until_convergence(
                demand=self.demand,
                actors=actors,
                generators=generators,
                solver_name=solver_name,
                tee=tee,
                tol=tol,
                max_iterations=max_iterations
            )
            
            # Store results with metadata
            combination_result = {
                "combination_id": idx,
                "costs": cost_dict,
                "results": results,
                "profit_history": profit_history,
                "price_of_anarchy": price_of_anarchy,
                "converged": len(results) < max_iterations * len(actors)
            }
            self.results_all.append(combination_result)
        
        print(f"\n{'='*60}")
        print(f"Completed all {total_combinations} combinations!")
        print(f"{'='*60}")
        
        return self.results_all
    
    def summarize_results(self):
        """Print a summary of all combination results."""
        if not self.results_all:
            print("No results to summarize. Run run_all_combinations() first.")
            return
        
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL COMBINATIONS")
        print(f"{'='*60}\n")
        
        # Track maximum PoA
        max_poa = float('-inf')
        max_poa_costs = None
        max_poa_combo_id = None
        
        for combo in self.results_all:
            print(f"Combination {combo['combination_id']}: {combo['costs']}")
            print(f"  Converged: {combo['converged']}")
            print(f"  Total iterations: {len(combo['results']) // len(self.generator_names)}")
            
            # Show final profits
            final_profits = {
                actor_id: profits[-1] if profits else 0
                for actor_id, profits in combo['profit_history'].items()
            }
            print(f"  Final profits: {final_profits}")
            
            # Show price of anarchy
            if combo['price_of_anarchy'] is not None:
                print(f"  Price of Anarchy: {combo['price_of_anarchy']:.2f}")
                
                # Track maximum PoA
                if combo['price_of_anarchy'] > max_poa:
                    max_poa = combo['price_of_anarchy']
                    max_poa_costs = combo['costs']
                    max_poa_combo_id = combo['combination_id']
            else:
                print(f"  Price of Anarchy: N/A")
            print()
        
        # Print maximum PoA summary
        if max_poa > float('-inf'):
            print(f"\n{'='*60}")
            print("MAXIMUM PRICE OF ANARCHY")
            print(f"{'='*60}")
            print(f"Maximum PoA: {max_poa:.2f}")
            print(f"Combination ID: {max_poa_combo_id}")
            print(f"Generator Costs: {max_poa_costs}")
            print(f"{'='*60}\n")
            
            # Store as instance variables for easy access
            self.rpoa = max_poa
            self.rpoa_costs = max_poa_costs
            self.rpoa_combo_id = max_poa_combo_id
        else:
            print("\nNo valid Price of Anarchy values found.")
            self.rpoa = None
            self.rpoa_costs = None
            self.rpoa_combo_id = None


# Example usage
if __name__ == "__main__":
    # Define strategy space
    s_space = pd.DataFrame(
        [[15, 17, 20],
         [12, 16, 13],
         [14, 19, 18]],
        columns=['g0', 'g1', 'g2']
    )
    
    # Optional: specify different p_max for each generator
    p_max_dict = {'g0': 350, 'g1': 350, 'g2': 350}
    
    # Create discretized space
    disc_space = DiscretizedSpace(
        s_space_df=s_space,
        p_max_dict=p_max_dict,
        demand=1000
    )
    
    # Run all combinations
    all_results = disc_space.run_all_combinations(
        solver_name="gurobi",
        tee=False,
        tol=0.005,
        max_iterations=50
    )
    
    # Print summary
    disc_space.summarize_results()
    
    # Access maximum PoA results
    print(f"\nAccessing rpoa variable:")
    print(f"Maximum PoA (rpoa): {disc_space.rpoa:.2f}")
    print(f"Costs leading to rpoa: {disc_space.rpoa_costs}")
    print(f"Combination ID: {disc_space.rpoa_combo_id}")