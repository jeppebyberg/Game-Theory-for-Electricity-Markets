import pyomo.environ as pyo
from copy import deepcopy
from mpec_model import mpec, Actor, Generator, market_clearing  # replace with actual filename


class StrategicRotation:
    def __init__(self, demand, actors, generators, iterations=1, solver_name="gurobi", tee=False):
        """
        Rotates the strategic bidder across actors in k iterations.
        One iteration (k) = each actor gets one chance to be strategic.

        Parameters
        ----------
        demand : float
            Total demand in the mp.
        actors : list[Actor]
            List of actors (with their generators).
        generators : list[Generator]
            List of generators.
        iterations : int
            Number of iterations (k). Each iteration = |actors| rounds.
        solver_name : str
            Solver to use (default 'gurobi').
        tee : bool
            Whether to display solver output.
        """
        self.demand = demand
        self.actors = deepcopy(actors)   # keep original safe
        self.generators = generators
        self.iterations = iterations
        self.solver_name = solver_name
        self.tee = tee
        self.results = []

    def run(self, last_bids = None):
        
        round_counter = 0

        for k in range(self.iterations):
            for actor in self.actors:
                # Reset all actors to non-strategic
                for a in self.actors:
                    a.is_strategic = False
                actor.is_strategic = True

                # Build mpec model with fixed bids
                mp = mpec(
                    demand=self.demand,
                    actors=self.actors,
                    generators=self.generators,
                    fixed_bids=last_bids
                )

                # Solve
                mp.solve(solver_name=self.solver_name, tee=self.tee)

                # Store results
                round_counter += 1
                round_info = {
                    "iteration": k + 1,
                    "round": round_counter,
                    "strategic_actor": actor.actor_id,
                    "objective": -pyo.value(mp.model.obj),
                    "bids": {i: pyo.value(mp.model.alpha[i]) for i in mp.model.G},
                    "dispatch": {i: pyo.value(mp.model.p[i]) for i in mp.model.G},
                    "lambda": pyo.value(mp.model.lmbda),
                }
                self.results.append(round_info)

                # Update last bids for next round
                last_bids = round_info["bids"]

        return self.results

    def print_results(self, results=None):
        """
        Print results in a table format comparing MPEC to market clearing.
        
        Parameters
        ----------
        results : list, optional
            Results to print. If None, uses self.results.
        """
        data = results if results is not None else self.results
        
        # Calculate market clearing once for comparison
        mc = market_clearing(demand=self.demand, generators=self.generators)
        mc.solve(solver_name=self.solver_name, tee=False)
        mc_price = mc.model.dual[mc.model.balance]
        mc_dispatch = {i: pyo.value(mc.model.p[i]) for i in mc.model.G}
        mc_costs = {i: self.generators[i].cost for i in range(len(self.generators))}
        
        for res in data:
            print(f"\n{'='*100}")
            print(f"ITERATION {res['iteration']} | ROUND {res['round']} | Strategic Actor: {res['strategic_actor']}")
            print(f"{'='*100}")
            
            # Header
            print(f"\n{'Generator':<12} {'True Cost':<12} {'MC Dispatch':<15} {'MPEC Bid':<12} {'MPEC Dispatch':<15} {'Markup':<12}")
            print(f"{'-'*100}")
            
            # Generator rows
            for gen_idx in sorted(res['bids'].keys()):
                true_cost = mc_costs[gen_idx]
                mc_disp = mc_dispatch[gen_idx]
                mpec_bid = res['bids'][gen_idx]
                mpec_disp = res['dispatch'][gen_idx]
                markup = mpec_bid - true_cost
                
                strategic_marker = " *" if gen_idx == res['strategic_actor'] else ""
                
                print(f"G{gen_idx}{strategic_marker:<10} "
                    f"${true_cost:<11.2f} "
                    f"{mc_disp:<15.2f} "
                    f"${mpec_bid:<11.2f} "
                    f"{mpec_disp:<15.2f} "
                    f"${markup:+.2f}")
            
            print(f"{'-'*100}")
            
            # Summary statistics
            print(f"\n{'Market Clearing Price:':<30} MC: ${mc_price:.2f}/MWh  |  MPEC: ${res['lambda']:.2f}/MWh")
            price_increase = res['lambda'] - mc_price
            pct_increase = (price_increase / mc_price * 100) if mc_price > 0 else 0
            print(f"{'Price Increase:':<30} ${price_increase:.2f}/MWh ({pct_increase:+.2f}%)")
            print(f"{'Strategic Actor Profit:':<30} ${res['objective']:.2f}")
            
            # Total system cost comparison
            mc_total_cost = sum(mc_costs[i] * mc_dispatch[i] for i in mc_costs.keys())
            mpec_total_cost = sum(mc_costs[i] * res['dispatch'][i] for i in mc_costs.keys())
            print(f"{'Total Generation Cost:':<30} MC: ${mc_total_cost:.2f}  |  MPEC: ${mpec_total_cost:.2f}")
            
            print(f"\n* = Strategic bidder")


# Example usage
if __name__ == "__main__":
    # Same setup as your script
    g0 = Generator(p_max=400, cost=15)
    g1 = Generator(p_max=650, cost=17)
    g2 = Generator(p_max=400, cost=12)

    actors = [
        Actor(actor_id=0, generators=[g0], is_strategic=False),
        Actor(actor_id=1, generators=[g1], is_strategic=False),
        Actor(actor_id=2, generators=[g2], is_strategic=False),
    ]

    generators = [g0, g1, g2]


    rotation = StrategicRotation(demand=1000, actors=actors, generators=generators, iterations=3)
    results = rotation.run()
    rotation.print_results()
