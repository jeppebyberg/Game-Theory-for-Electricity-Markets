import pyomo.environ as pyo

from mpec_model import mpec, Actor, Generator, market_clearing
from br_algo import StrategicRotation, Actor, Generator


def run_until_convergence(demand, actors, generators, solver_name="gurobi", tee=False, tol=0.05, max_iterations=100):
    """
    Runs StrategicRotation until actor profits converge within tolerance.

    Parameters
    ----------
    demand : float
        Total demand.
    actors : list[Actor]
        List of actors.
    generators : list[Generator]
        List of generators.
    solver_name : str
        Solver to use.
    tee : bool
        Show solver output if True.
    tol : float
        Convergence tolerance (relative difference, default 5%).
    max_iterations : int
        Maximum number of iterations (safety stop).

    Returns
    -------
    all_results : list
        Results from all iterations until convergence.
    """
    n_actors = len(actors)
    iteration = 0
    profit_history = {a.actor_id: [] for a in actors}
    all_results = []
    last_bids = None  # Initialize last_bids to None

    # Calculate social optimum price once using market_clearing
    mc = market_clearing(demand=demand, generators=generators)
    mc.solve(solver_name=solver_name, tee=False)
    social_optimum_price = mc.model.dual[mc.model.balance]
    social_optimum_dispatch = sum(pyo.value(mc.model.p[g]) for g in mc.model.G)

    while iteration < max_iterations:
        iteration += 1

        # Run one full rotation (k=1)
        rotation = StrategicRotation(
            demand=demand,
            actors=actors,
            generators=generators,
            iterations=1,
            solver_name=solver_name,
            tee=tee
        )
        results = rotation.run(last_bids=last_bids)
        
        # Update iteration numbers to reflect actual convergence iteration
        for res in results:
            res["iteration"] = iteration
        
        all_results.extend(results)

        # update last_bids to final bids of this rotation
        last_bids = results[-1]["bids"]

        # Collect profits from this iteration
        for res in results:
            profit_history[res["strategic_actor"]].append(res["objective"])

        # Check convergence if we have at least 2 iterations
        if iteration > 1:
            converged = True
            for actor_id, profits in profit_history.items():
                if len(profits) < 2:
                    converged = False
                    break
                prev, curr = profits[-2], profits[-1]
                if prev != 0:
                    lower = (1 - tol) * prev
                    upper = (1 + tol) * prev
                    if not (lower <= curr <= upper):
                        converged = False
                        break
            if converged:
                print(f"\n✅ Converged after {iteration} iterations.")
                break
    
    # Calculate price of anarchy after convergence using highest bid
    if all_results:
        # Final equilibrium dispatch after convergence
        final_dispatch = all_results[-1]["dispatch"]

        # Equilibrium cost: sum of (true cost * equilibrium dispatch)
        eq_cost = sum(
            gen.cost * final_dispatch[i]
            for i, gen in enumerate(generators)
        )

        # Social optimum cost: sum of (true cost * optimum dispatch)
        opt_cost = sum(
            gen.cost * pyo.value(mc.model.p[g])
            for g, gen in enumerate(generators)
        )

        # Avoid divide-by-zero
        if opt_cost > 0:
            price_of_anarchy = eq_cost / opt_cost
        else:
            price_of_anarchy = None
    else:
        price_of_anarchy = None

    # ✅ print all results using StrategicRotation's print_results
    printer = StrategicRotation(demand, actors, generators)
    printer.print_results(all_results)
    
    # Print price of anarchy summary
    if price_of_anarchy is not None:
        print(f"\n{'='*60}")
        print(f"PRICE OF ANARCHY SUMMARY")
        print(f"{'='*60}")
        print(f"Social Optimum Price: {social_optimum_price:.2f}")
        print(f"Price of Anarchy: {price_of_anarchy:.2f}")
        print(f"{'='*60}")

    return all_results, profit_history, price_of_anarchy



# Example usage
if __name__ == "__main__":
    g0 = Generator(p_max=60, cost=2)
    g1 = Generator(p_max=80, cost=3)
    g2 = Generator(p_max=55, cost=4)

    actors = [
        Actor(actor_id=0, generators=[g0], is_strategic=False),
        Actor(actor_id=1, generators=[g1], is_strategic=False),
        Actor(actor_id=2, generators=[g2], is_strategic=False),
    ]

    generators = [g0, g1, g2]

    results, profits, poa = run_until_convergence(
        demand=150,
        actors=actors,
        generators=generators,
        solver_name="gurobi",
        tol=0.01
    )

    # Print summary
    for actor_id, profit_list in profits.items():
        print(f"\nActor {actor_id} profit history: {profit_list}")
    
    print(f"\nFinal Price of Anarchy: {poa:.2f}")