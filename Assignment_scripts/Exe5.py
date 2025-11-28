from Assignment_scripts.EPEC import EPEC

if __name__ == "__main__":
    
    # Exercise 5
    Pmin = [ 0,  0,  0,  0,  0,  0,  0]
    Pmax = [30, 30, 30, 30, 30, 30, 30]

    cost = [1, 1.5, 2.5, 24, 25, 27.5, 29]

    demand = 175

    # Run single experiment - exercise 3
    epec = EPEC(
        Pmin = Pmin, 
        Pmax = Pmax, 
        demand = demand, 
        cost = cost,
        exercise = "5"
    )

    epec.iterate_ownership_combinations(2)

    for run_id in epec.results:
        epec.plot_merit_order_curve(run_id = run_id)
        epec.plot_clearing_price_over_iterations(run_id = run_id)

    epec.plot_PoA()
