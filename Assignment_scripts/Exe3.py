from EPEC import EPEC

if __name__ == "__main__":
    
    # Exercise 3 and 4 setup
    Pmin = [ 0,  0,  0,  0]
    Pmax = [55, 60, 65, 70]

    demand = 160

    cost = [80, 100, 120, 140]

    # Run single experiment - exercise 3
    epec = EPEC(
        Pmin = Pmin, 
        Pmax = Pmax, 
        demand = demand, 
        cost = cost,
        exercise = "3"
    )

    epec.run_single_experiment()
    epec.plot_merit_order_curve(run_id = 0)

