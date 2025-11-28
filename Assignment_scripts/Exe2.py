from Assignment_scripts.EPEC import EPEC

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

    # Change the demand to 190 and plot again    
    demand = 190

    # Run single experiment - exercise 3
    epec = EPEC(
                Pmin = Pmin, 
                Pmax = Pmax, 
                demand = demand, 
                cost = cost,
                exercise = "2"
    )

    epec.plot_merit_order_from_MPEC(strategic_index=0, cost_vector=cost)