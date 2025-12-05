from Assignment_scripts.EPEC import EPEC

if __name__ == "__main__":    
    # Exercise 2
    # With the base setup
    Pmin = [ 0,  0,  0,  0]
    Pmax = [60, 80, 55, 70]

    demand = 190

    cost = [200, 300, 400, 500]

    # Run single experiment - exercise 3
    epec = EPEC(
                Pmin = Pmin, 
                Pmax = Pmax, 
                demand = demand, 
                cost = cost,
                exercise = "2"
    )

    # Plot the market clearing problem competitors
    epec.plot_merit_order_from_MPEC(strategic_index=0, cost_vector=cost)

    # Exercise 2_1
    #Overwrite the cost vector but keep the cost of the strategic player the same
    cost = [200, 600, 800, 1000]

    # Run single experiment - exercise 3
    epec = EPEC(
                Pmin = Pmin, 
                Pmax = Pmax, 
                demand = demand, 
                cost = cost,
                exercise = "2_1"
    )

    epec.plot_merit_order_from_MPEC(strategic_index=0, cost_vector=cost)

    # Change the demand to 150 
    demand = 150

    # True cost
    cost = [200, 300, 400, 500]

    # Run single experiment - exercise 3
    epec = EPEC(
                Pmin = Pmin, 
                Pmax = Pmax, 
                demand = demand, 
                cost = cost,
                exercise = "2"
    )

    epec.plot_merit_order_from_MPEC(strategic_index=0, cost_vector=cost)