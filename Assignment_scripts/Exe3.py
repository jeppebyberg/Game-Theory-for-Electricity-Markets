from Assignment_scripts.EPEC import EPEC
import os

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
    epec.plot_dispatch_comparison(run_id = 0)
    epec.plot_clearing_price_over_iterations(run_id = 0)
    epec.plot_profits(run_id = 0)

    df = epec.build_iteration_table(run_id=0)

    os.makedirs('Assignment_scripts/outputs', exist_ok=True)
    df.to_csv('Assignment_scripts/outputs/exercise_3_iteration_table.csv', index=False)


