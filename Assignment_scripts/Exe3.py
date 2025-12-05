from Assignment_scripts.EPEC import EPEC
import os
import numpy as np

def compute_metrics(epec: EPEC, run_id: int):
    """
    Computes and prints PoA (productive inefficiency) and 
    CCI (consumer cost inflation) for a given run_id.
    """

    # Extract arrays
    cost = np.array(epec.cost)
    dispatch_EQ = np.array(epec.results[run_id]["final_dispatch"])
    dispatch_ED = np.array(epec.results[run_id]["dispatch_ED"])

    # ----- Price of Anarchy -----
    num_PoA = (cost * dispatch_EQ).sum()

    den_PoA = (cost * dispatch_ED).sum()
    PoA = num_PoA / den_PoA

    print("\n=== PRICE OF ANARCHY (PoA) ===")
    print(f"PoA: {PoA:.6f}")
    print(f"Numerator (SC_eq):   {num_PoA}")
    print(f"Denominator (SC_opt): {den_PoA}")

    # ----- Consumer Cost Inflation (CCI) -----
    clearing_price_EQ = epec.results[run_id]["final_clearing_price"]
    clearing_price_ED = epec.results[run_id]["clearing_price_ED"]

    num_CCI = clearing_price_EQ * epec.demand
    den_CCI = clearing_price_ED * epec.demand
    CCI = num_CCI / den_CCI

    print("\n=== CONSUMER COST INFLATION (CCI) ===")
    print(f"CCI: {CCI:.6f}")
    print(f"Numerator (CE_eq):   {num_CCI}")
    print(f"Denominator (CE_opt): {den_CCI}")

    return PoA, CCI

if __name__ == "__main__":
    
    # Exercise 3
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
    epec.plot_alpha_over_iterations(run_id = 0)

    compute_metrics(epec, run_id=0)

    df = epec.build_iteration_table(run_id=0)

    os.makedirs('Assignment_scripts/outputs', exist_ok=True)
    df.to_csv('Assignment_scripts/outputs/exercise_3_iteration_table.csv', index=False)


