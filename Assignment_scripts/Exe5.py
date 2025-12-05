# from Assignment_scripts.EPEC import EPEC
from EPEC import EPEC
import pandas as pd

def extract_ownership_profits(epec: EPEC):
    rows = []
    for run_id, res in epec.results.items():

        owners = res["owner_indexes"]
        profit_history = res["profit_history"]
        final_profits = profit_history[-1]   # actor-level profits

        owner_profit = final_profits[0]      # owner is always actor 0
        poa = res["PoA"]
        cci = res["CCI"]
        converged = res["converged"]
        RSI = (30 * len(epec.Pmax) - sum(epec.Pmax[i] for i in owners)) / epec.demand

        rows.append({
            "run_id": run_id,
            "owners": owners,
            "owner_profit": owner_profit,
            "PoA": poa,
            "CCI": cci,
            "converged": converged,
            "RSI": RSI
        })

    df = pd.DataFrame(rows)

    df = df.sort_values("owner_profit", ascending=False)
    return df

def df_to_latex_ownership(df, caption="Ownership Results for Exercise 5", label="tab:ex5_ownership"):
    # Format owners as a string, e.g. (0,3)
    df_fmt = df.copy()

    if df_fmt["owners"].apply(len).max() == 1:
        df_fmt["owners"] = df_fmt["owners"].apply(lambda x: f"({x[0]})")

    if df_fmt["owners"].apply(len).max() == 2:
        df_fmt["owners"] = df_fmt["owners"].apply(lambda x: f"({x[0]}, {x[1]})")
    
    if df_fmt["owners"].apply(len).max() == 3:
        df_fmt["owners"] = df_fmt["owners"].apply(lambda x: f"({x[0]}, {x[1]}, {x[2]})")
    
    # Choose columns to include
    cols = ["owners", "owner_profit", "PoA", "CCI", "RSI", "converged"]

    latex_table = df_fmt[cols].to_latex(
        index=False,
        escape=False,
        float_format="%.3f",
        caption=caption,
        label=label
    )

    return latex_table

if __name__ == "__main__":
    
    # # Exercise 5_1
    Pmin = [ 0,  0,  0,  0,  0,  0,  0]
    Pmax = [30, 30, 30, 30, 30, 30, 30]

    cost = [20, 30, 50, 480, 500, 550, 580]

    demand = 85

    epec_benchmark_low = EPEC(
        Pmin = Pmin,
        Pmax = Pmax,
        demand = demand,
        cost = cost,
        exercise = "5_benchmark_low"
    )

    epec_low = EPEC(
        Pmin = Pmin, 
        Pmax = Pmax, 
        demand = demand, 
        cost = cost,
        exercise = "5_low_demand"
    )

    epec_benchmark_low.iterate_ownership_combinations(1)
    epec_low.iterate_ownership_combinations(2)

    for run_id in epec_benchmark_low.results:
        epec_benchmark_low.plot_merit_order_curve(run_id = run_id)
    
    for run_id in epec_low.results:
        epec_low.plot_merit_order_curve(run_id = run_id)

    print(df_to_latex_ownership(extract_ownership_profits(epec_low)))

    # High demand setup
    demand = 175

    epec_benchmark_high = EPEC(
        Pmin = Pmin, 
        Pmax = Pmax, 
        demand = demand, 
        cost = cost,
        exercise = "5_benchmark_high"
    )  

    epec_high = EPEC(
        Pmin = Pmin, 
        Pmax = Pmax, 
        demand = demand, 
        cost = cost,
        exercise = "5_high_demand"
    )

    epec_benchmark_high.iterate_ownership_combinations(1)
    epec_high.iterate_ownership_combinations(2)

    for run_id in epec_benchmark_high.results:
        epec_benchmark_high.plot_merit_order_curve(run_id = run_id)

    for run_id in epec_high.results:
        epec_high.plot_merit_order_curve(run_id = run_id)

    print(df_to_latex_ownership(extract_ownership_profits(epec_high)))
