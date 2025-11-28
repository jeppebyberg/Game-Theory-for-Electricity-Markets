from EPEC import EPEC
import numpy as np 
import matplotlib.pyplot as plt
from Exe4_utils import generate_scaled_setup

def run_multiple_player_setups(max_players: int, P_max_ref: list, cost_ref: list, demand_ref: float):   

    epec_results = {}

    players_list = range(4, max_players + 1)
    convergence_rate = []
    worst_poa_list   = []

    for n_players in players_list:
        print(f"\n--- Running EPEC for {n_players} players ---")
        Pmin, Pmax, cost, demand = generate_scaled_setup(n_players=n_players, P_max_ref=P_max_ref, demand_ref=demand_ref, cost_ref=cost_ref)
        print("Pmin:", Pmin)
        print("Pmax:", Pmax)
        print("Cost:", cost)
        print("Demand:", demand)

        epec = EPEC(Pmin = Pmin, 
                    Pmax = Pmax, 
                    demand = demand, 
                    cost = cost, 
                    segments = segments, 
                    )
        
        share_converged, worst_poa = epec.iterate_cost_combinations()
        convergence_rate.append(share_converged)
        worst_poa_list.append(worst_poa)
        epec_results[n_players] = epec
        if n_players == 4:
            # Plot detailed results for 4 players only
            for run_id in epec.results:
                epec.plot_merit_order_curve(run_id = run_id)
                epec.plot_clearing_price_over_iterations(run_id = run_id)
            epec.plot_PoA()
    
    # # --- Plot convergence rate vs number of players ---
    plt.figure(figsize=(8, 5))
    plt.plot(players_list, convergence_rate, marker='o')
    plt.xlabel('Number of Players')
    plt.ylabel('Convergence Rate (%)')
    # plt.title('EPEC Convergence Rate vs Number of Players')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # # --- Plot worst PoA vs number of players ---
    plt.figure(figsize=(8, 5))
    plt.plot(players_list, worst_poa_list, marker='o', color='orange')
    plt.xlabel('Number of Players')
    plt.ylabel('Worst Price of Anarchy (PoA)')
    # plt.title('Worst PoA vs Number of Players')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return epec_results

if __name__ == "__main__":
    
    # Exercise 3 and 4 setup
    Pmin = [ 0,  0,  0,  0]
    Pmax = [55, 60, 65, 70]

    demand = 160

    cost = [80, 100, 120, 140]

    segments = 2

    # Run multiple player setups
    max_players = 10
    epec_results = run_multiple_player_setups(max_players=max_players, P_max_ref=Pmax, cost_ref=cost, demand_ref=demand)



