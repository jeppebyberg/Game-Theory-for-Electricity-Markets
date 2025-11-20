import sys
import os
from pathlib import Path

# Add the script's directory to Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import the EPEC class
from EPEC_policy import EPEC

# Create outputs folder one level up from script directory
outputs_dir = script_dir.parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

# Parameters
alpha_min = 0
alpha_max = 1200

Pmin = [0, 0, 0, 0]
Pmax = [60, 80, 55, 50]

cost_fix = [80, 125, 130, 110]  # Initial cost vector (truthful costs)

# These are not used in this script but kept for EPEC class compatibility
cost_min = [2, 3, 4, 5]
cost_max = [60, 80, 55, 70]
segments = 2

max_iter = 100
demand = 150
convergence_tol = 0.003

# --- TOGGLE VERBOSE OUTPUT ---
VERBOSE = True  # Set to False to disable detailed iteration printouts

# Initialize EPEC
epec = EPEC(alpha_min, alpha_max, 
            Pmin, Pmax, demand, 
            cost_min, cost_max, 
            segments,
            cost_ownership=None, 
            max_iter = max_iter, 
            convergence_tol = convergence_tol)

# Run single best response with cost_fix as initial cost vector
print("\n" + "="*100)
print("STARTING BEST RESPONSE ALGORITHM")
print("="*100)
print(f"Initial truthful costs: {cost_fix}")
print(f"Demand: {demand} MW")
print(f"Max iterations: {max_iter}")
print(f"Verbose output: {'ENABLED' if VERBOSE else 'DISABLED'}")
print("="*100 + "\n")

# Use EPEC's built-in run_best_response method
(profit_history, alpha_history, dispatch_history, iterations, PoA, 
 dispatch_ED, clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, 
 clearing_price_history, weight_history, converged, not_converged, 
 player_order_history, round_by_round_bids, round_by_round_dispatch, 
 round_by_round_prices, round_by_round_profits) = epec.run_best_response(
    np.array(cost_fix), run_id=0
)

# --- VERBOSE OUTPUT: Print all rounds from all iterations ---
if VERBOSE:
    print("\n" + "="*100)
    print("DETAILED ITERATION BREAKDOWN")
    print("="*100 + "\n")
    
    num_generators = len(cost_fix)
    
    # Iterate through each iteration
    for iter_idx in range(len(alpha_history)):
        # Get the player order for this iteration
        players_this_iter = player_order_history[iter_idx]
        
        # For each player that acted in this iteration
        for round_idx, strategic_player in enumerate(players_this_iter):
            
            bids_after_round = round_by_round_bids[iter_idx][round_idx]
            dispatch_after_round = round_by_round_dispatch[iter_idx][round_idx]
            price_after_round = round_by_round_prices[iter_idx][round_idx]

            print("="*100)
            print(f"ITERATION {iter_idx + 1} | ROUND {round_idx + 1} | Strategic Player: {strategic_player}")
            print("="*100)
            print()
            print(f"{'Generator':<12} {'True Cost':<12} {'MC Dispatch':<15} {'MPEC Bid':<12} {'MPEC Dispatch':<15} {'Markup':<12}")
            print("-"*100)
            
            for j in range(num_generators):
                strategic_marker = "*" if j == strategic_player else " "
                true_cost = cost_fix[j]
                mc_dispatch = dispatch_ED[j]
                mpec_bid = bids_after_round[j]
                mpec_dispatch = dispatch_after_round[j]
                markup = mpec_bid - true_cost
                markup_sign = "+" if markup >= 0 else ""
                
                print(f"G{j} {strategic_marker:<10} ${true_cost:<11.2f} {mc_dispatch:<15.2f} ${mpec_bid:<11.2f} {mpec_dispatch:<15.2f} ${markup_sign}{markup:.2f}")
            
            print("-"*100)
            print(f"\nMarket Clearing Price:         MC: ${clearing_price_ED:.2f}/MWh  |  MPEC: ${price_after_round:.2f}/MWh")
            
            if iter_idx > 0 or round_idx > 0:
                price_increase = clearing_price_history[iter_idx] - clearing_price_ED
                pct_increase = (price_increase / clearing_price_ED * 100) if clearing_price_ED > 0 else 0
                print(f"Price Increase:                ${price_increase:.2f}/MWh ({pct_increase:+.2f}%)")
            
            # Calculate profit for strategic player
            strategic_profit = (price_after_round * dispatch_after_round[strategic_player] 
                              - cost_fix[strategic_player] * dispatch_after_round[strategic_player])
            print(f"Strategic Player Profit:       ${strategic_profit:.2f}")

            # Total generation cost comparison
            mc_total_cost = sum(cost_fix[i] * dispatch_ED[i] for i in range(num_generators))
            mpec_total_cost = sum(cost_fix[i] * dispatch_after_round[i] for i in range(num_generators))
            print(f"Total Generation Cost:         MC: ${mc_total_cost:.2f}  |  MPEC: ${mpec_total_cost:.2f}")
            print()
    
    print("="*100)
    print("END OF DETAILED BREAKDOWN")
    print("="*100 + "\n")

# Extract data for plotting
alphas = np.array(alpha_history)
dispatches = np.array(dispatch_history)
num_generators = len(cost_fix)

print("\n" + "="*100)
print("GENERATING PLOTS")
print("="*100 + "\n")

# --- PLOT 1: Alpha evolution for each player ---
plt.figure(figsize=(10, 6))
for i in range(num_generators):
    plt.plot(alphas[:, i], marker='o', label=f'G{i}', linewidth=2)

# Add horizontal lines for truthful costs and final bids (lighter, no labels)
for i in range(num_generators):
    plt.axhline(y=cost_fix[i], color=f'C{i}', linestyle='--', alpha=0.3)
    plt.axhline(y=final_bid[i], color=f'C{i}', linestyle=':', alpha=0.3)

# Add generic legend entries for line styles
legend_lines = [Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='Truthful cost'),
                Line2D([0], [0], color='gray', linestyle=':', alpha=0.5, label='Final bid')]

# Get existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
# Combine with custom handles
all_handles = handles + legend_lines
all_labels = labels + ['Truthful cost', 'Final bid']

plt.xlabel('Iteration k')
plt.ylabel('Bid α_i(k)')
plt.title('Bid Evolution Over Iterations')
plt.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(outputs_dir / 'alpha_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PLOT 2: Dispatch comparison (bar chart) ---
x = np.arange(num_generators)
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, dispatch_ED, width, label='Central Dispatch (ED)', color='green', alpha=0.7)
bars2 = ax.bar(x + width/2, final_dispatch, width, label='Equilibrium Dispatch (SP)', color='purple', alpha=0.7)

ax.set_xlabel('Generator')
ax.set_ylabel('Dispatch (MW)')
ax.set_title('Dispatch Comparison: Central vs Equilibrium')
ax.set_xticks(x)
ax.set_xticklabels([f'G{i}' for i in range(num_generators)])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig(outputs_dir / 'dispatch_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PLOT 3: Dispatch evolution over iterations ---
plt.figure(figsize=(10, 6))
for i in range(num_generators):
    plt.plot(dispatches[:, i], marker='o', label=f'G{i}', linewidth=2)

# Add horizontal lines for central and final dispatch (lighter, no labels)
for i in range(num_generators):
    plt.axhline(y=dispatch_ED[i], color=f'C{i}', linestyle='--', alpha=0.3)
    plt.axhline(y=final_dispatch[i], color=f'C{i}', linestyle=':', alpha=0.3)

# Add generic legend entries for line styles
legend_lines = [Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='Central dispatch'),
                Line2D([0], [0], color='gray', linestyle=':', alpha=0.5, label='Final dispatch')]

# Get existing handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
# Combine with custom handles
all_handles = handles + legend_lines
all_labels = labels + ['Central dispatch', 'Final dispatch']

plt.xlabel('Iteration k')
plt.ylabel('Dispatch p_i(k) [MW]')
plt.title('Dispatch Evolution Over Iterations')
plt.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(outputs_dir / 'dispatch_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PLOT 4: Market Clearing Price per Round across Iterations ---
plt.figure(figsize=(12, 6))

# Organize prices by round (one line per strategic player position)
for round_idx in range(num_generators):
    prices_for_round = []
    iterations_list = []
    
    for iter_idx in range(len(round_by_round_prices)):
        if round_idx < len(round_by_round_prices[iter_idx]):
            prices_for_round.append(round_by_round_prices[iter_idx][round_idx])
            iterations_list.append(iter_idx + 1)
    
    # Get which player acted in this round for labeling
    player_label = f'Round {round_idx + 1}'
    plt.plot(iterations_list, prices_for_round, marker='o', label=player_label, linewidth=2)

# Add horizontal line for central clearing price
plt.axhline(y=clearing_price_ED, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Central Price')

plt.xlabel('Iteration')
plt.ylabel('Market Clearing Price ($/MWh)')
plt.title('Market Clearing Price After Each Strategic Round')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outputs_dir / 'price_by_round.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PLOT 5: Strategic Player Profit After Their Round across Iterations ---
plt.figure(figsize=(12, 6))

# Organize profits by which player was strategic (one line per player)
for player_idx in range(num_generators):
    profits_for_player = []
    iterations_list = []
    
    for iter_idx in range(len(player_order_history)):
        players_this_iter = player_order_history[iter_idx]
        # Find which round this player acted in
        if player_idx in players_this_iter:
            round_idx = players_this_iter.index(player_idx)
            profit = round_by_round_profits[iter_idx][round_idx]
            profits_for_player.append(profit)
            iterations_list.append(iter_idx + 1)
    
    plt.plot(iterations_list, profits_for_player, marker='o', label=f'G{player_idx}', linewidth=2)

plt.xlabel('Iteration')
plt.ylabel('Profit After Strategic Round ($)')
plt.title('Strategic Player Profit After Their Round')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outputs_dir / 'strategic_profit_by_round.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PLOT 6: Final Profits for Each Generator across Iterations ---
plt.figure(figsize=(12, 6))

# profit_history contains end-of-iteration profits for all generators
profits_array = np.array(profit_history)

for i in range(num_generators):
    plt.plot(range(1, len(profit_history) + 1), profits_array[:, i], 
             marker='o', label=f'G{i}', linewidth=2)

plt.xlabel('Iteration')
plt.ylabel('Final Profit (End of Iteration) ($)')
plt.title('Generator Profits at End of Each Iteration')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outputs_dir / 'final_profits_per_iteration.png', dpi=300, bbox_inches='tight')
plt.show()

# --- COMPUTE INEFFICIENCY ---
# Central dispatch cost (numerator in PoA calculation)
central_cost = sum(cost_fix[i] * dispatch_ED[i] for i in range(num_generators))

# Equilibrium cost
equilibrium_cost = sum(cost_fix[i] * final_dispatch[i] for i in range(num_generators))

# Inefficiency = PoA
inefficiency = equilibrium_cost / central_cost

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Convergence status: {'Converged' if converged else 'Not converged'}")
print(f"Total iterations: {iterations}")
print()
print("Truthful costs:", cost_fix)
print("Final bids:", [f"{bid:.2f}" for bid in final_bid])
print()
print("Central Dispatch (ED):", [f"{d:.2f}" for d in dispatch_ED])
print("Equilibrium Dispatch (SP):", [f"{d:.2f}" for d in final_dispatch])
print()
print(f"Central clearing price: {clearing_price_ED:.2f}")
print(f"Equilibrium clearing price: {clearing_price_SP:.2f}")
print()
print("INEFFICIENCY CALCULATION:")
print(f"  Numerator (Equilibrium cost):   {equilibrium_cost:.2f}")
print(f"  Denominator (Central cost):     {central_cost:.2f}")
print(f"  Inefficiency (PoA):             {inefficiency:.4f}")
print(f"  PoA (from EPEC):                {PoA:.4f}")
print("=" * 60)