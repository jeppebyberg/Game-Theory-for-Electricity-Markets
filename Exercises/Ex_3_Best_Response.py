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

# Initialize EPEC
epec = EPEC(alpha_min, alpha_max, 
            Pmin, Pmax, demand, 
            cost_min, cost_max, 
            segments, 
            max_iter, convergence_tol)

# Run single best response with cost_fix as initial cost vector
print("\n" + "="*100)
print("STARTING BEST RESPONSE ALGORITHM")
print("="*100)
print(f"Initial truthful costs: {cost_fix}")
print(f"Demand: {demand} MW")
print(f"Max iterations: {max_iter}")
print("="*100 + "\n")

# Store original run_best_response method
original_run = epec.run_best_response

def run_best_response_verbose(init_cost_vector, run_id):
    # Reset histories for this run
    profit_history = []
    alpha_history = []
    dispatch_history = []
    convergence_check = []
    clearing_price_history = []
    weight_history = []

    dispatch_ED, clearing_price_ED, minimum_cost_ED = epec.economic_dispatch(init_cost_vector)

    iter = 0
    cost_vector = init_cost_vector.copy()

    # Best-response iterations
    while iter <= epec.max_iter:
        profit_history.append([None] * epec.num_generators)
        alpha_history.append([None] * epec.num_generators)
        dispatch_history.append([None] * epec.num_generators)
        convergence_check.append([False] * epec.num_generators)
        weight_history.append([None] * epec.num_generators)

        for i in range(epec.num_generators):
            epec._build_model(i, cost_vector, init_cost_vector)
            epec.solve()
            cost_vector[i] = epec.model.alpha.value
            profit_history[iter][i] = -epec.model.objective()
            alpha_history[iter][i] = epec.model.alpha.value
            dispatch_history[iter] = [epec.model.P_G[ii].value for ii in epec.model.n_gen]
            clearing_price_SP = epec.model.lambda_dual.value
            weight_history[iter][i] = [epec.model.omega[ii].value for ii in epec.model.n_gen - epec.model.strategic_index]

            # Print iteration details
            print("="*100)
            print(f"ITERATION {iter + 1} | ROUND {i + 1} | Strategic Actor: {i}")
            print("="*100)
            print()
            print(f"{'Generator':<12} {'True Cost':<12} {'MC Dispatch':<15} {'MPEC Bid':<12} {'MPEC Dispatch':<15} {'Markup':<12}")
            print("-"*100)
            
            for j in range(epec.num_generators):
                strategic_marker = "*" if j == i else " "
                true_cost = init_cost_vector[j]
                mc_dispatch = dispatch_ED[j]
                mpec_bid = cost_vector[j]
                mpec_dispatch = dispatch_history[iter][j]
                markup = mpec_bid - true_cost
                markup_sign = "+" if markup >= 0 else ""
                
                print(f"G{j} {strategic_marker:<10} ${true_cost:<11.2f} {mc_dispatch:<15.2f} ${mpec_bid:<11.2f} {mpec_dispatch:<15.2f} ${markup_sign}{markup:.2f}")
            
            print("-"*100)
            print(f"\nClearing Price: ${clearing_price_SP:.2f}")
            print()

            if iter > 0:
                if profit_history[iter][i] >= (1 - epec.convergence_tol) * profit_history[iter - 1][i] and profit_history[iter][i] <= (1 + epec.convergence_tol) * profit_history[iter - 1][i]:
                    convergence_check[iter][i] = True

        clearing_price_history.append(clearing_price_SP)

        if all(convergence_check[iter]):
            print("\n" + "="*100)
            print(f"✅ CONVERGED AFTER {iter} ITERATIONS")
            print("="*100 + "\n")
            PoA = clearing_price_SP * epec.demand / minimum_cost_ED
            final_bid = cost_vector.copy()
            final_dispatch = dispatch_history[iter]
            break
        if iter == epec.max_iter:
            print(f"\n⚠️  Reached max iterations - {epec.max_iter}.\n")
            PoA = clearing_price_SP * epec.demand / minimum_cost_ED
            
            # Compute mean of last 10 dispatches 
            dispatch_array = np.array(dispatch_history[-10:])
            mean_dispatch = np.mean(dispatch_array, axis=0)
            final_dispatch = mean_dispatch

            bid_array = np.array(alpha_history[-10:])
            mean_bid = np.mean(bid_array, axis=0)
            final_bid = mean_bid
            break
        iter += 1
    
    return profit_history, alpha_history, dispatch_history, iter, PoA, dispatch_ED, clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, clearing_price_history, weight_history

# Replace method temporarily
profits, alphas, dispatches, iterations, PoA, dispatch_ED, clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, clearing_price_history, weight_history = run_best_response_verbose(np.array(cost_fix), run_id=0)

# --- PLOT 1: Alpha evolution for each player ---
alphas_array = np.array(alphas)
num_generators = len(cost_fix)

plt.figure(figsize=(10, 6))
for i in range(num_generators):
    plt.plot(alphas_array[:, i], marker='o', label=f'G{i}', linewidth=2)

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
dispatches_array = np.array(dispatches)

plt.figure(figsize=(10, 6))
for i in range(num_generators):
    plt.plot(dispatches_array[:, i], marker='o', label=f'G{i}', linewidth=2)

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
print(f"Converged after {iterations} iterations")
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
print("=" * 60)