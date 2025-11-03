import sys
import os
from pathlib import Path

# Add the script's directory to Python path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

# Import the EPEC class
from EPEC_policy import EPEC

# Create outputs folder one level up from script directory
outputs_dir = script_dir.parent / "outputs"
outputs_dir.mkdir(exist_ok=True)

# Parameters
alpha_min = -250
alpha_max = 1000

Pmin = [0, 0, 0]
Pmax = [55, 90, 95]

cost_fix = [80, 250, 350]  # Initial cost vector (truthful costs)

# These are not used in this script but kept for EPEC class compatibility
cost_min = [2, 3, 4]
cost_max = [60, 80, 55]
segments = 2

max_iter = 100
demand = 150
convergence_tol = 0.01

# Initialize EPEC
epec = EPEC(alpha_min, alpha_max, 
            Pmin, Pmax, demand, 
            cost_min, cost_max, 
            segments, 
            max_iter, convergence_tol)

# Run single best response with cost_fix as initial cost vector
profits, alphas, dispatches, iterations, PoA, dispatch_ED, clearing_price_ED, final_dispatch, final_bid, clearing_price_SP, clearing_price_history, weight_history = epec.run_best_response(np.array(cost_fix), run_id=0)

# --- PLOT 1: Alpha evolution for each player ---
alphas_array = np.array(alphas)
num_generators = len(cost_fix)

plt.figure(figsize=(10, 6))
for i in range(num_generators):
    plt.plot(alphas_array[:, i], marker='o', label=f'Player {i}')
    plt.axhline(y=cost_fix[i], color=f'C{i}', linestyle='--', alpha=0.5, label=f'Truthful cost G{i}')
    plt.axhline(y=final_bid[i], color=f'C{i}', linestyle=':', alpha=0.7, label=f'Final bid G{i}')

plt.xlabel('Iteration k')
plt.ylabel('Bid α_i(k)')
plt.title('Bid Evolution Over Iterations')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(outputs_dir /'alpha_evolution.png', dpi=300, bbox_inches='tight')
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
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig(outputs_dir /'dispatch_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# --- PLOT 3: Dispatch evolution over iterations ---
dispatches_array = np.array(dispatches)

plt.figure(figsize=(10, 6))
for i in range(num_generators):
    plt.plot(dispatches_array[:, i], marker='o', label=f'Generator {i}', linewidth=2)
    plt.axhline(y=dispatch_ED[i], color=f'C{i}', linestyle='--', alpha=0.5, label=f'Central dispatch G{i}')
    plt.axhline(y=final_dispatch[i], color=f'C{i}', linestyle=':', alpha=0.7, label=f'Final dispatch G{i}')

plt.xlabel('Iteration k')
plt.ylabel('Dispatch p_i(k) [MW]')
plt.title('Dispatch Evolution Over Iterations')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.grid(True)
plt.tight_layout()
plt.savefig(outputs_dir / 'dispatch_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

# --- COMPUTE INEFFICIENCY ---
# Central dispatch cost (numerator in PoA calculation)
central_cost = sum(cost_fix[i] * dispatch_ED[i] for i in range(num_generators))

# Equilibrium cost
equilibrium_cost = clearing_price_SP * demand

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