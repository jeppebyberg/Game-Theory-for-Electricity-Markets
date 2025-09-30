from pyomo.environ import *
import matplotlib.pyplot as plt
import numpy as np

#Number of generators
n_gen = 3

#Number of consumers
n_cons = 3

np.random.seed(42)  # For reproducibility

cost_array = np.random.randint(1, 51, size=n_gen)
prod_array = np.random.randint(10, 20, size=n_gen)

bid_array = np.random.randint(1, 30, size=n_cons)
demand_array = np.random.randint(10, 20, size=n_cons)

m = ConcreteModel()
m.x_gen = Var(range(n_gen), domain=NonNegativeReals)
m.x_cons = Var(range(n_cons), domain=NonNegativeReals)

m.obj = Objective(expr= sum(bid_array[i] * m.x_cons[i] for i in range(n_cons)) - sum(cost_array[i] * m.x_gen[i] for i in range(n_gen)), sense=maximize)

# Balance equality (this is the one whose dual you want)
m.balance = Constraint(expr=sum(m.x_cons[i] for i in range(n_cons)) ==
                            sum(m.x_gen[i] for i in range(n_gen)))

for i in range(n_gen):
    m.add_component(f"c{i+2}", Constraint(expr=m.x_gen[i] <= prod_array[i]))

for i in range(n_cons):
    m.add_component(f"c{i+2+n_gen}", Constraint(expr=m.x_cons[i] <= demand_array[i]))

# --- ask for duals & reduced costs ---
m.dual = Suffix(direction=Suffix.IMPORT)
m.rc   = Suffix(direction=Suffix.IMPORT)

# Pick a solver
solver_name = None
for cand in ("appsi_highs", "highs", "glpk"):
    try:
        sf = SolverFactory(cand)
        if sf.available():
            solver_name = cand
            solver = sf
            break
    except Exception:
        pass

res = solver.solve(m, tee=False)

print(f"Termination: {res.solver.termination_condition}")

print(f"Objective value = {value(m.obj):.4f}")
print(f"Production levels:")
for i in range(n_gen):
    if value(m.x_gen[i]) > 1e-6:
        print(f"Generator {i+1} Cost: {cost_array[i]}   Capacity: {prod_array[i]}")
        print(f"Generator {i+1} Production: {value(m.x_gen[i]):.4f}")
        print()

for i in range(n_cons):
    if value(m.x_cons[i]) > 1e-6:
        print(f"Consumer {i+1} Bid: {bid_array[i]}   Demand: {demand_array[i]}")
        print(f"Consumer {i+1} Consumption: {value(m.x_cons[i]):.4f}")
        print()

# Shadow price (Lagrange multiplier) of the equality:
print(f"Market-clearing price = {m.dual[m.balance]:.6f}")

# Sort generators by cost (ascending)
gen_sorted_idx = np.argsort(cost_array)
gen_sorted_costs = cost_array[gen_sorted_idx]
gen_sorted_caps = prod_array[gen_sorted_idx]

# Merit order curve for generators (supply)
gen_cum_cap = np.cumsum(gen_sorted_caps)
gen_curve_x = np.insert(gen_cum_cap, 0, 0)
gen_curve_y = np.repeat(gen_sorted_costs, 2)
gen_curve_y = gen_curve_y[:len(gen_curve_x)]  # Ensure y matches x length

# Sort consumers by bid (descending)
cons_sorted_idx = np.argsort(-bid_array)
cons_sorted_bids = bid_array[cons_sorted_idx]
cons_sorted_demands = demand_array[cons_sorted_idx]

# Merit order curve for consumers (demand)
cons_cum_dem = np.cumsum(cons_sorted_demands)
cons_curve_x = np.insert(cons_cum_dem, 0, 0)
cons_curve_y = np.repeat(cons_sorted_bids, 2)
cons_curve_y = cons_curve_y[:len(cons_curve_x)]  # Ensure y matches x length

plt.figure(figsize=(10, 6))
plt.step(gen_curve_x, gen_curve_y, where='pre', label='Supply (Generators)', color='blue')
plt.step(cons_curve_x, cons_curve_y, where='pre', label='Demand (Consumers)', color='red')

plt.xlabel('Cumulative Quantity')
plt.ylabel('Price')
plt.title('Merit Order Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()