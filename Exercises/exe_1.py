from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, Constraint, minimize, value
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import numpy as np


#Number of generators
n = 30

np.random.seed(42)  # For reproducibility

cost_array = np.random.randint(1, 51, size=n)
prod_array = np.random.randint(10, 20, size=n)

demand = 100

m = ConcreteModel()
m.x = Var(range(n), domain=NonNegativeReals)
m.obj = Objective(expr= sum(cost_array[i] * m.x[i] for i in range(n)), sense=minimize)
m.c = Constraint(expr= sum(m.x[i] for i in range(n)) == demand)

for i in range(n):
    m.add_component(f"c{i+2}", Constraint(expr=m.x[i] <= prod_array[i]))

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
for i in range(n):
    if value(m.x[i]) > 1e-6:
        print(f"Generator {i+1} Cost: {cost_array[i]}   Capacity: {prod_array[i]}")
        print(f"Generator {i+1} Production: {value(m.x[i]):.4f}")
        print()

# Plot the merit order curve
sorted_indices = np.argsort(cost_array)
sorted_costs = cost_array[sorted_indices]
sorted_prods = prod_array[sorted_indices]
cum_prods = np.cumsum(sorted_prods)

plt.figure(figsize=(8, 5))
plt.step(np.insert(cum_prods, 0, 0), np.insert(sorted_costs, 0, sorted_costs[0]), where='pre', label='Merit Order Curve')
plt.axvline(demand, color='r', linestyle='--', label='Demand')
plt.xlabel('Cumulative Production (MW)')
plt.ylabel('Cost per Unit')
plt.title('Merit Order Curve')
plt.xlim(0, sum(prod_array) + 5)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()