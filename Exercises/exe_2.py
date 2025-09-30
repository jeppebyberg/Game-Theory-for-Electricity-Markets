from pyomo.environ import ConcreteModel, Var, NonNegativeReals, Objective, Constraint, maximize, value
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

# Create a variable for each generator's production
m.y = Var(range(n), domain=NonNegativeReals)

# Create a variable for the equality constraint
m.y_eq = Var()

# Objective function: maximize total profit (or minimize cost, depending on your formulation)
m.obj = Objective(expr= m.y_eq * demand - sum(prod_array[i] * m.y[i] for i in range(n)), sense=maximize)

for i in range(n):
    m.add_component(f"c1_{i}", Constraint(expr= m.y_eq - m.y[i] <= cost_array[i] ))

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
