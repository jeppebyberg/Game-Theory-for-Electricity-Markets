from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

#Number of nodes
n_nodes = 3

np.random.seed(42)  # For reproducibility

B_matrix = [[10, 10, 10],
            [10, 10, 10],
            [10, 10, 10]]

for i in range(n_nodes):
    for j in range(n_nodes):
        B_matrix[i][j] * 10


cost_array = np.random.randint(1, 51, size=n_nodes)
prod_array = np.random.randint(10, 20, size=n_nodes)

bid_array = np.random.randint(1, 30, size=n_nodes)
demand_array = np.random.randint(10, 20, size=n_nodes)

# cost_array = [5, 10, 15]
# prod_array = [20, 10, 5]

# bid_array = [10, 15, 20]
# demand_array = [15, 10, 5]

m = ConcreteModel()

m.voltage_angle = Var(range(n_nodes), domain=Reals)
m.x_gen = Var(range(n_nodes), domain=NonNegativeReals)
m.x_cons = Var(range(n_nodes), domain=NonNegativeReals)

m.obj = Objective(expr= sum(bid_array[i] * m.x_cons[i] for i in range(n_nodes)) 
                      - sum(cost_array[i] * m.x_gen[i] for i in range(n_nodes)), sense=maximize)

for i in range(n_nodes):
    m.add_component(f"c{i+2}", Constraint(expr=m.x_gen[i] <= prod_array[i]))
    m.add_component(f"c{i+2+n_nodes}", Constraint(expr=m.x_cons[i] <= demand_array[i]))
    m.add_component(f"voltage_angles_(1)_{i+1}", Constraint(expr=m.voltage_angle[i] >= -np.pi))
    m.add_component(f"voltage_angles_(2)_{i+1}", Constraint(expr=m.voltage_angle[i] <= np.pi))
    m.add_component(f"balance_eq_{i+1}", Constraint(expr=m.x_cons[i] - m.x_gen[i] + sum(B_matrix[i][j] * (m.voltage_angle[i] - m.voltage_angle[j]) for j in range(n_nodes)) == 0))

m.reference = Constraint(expr=m.voltage_angle[0] == 0)  # Reference bus

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
# Shadow price (Lagrange multiplier) of the equality:
for i in range(n_nodes):
    con = getattr(m, f"balance_eq_{i+1}")   # grab the Constraint object
    lam = m.dual[con]                       # shadow price / LMP at node i
    print(f"Node {i} price = {lam:.6f}")

    print(f"Node {i} voltage angle = {value(m.voltage_angle[i]):.6f}")
    print(f"Node {i} generation = {value(m.x_gen[i]):.6f}")
    print(f"Node {i} consumption = {value(m.x_cons[i]):.6f}")

