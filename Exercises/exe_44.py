from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt

class MarketClearingLP:
    def __init__(self, A, prod_costs, cons_bids, demand=None):
        """
        Initialize the market clearing problem.

        Parameters
        ----------
        A : 2D numpy array
            Inequality matrix (e.g. generator constraints Ax <= b)
        prod_costs : 1D numpy array
            Cost vector for producers
        cons_bids : 1D numpy array
            Bid vector for consumers
        demand : float or None
            If None, total demand = sum(cons_bids > 0)
        """
        self.A = np.array(A)
        self.prod_costs = np.array(prod_costs)
        self.cons_bids = np.array(cons_bids)
        self.n_prod = len(prod_costs)
        self.n_cons = len(cons_bids)
        
        # Default demand: sum of consumer bids (could be flexible)
        self.demand = demand if demand is not None else self.cons_bids.sum()

        self.model = ConcreteModel()

    def build_primal(self):
        m = self.model
        # Variables
        m.g = Var(range(self.n_prod), domain=NonNegativeReals)  # production
        m.q = Var(range(self.n_cons), domain=NonNegativeReals)  # consumption
        
        # Objective: maximize consumer surplus - production cost
        m.obj = Objective(
            expr=  sum(self.cons_bids[j] * m.q[j] for j in range(self.n_cons)) 
                 - sum(self.prod_costs[i] * m.g[i] for i in range(self.n_prod)),
            sense=maximize
        )

        # Balance constraint: total generation = total consumption
        def balance_rule(m):
            return sum(m.g[i] for i in range(self.n_prod)) == sum(m.q[j] for j in range(self.n_cons))
        m.balance = Constraint(rule=balance_rule)

        # Inequality constraints: A g <= b
        if self.A.size > 0:
            b = np.ones(self.A.shape[0]) * 100  # default RHS (can be parameterized)
            def ineq_rule(m, r):
                return sum(self.A[r, i] * m.g[i] for i in range(self.n_prod)) <= b[r]
            m.ineq = Constraint(range(self.A.shape[0]), rule=ineq_rule)

    def solve(self, solver="glpk"):
        self.build_primal()
        solver = SolverFactory("gurobi")
        result = solver.solve(self.model, tee=True)  # tee=True → show solver log
        return result

    def get_solution(self):
        m = self.model
        g = np.array([m.g[i].value for i in range(self.n_prod)])
        q = np.array([m.q[j].value for j in range(self.n_cons)])
        obj = m.obj()
        return {"generation": g, "consumption": q, "objective": obj}
    
if __name__ == "__main__":
    # 2 producers, 3 consumers
    A = np.array([[1, 0],
                  [0, 1]])  # simple capacity constraints

    prod_costs = np.array([10, 20])   # cost per unit
    cons_bids = np.array([25, 18, 15])  # willingness to pay

    market = MarketClearingLP(A, prod_costs, cons_bids)
    market.solve()
    sol = market.get_solution()

    print("Optimal generation:", sol["generation"])
    print("Optimal consumption:", sol["consumption"])
    print("Objective value:", sol["objective"])