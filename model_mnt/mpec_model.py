import pyomo.environ as pyo


class Actor:
    def __init__(self, actor_id, generators, is_strategic=False):
        """
        Parameters:
        -----------
        actor_id : int
            Unique identifier for the actor
        generators : list[Generator]
            Generators owned by this actor
        is_strategic : bool
            Whether this actor is a strategic bidder
        """
        self.actor_id = actor_id
        self.generators = generators if isinstance(generators, list) else [generators]
        self.is_strategic = is_strategic

class Generator:
    def __init__(self, p_max, cost, p_min=0):
        self.p_max = p_max
        self.cost = cost
        self.p_min = p_min

class mpec:
    def __init__(self, demand, actors, generators, market_bounds=(0, 1200), fixed_bids=None):
        self.demand = demand
        self.actors = actors

        # Flatten the list of generators across all actors
        self.generators = []
        self.gen_to_actor = {}
        for a in actors:
            for g in a.generators:
                self.gen_to_actor[len(self.generators)] = a.actor_id
                self.generators.append(g)

        self.n_generators = len(self.generators)  # <-- fixed

        self.market_bounds = market_bounds
        self.fixed_bids = fixed_bids

        self.M = 100000  # Big M parameter
        self.M_tau = 10000  # Big M for bid difference

        self.model = self._build_model()
        
    def _build_model(self):
        model = pyo.ConcreteModel()
        
        # Sets
        model.G = pyo.RangeSet(0, self.n_generators - 1)
        
        # model.J might be empty; create as a Set (Pyomo handles empty sets)
        strategic_idx = [
            i for i in range(self.n_generators)
            if any(a.actor_id == self.gen_to_actor[i] and a.is_strategic for a in self.actors)
        ]
        model.J = pyo.Set(initialize=strategic_idx)

        # Now add the variables, objective and constraints using your helpers
        self._add_variables(model)
        self._add_objective(model)
        self._add_constraints(model)

        return model
    
    def _add_variables(self, model):
        # UL Primal Variables
        model.alpha = pyo.Var(model.G, within=pyo.NonNegativeReals)
        model.p = pyo.Var(model.G, within=pyo.NonNegativeReals)
        
        # LL Dual Variables
        model.lmbda = pyo.Var(domain=pyo.Reals)
        model.mu_min = pyo.Var(model.G, within=pyo.NonNegativeReals)
        model.mu_max = pyo.Var(model.G, within=pyo.NonNegativeReals)
        
        # Auxiliary Variables
        model.z_min = pyo.Var(model.G, domain=pyo.Binary)
        model.z_max = pyo.Var(model.G, domain=pyo.Binary)

        model.z_tau = pyo.Var(
            [(j, g) for j in model.J for g in model.G if g not in model.J],
            domain=pyo.Binary
            )

    def _add_objective(self, model):
        model.obj = pyo.Objective(
            expr=sum(-model.lmbda*model.p[i] + self.generators[i].cost*model.p[i] 
                    for i in model.J),
            sense=pyo.minimize
        )

    # def _add_objective(self, model):
    #     model.obj = pyo.Objective(
    #         expr=sum(-model.lmbda*model.p[i] + self.generators[i].cost*model.p[i] 
    #                 for i in model.J),
    #         sense=pyo.minimize
    #     )
    
    def _add_constraints(self, model):
        # Bid bounds
        model.bid_min = pyo.Constraint(
            model.J, 
            rule=lambda m, i: self.market_bounds[0] <= m.alpha[i]
        )
        model.bid_max = pyo.Constraint(
            model.J, 
            rule=lambda m, i: m.alpha[i] <= self.market_bounds[1]
        )

        # Non-strategic generator bids
        def bid_rule(m, i):
            if self.fixed_bids is not None:  # use last round bids
                return m.alpha[i] == self.fixed_bids[i]
            else:  # first round, truthful bids
                return m.alpha[i] == self.generators[i].cost

        model.non_strategic_bid = pyo.Constraint(
            [i for i in model.G if i not in model.J],
            rule=bid_rule
        )
        
        # Bids between generators should be different
        model.bid_diff_higher = pyo.Constraint(
            [(j, g) for j in model.J for g in model.G if g not in model.J],
            rule=lambda m, j, g: m.alpha[j] >= m.alpha[g] + 0.01 - self.M_tau*(1 - m.z_tau[j,g])
        )

        model.bid_diff_lower = pyo.Constraint(
            [(j, g) for j in model.J for g in model.G if g not in model.J],
            rule=lambda m, j, g: m.alpha[j] <= m.alpha[g] - 0.01 + self.M_tau*(m.z_tau[j,g])
        )

        # # KKT conditions
        # def kkt_rule(m, i):
        #     if i in m.J:
        #         return m.alpha[i] - m.lmbda - m.mu_min[i] + m.mu_max[i] == 0
        #     return self.generators[i].cost - m.lmbda - m.mu_min[i] + m.mu_max[i] == 0
        
                # KKT conditions: always use bids (alpha) as the marginal cost in the stationarity condition
        def kkt_rule(m, i):
            # Use alpha for both strategic and non-strategic generators
            return m.alpha[i] - m.lmbda - m.mu_min[i] + m.mu_max[i] == 0

        model.lngrn = pyo.Constraint(model.G, rule=kkt_rule)
        
        # Power balance
        model.balance = pyo.Constraint(
            expr=self.demand - sum(model.p[i] for i in model.G) == 0
        )
        
        # Capacity constraints
        model.cap_max = pyo.Constraint(
            model.G, 
            rule=lambda m, i: m.p[i] <= self.generators[i].p_max
        )
        
        # Complementarity constraints
        model.cs_g_min = pyo.Constraint(
            model.G, 
            rule=lambda m, i: -(self.generators[i].p_min - m.p[i]) <= self.M*m.z_min[i]
        )
        model.cs_mu_min = pyo.Constraint(
            model.G, 
            rule=lambda m, i: m.mu_min[i] <= self.M*(1 - m.z_min[i])
        )
        model.cs_g_max = pyo.Constraint(
            model.G, 
            rule=lambda m, i: -(m.p[i] - self.generators[i].p_max) <= self.M*m.z_max[i]
        )
        model.cs_mu_max = pyo.Constraint(
            model.G, 
            rule=lambda m, i: m.mu_max[i] <= self.M*(1 - m.z_max[i])
        )
    
    def solve(self, solver_name="gurobi", tee=False):
        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(self.model, tee=tee)
        return results
    
    def print_results(self):
        print("Solver status:", self.model.obj())
        print("\nDispatch results:")
        for i in self.model.G:
            print(f"Generator {i}: p = {pyo.value(self.model.p[i]):.2f}, "
                  f"alpha = {pyo.value(self.model.alpha[i]):.2f}, "
                  f"mu_min = {pyo.value(self.model.mu_min[i]):.2f}, "
                  f"mu_max = {pyo.value(self.model.mu_max[i]):.2f}")
        print("\nMarket clearing price (lambda):", pyo.value(self.model.lmbda))

class market_clearing:
    def __init__(self, demand, generators):
        """
        Lower-level market clearing model (no strategic actors).
        
        Parameters
        ----------
        demand : float
            Total system demand.
        generators : list[Generator]
            List of Generator objects.
        """
        self.demand = demand
        self.generators = generators
        self.n_generators = len(generators)
        self.model = self._build_model()
    
    def _build_model(self):
        model = pyo.ConcreteModel()
        
        # Sets
        model.G = pyo.RangeSet(0, self.n_generators - 1)
        
        # Variables
        model.p = pyo.Var(model.G, within=pyo.NonNegativeReals)  # dispatch
        
        # Objective: minimize total cost
        model.obj = pyo.Objective(
            expr=sum(self.generators[i].cost * model.p[i] for i in model.G),
            sense=pyo.minimize
        )
        
        # Demand balance
        model.balance = pyo.Constraint(
            expr=sum(model.p[i] for i in model.G) == self.demand
        )
        
        # Capacity bounds
        model.cap_min = pyo.Constraint(
            model.G, rule=lambda m, i: m.p[i] >= self.generators[i].p_min
        )
        model.cap_max = pyo.Constraint(
            model.G, rule=lambda m, i: m.p[i] <= self.generators[i].p_max
        )
        
        # Store dual variables (for market clearing price)
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        
        return model
    
    def solve(self, solver_name="gurobi", tee=False):
        solver = pyo.SolverFactory(solver_name)
        results = solver.solve(self.model, tee=tee)
        return results
    
    def print_results(self):
        print("Objective value (total cost):", pyo.value(self.model.obj))
        print("\nDispatch results:")
        for i in self.model.G:
            print(f"Generator {i}: "
                  f"p = {pyo.value(self.model.p[i]):.2f}, "
                  f"cost = {self.generators[i].cost}")
        # Market clearing price = dual of balance
        lmbda = self.model.dual[self.model.balance]
        print("\nMarket clearing price (lambda):", lmbda)

# Usage example:
if __name__ == "__main__":
    # Create generators
    g0 = Generator(p_max=40, cost=17)
    g1 = Generator(p_max=50, cost=15)
    g2 = Generator(p_max=30, cost=12)


    # Create actors and assign generators
    actors = [
        Actor(actor_id=0, generators=[g0], is_strategic=True),   # owns g0, strategic
        Actor(actor_id=1, generators=[g1], is_strategic=False),  # owns g1
        Actor(actor_id=2, generators=[g2], is_strategic=False),  # owns g2
    ]

    # Flatten generator list
    generators = [g0, g1, g2]

    # Create and solve market model
    market = mpec(demand=100, market_bounds=(-100, 100), actors=actors, generators=generators)
    results = market.solve()
    market.print_results()

    mc = market_clearing(demand=100, generators=generators)
    mc.solve()
    mc.print_results()