import numpy as np

def generate_scaled_setup(n_players: int, P_max_ref: list, demand_ref: float, cost_ref: list):
    """
    Generates ordered generator parameters for n_players,
    preserving the relative pattern from P_max_ref
    Keeps total system capacity constant for comparable social welfare.
    """

    # --- Reference pattern base case ---
    base_pattern = np.array(P_max_ref)
    base_pattern_sum = base_pattern.sum()
    base_pattern = base_pattern / base_pattern.sum()  # normalize

    # --- Interpolate to match n_players ---
    x_base = np.linspace(0, 1, len(base_pattern))
    x_new = np.linspace(0, 1, n_players)
    pattern_scaled = np.interp(x_new, x_base, base_pattern)

    # Normalize so total capacity is constant (≈165)
    Pmax = pattern_scaled / pattern_scaled.sum() * base_pattern_sum
    Pmin = np.zeros(n_players)

    # --- Costs: follow same increasing pattern ---
    base_cost_min = np.array(cost_ref)

    cost_pattern = np.interp(x_new, x_base, base_cost_min)

    cost = np.round(cost_pattern, 1)

    demand = demand_ref

    return (
        Pmin.tolist(),
        Pmax.round(1).tolist(),
        cost.tolist(),
        demand,
    )
