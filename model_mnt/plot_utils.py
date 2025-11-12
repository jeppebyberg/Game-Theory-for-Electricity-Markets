import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pyomo.environ as pyo

def extract_dispatch_results(model, generators, is_mpec=False):
    """
    Extract dispatch results from a solved Pyomo model.
    
    Parameters
    ----------
    model : pyomo.ConcreteModel
        The solved model (from market_clearing or mpec).
    generators : list[Generator]
        Generator objects.
    is_mpec : bool
        If True, uses alpha (strategic bids). Otherwise uses true cost.
    
    Returns
    -------
    bids : list[float]
        Bid or cost for each generator.
    capacities : list[float]
        Capacity for each generator.
    dispatch : list[float]
        Actual dispatched output.
    """
    bids, capacities, dispatch = [], [], []
    for i, g in enumerate(generators):
        bid = pyo.value(model.alpha[i]) if is_mpec else g.cost
        p = pyo.value(model.p[i])
        bids.append(bid)
        capacities.append(g.p_max)
        dispatch.append(p)
    return bids, capacities, dispatch

def plot_merit_order_comparison(generators, mc_model, mpec_model, demand=None, actors=None):
    """
    Plot market clearing and MPEC models side by side for easy comparison.
    
    Parameters
    ----------
    generators : list[Generator]
    mc_model : pyomo.ConcreteModel
        Solved market_clearing model.
    mpec_model : pyomo.ConcreteModel
        Solved mpec model.
    demand : float, optional
        Total demand to show as vertical line.
    actors : list[Actor], optional
        List of actors to determine which generators are strategic.
    """
    
    # Helper function to determine if generator is strategic
    def is_generator_strategic(gen_idx):
        if actors is not None:
            for actor in actors:
                if generators[gen_idx] in actor.generators and actor.is_strategic:
                    return True
        elif hasattr(generators[gen_idx], 'is_strategic'):
            return generators[gen_idx].is_strategic
        return False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for generator blocks
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightgray']
    
    # ==========================================
    # LEFT PLOT: Market Clearing (Competitive)
    # ==========================================
    
    # Extract market clearing data (no strategic bids)
    bids_mc = [gen.cost for gen in generators]
    caps_mc = [gen.p_max for gen in generators]
    dispatch_mc = [pyo.value(mc_model.p[i]) for i in range(len(generators))]
    
    # Sort by cost (merit order)
    sorted_mc = sorted(zip(bids_mc, caps_mc, dispatch_mc, range(len(generators))), key=lambda x: x[0])
    
    # Draw generator blocks
    x_pos = 0
    for i, (bid, cap, dispatch, gen_idx) in enumerate(sorted_mc):
        # Generator block
        color = colors[gen_idx % len(colors)]
        rect = patches.Rectangle((x_pos, 0), cap, bid, 
                            #    linewidth=2, edgecolor='blue', 
                               facecolor=color, alpha=0.7)
        ax1.add_patch(rect)
        
        # Generator label
        gen_type = "S" if is_generator_strategic(gen_idx) else "C"
        ax1.text(x_pos + cap/2, bid/2, f'G{gen_idx}({gen_type})\n{cap:.0f}MW\n${bid:.1f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Dispatch shading
        # if dispatch > 0.1:
        #     dispatch_rect = patches.Rectangle((x_pos, 0), min(dispatch, cap), bid,
        #                                    linewidth=0, facecolor='darkblue', alpha=0.4)
        #     ax1.add_patch(dispatch_rect)
        
        x_pos += cap
    
    # Merit order step curve
    x_mc, y_mc = [0], [sorted_mc[0][0]]
    total_cap = 0
    for bid, cap, _, _ in sorted_mc:
        total_cap += cap
        x_mc.append(total_cap)
        y_mc.append(bid)
    
    # ax1.step(x_mc, y_mc, where="post", color="blue", linewidth=3, alpha=0.8)
    

    # Market clearing price and demand
    try:
        mc_price = mc_model.dual[mc_model.balance]
        ax1.axhline(y=mc_price, color='blue', linewidth=2, linestyle='--', alpha=0.7)
        ax1.text(0.02, 0.95, f'Market Price: ${mc_price:.2f}/MWh', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                facecolor='white',
                bbox=dict(boxstyle='round', alpha=0.8))
    except:
        mc_price = None
    
    if demand is not None:
        ax1.axvline(x=demand, color='orange', linewidth=3, linestyle=':', alpha=0.8)
        if mc_price is not None:
            ax1.plot(demand, mc_price, 'o', color='blue', markersize=10)
    
    # Styling for left plot
    ax1.set_xlabel("Cumulative Capacity (MW)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Cost ($/MWh)", fontsize=12, fontweight='bold')
    ax1.set_title("Market Clearing (Competitive Bidding)", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ==========================================
    # RIGHT PLOT: MPEC (Strategic Bidding)
    # ==========================================
    
    # Extract MPEC data (with strategic bids)
    bids_mpec, caps_mpec, dispatch_mpec = extract_dispatch_results(mpec_model, generators, is_mpec=True)
    
    # Sort by bid price (strategic merit order)
    sorted_mpec = sorted(zip(bids_mpec, caps_mpec, dispatch_mpec, range(len(generators))), key=lambda x: x[0])
    
    # Draw generator blocks
    x_pos = 0
    for i, (bid, cap, dispatch, gen_idx) in enumerate(sorted_mpec):
        # Generator block
        color = colors[gen_idx % len(colors)]
        rect = patches.Rectangle((x_pos, 0), cap, bid, 
                            #    linewidth=2, edgecolor='red', 
                               facecolor=color, alpha=0.7)
        ax2.add_patch(rect)
        
        # Generator label with markup indication
        gen_type = "S" if is_generator_strategic(gen_idx) else "C"
        true_cost = generators[gen_idx].cost
        markup = bid - true_cost
        
        if markup > 0.1:  # Show markup for strategic generators
            label = f'G{gen_idx}({gen_type})\n{cap:.0f}MW\n${bid:.1f}\n(+${markup:.1f})'
        else:
            label = f'G{gen_idx}({gen_type})\n{cap:.0f}MW\n${bid:.1f}'
            
        ax2.text(x_pos + cap/2, bid/2, label, 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        
        x_pos += cap
    
    # Merit order step curve
    x_mpec, y_mpec = [0], [sorted_mpec[0][0]]
    total_cap = 0
    for bid, cap, _, _ in sorted_mpec:
        total_cap += cap
        x_mpec.append(total_cap)
        y_mpec.append(bid)
    
    #ax2.step(x_mpec, y_mpec, where="post", color="red", linewidth=3, alpha=0.8)
    
    # Market clearing price and demand
    try:
        mpec_price = pyo.value(mpec_model.lmbda)
        ax2.axhline(y=mpec_price, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax2.text(0.02, 0.95, f'Market Price: ${mpec_price:.2f}/MWh', 
                transform=ax2.transAxes, fontsize=12, fontweight='bold', color='red',
                facecolor='white',
                bbox=dict(boxstyle='round', alpha=0.8))
    except:
        mpec_price = None
    
    if demand is not None:
        ax2.axvline(x=demand, color='orange', linewidth=3, linestyle=':', alpha=0.8)
        if mpec_price is not None:
            ax2.plot(demand, mpec_price, 'o', color='red', markersize=10)
    
    # Styling for right plot
    ax2.set_xlabel("Cumulative Capacity (MW)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Bid Price ($/MWh)", fontsize=12, fontweight='bold')
    ax2.set_title("MPEC (Strategic Bidding)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ==========================================
    # Overall styling and comparison info
    # ==========================================
    
    # Set consistent y-axis limits
    max_price = max(max(y_mc), max(y_mpec)) * 1.1
    max_cap = sum(gen.p_max for gen in generators) * 1.1
    
    ax1.set_xlim(0, max_cap)
    ax1.set_ylim(0, max_price)
    ax2.set_xlim(0, max_cap)
    ax2.set_ylim(0, max_price)
    
    # Add comparison summary
    price_increase = 0
    pct_increase = 0
    print(f"mc_price: {mc_price}")
    print(f"mpec_price: {mpec_price}")
    if mc_price is not None and mpec_price is not None:
        price_increase = mpec_price - mc_price
        pct_increase = (price_increase / mc_price) * 100
        
        fig.suptitle(f'Merit Order Comparison\n'
                    f'Strategic Bidding Increases Price by ${price_increase:.2f}/MWh ({pct_increase:.1f}%)', 
                    fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Merit Order Comparison: Competitive vs Strategic Bidding', 
                    fontsize=16, fontweight='bold')
    
    # Add legend for dispatch shading
    # ax1.text(0.02, 0.85, 'Dark shading = Dispatched', 
    #         transform=ax1.transAxes, fontsize=10,
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.85, f'+${price_increase:.4f} Markup', 
            transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add demand line legend
    if demand is not None:
        ax1.text(0.02, 0.02, f'Orange line = Demand ({demand:.0f} MW)', 
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        ax2.text(0.02, 0.02, f'Orange line = Demand ({demand:.0f} MW)', 
                transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2)

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']

def plot_profit_history(profit_history, title="Actor Profits Across Iterations", save_path=None, colors=colors):
    """
    Plots the profit trajectories of actors across iterations.

    Parameters
    ----------
    profit_history : dict
        Dictionary {actor_id: [profits_per_iteration]}.
    title : str
        Plot title.
    save_path : str or None
        If given, saves the figure to this path instead of showing.
    """
    plt.figure(figsize=(8, 6))

    for actor_id, profits in profit_history.items():
        iterations = range(1, len(profits) + 1)
        plt.plot(iterations, profits, marker="o", label=f"Actor {actor_id}",
                color=colors[actor_id % len(colors)], linewidth=2, markersize=6)

    plt.xlabel("Iteration")
    plt.ylabel("Profit (Objective Value)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_bid_history(all_results, generators, title="Generator Bid History Across Iterations", save_path=None, colors=colors):
    """
    Plots the bid trajectories of generators across iterations from br_convergence results.
    Each iteration contains multiple rounds (one per actor/generator).

    Parameters
    ----------
    all_results : list[dict]
        Results from run_until_convergence(), where each dict contains 'bids', 'iteration', etc.
    generators : list[Generator]
        List of Generator objects to get true costs.
    title : str
        Plot title.
    save_path : str or None
        If given, saves the figure to this path instead of showing.
    """
    # Extract bid history for each generator grouped by iteration
    n_generators = len(generators)
    bid_history = {i: [] for i in range(n_generators)}
    iterations = []
    
    # Group results by iteration and collect one bid per generator per iteration
    current_iteration = None
    iteration_bids = None
    
    for result in all_results:
        if result["iteration"] != current_iteration:
            # New iteration - store previous iteration's final bids if exists
            if iteration_bids is not None:
                iterations.append(current_iteration)
                for gen_idx in range(n_generators):
                    bid_history[gen_idx].append(iteration_bids[gen_idx])
            
            # Start tracking new iteration
            current_iteration = result["iteration"]
            iteration_bids = result["bids"].copy()
        else:
            # Same iteration - update with latest bids
            iteration_bids = result["bids"].copy()
    
    # Don't forget the last iteration
    if iteration_bids is not None:
        iterations.append(current_iteration)
        for gen_idx in range(n_generators):
            bid_history[gen_idx].append(iteration_bids[gen_idx])
    
    plt.figure(figsize=(8, 6))
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bid trajectories
    for gen_idx in range(n_generators):
        true_cost = generators[gen_idx].cost
        ax.plot(iterations, bid_history[gen_idx], 
                marker="o", label=f"G{gen_idx} (cost=${true_cost:.1f})", 
                color=colors[gen_idx], linewidth=2, markersize=6)
        
        # Add horizontal line for true cost
        ax.axhline(y=true_cost, color=colors[gen_idx], 
                  linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel("Iteration", fontsize=12, fontweight='bold')
    ax.set_ylabel("Bid Price ($/MWh)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    # Set x-axis to show integer iterations
    if iterations:
        ax.set_xticks(iterations)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    return fig, ax


# Usage example
# def example_usage():
#     """
#     Example showing how to use the side-by-side comparison
#     """
#     print("Side-by-side merit order plotting ready!")
#     print("Usage:")
#     print("plot_merit_order_comparison(generators, mc.model, m.model, demand=1000, actors=actors)")
    
# if __name__ == "__main__":
#     example_usage()

def plot_price_by_actor(all_results, generators, title="Market Clearing Price by Strategic Actor", save_path=None, colors=colors):
    """
    Plots market clearing price evolution with one line per actor.
    
    Parameters
    ----------
    all_results : list[dict]
        Results from run_until_convergence().
    generators : list[Generator]
        List of Generator objects.
    title : str
        Plot title.
    save_path : str or None
        If given, saves the figure to this path.
    """
    n_actors = len(generators)
    
    # Group results by actor and iteration
    actor_prices = {i: [] for i in range(n_actors)}
    iterations_by_actor = {i: [] for i in range(n_actors)}
    
    for result in all_results:
        actor_id = result["strategic_actor"]
        iteration = result["iteration"]
        price = result["lambda"]
        
        actor_prices[actor_id].append(price)
        iterations_by_actor[actor_id].append(iteration)
    
    # Create plot with consistent colors
    fig, ax = plt.subplots(figsize=(10, 6))

    
    for actor_id in range(n_actors):
        ax.plot(iterations_by_actor[actor_id], actor_prices[actor_id], 
                marker="o", label=f"Actor {actor_id}", 
                color=colors[actor_id % len(colors)], linewidth=2, markersize=6)
    
    ax.set_xlabel("Iteration", fontsize=12, fontweight='bold')
    ax.set_ylabel("Market Clearing Price ($/MWh)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    return fig, ax


def plot_dispatch_comparison_from_results(all_results, generators, mc_model, 
                                         title="Dispatch Comparison: Social Optimum vs Equilibrium", 
                                         save_path=None, colors=colors):
    """
    Bar chart comparing social optimum dispatch to final equilibrium dispatch.
    
    Parameters
    ----------
    all_results : list[dict]
        Results from run_until_convergence().
    generators : list[Generator]
        List of Generator objects.
    mc_model : market_clearing
        Solved market_clearing object (not the Pyomo model itself).
    title : str
        Plot title.
    save_path : str or None
        If given, saves the figure to this path.
    """
    n_generators = len(generators)
    
    # Get social optimum dispatch from market clearing model
    mc_dispatch = [pyo.value(mc_model.model.p[i]) for i in range(n_generators)]
    
    # Get final equilibrium dispatch from last result
    final_dispatch = [all_results[-1]["dispatch"][i] for i in range(n_generators)]
    
    # Create bar chart with consistent colors
    x = np.arange(n_generators)
    width = 0.35

    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars with generator-specific colors
    for i in range(n_generators):
        ax.bar(x[i] - width/2, mc_dispatch[i], width, 
               color=colors[i % len(colors)], alpha=0.5, 
               label='Social Optimum (MC)' if i == 0 else '')
        ax.bar(x[i] + width/2, final_dispatch[i], width, 
               color=colors[i % len(colors)], alpha=1.0,
               label='Equilibrium (MPEC)' if i == 0 else '')
    
    ax.set_xlabel('Generator', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dispatch (MW)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'G{i}' for i in range(n_generators)])
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    
    return fig, ax