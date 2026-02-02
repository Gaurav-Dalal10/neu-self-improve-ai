"""
Airline Overbooking MDP — Enhanced Version
===========================================
Features:
- State: (tickets_sold, time_remaining)
- Actions: 3 (Reject, Discount 20%, Full Price)
- Stochastic demand (request may or may not arrive)
- Monte Carlo evaluation with baseline comparisons

Data Source: Bureau of Transportation Statistics (BTS)
https://data.transportation.gov/d/xyfb-hgtv
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


# =============================================================================
# MDP CLASS
# =============================================================================

class AirlineOverbookingMDP:
    """
    Enhanced MDP with time horizon and multiple pricing actions.
    
    State: (tickets_sold, periods_remaining)
    Action: 0=Reject, 1=Discount (20% off), 2=Full Price
    """
    
    def __init__(self, capacity=10, max_overbook=5, horizon=20,
                 price_full=300, discount_rate=0.20, bumping_cost=500,
                 show_prob=0.90, demand_prob=0.80, holding_cost=20):
        
        # Capacity
        self.capacity = capacity
        self.max_overbook = max_overbook
        self.max_tickets = capacity + max_overbook
        self.horizon = horizon  # Booking periods before departure
        
        # Pricing
        self.price_full = price_full
        self.price_discount = price_full * (1 - discount_rate)
        self.discount_rate = discount_rate
        self.bumping_cost = bumping_cost
        self.holding_cost = holding_cost  # Empty seat penalty
        
        # Probabilities
        self.show_prob = show_prob
        self.demand_prob = demand_prob  # Prob of request arriving each period
        
        # Demand sensitivity: discount increases acceptance probability
        self.accept_prob_full = 0.70      # 70% accept full price
        self.accept_prob_discount = 0.95  # 95% accept discount
        
        # State space: (tickets, time)
        self.states = []
        for n in range(self.max_tickets + 1):
            for t in range(self.horizon + 1):
                self.states.append((n, t))
        self.n_states = len(self.states)
        
        # Actions
        self.actions = [0, 1, 2]
        self.action_names = ['Reject', 'Discount', 'Full Price']
        self.n_actions = len(self.actions)
        
        # Precompute bump costs
        self._precompute_bump_costs()
        
        print(f"MDP Created:")
        print(f"  States: {self.n_states} = {self.max_tickets+1} tickets × {self.horizon+1} periods")
        print(f"  Actions: {self.n_actions} (Reject, Discount, Full)")
        print(f"  Capacity: {self.capacity}, Max: {self.max_tickets}")
    
    def _precompute_bump_costs(self):
        """Precompute expected bumping cost for each ticket count."""
        self.expected_bump_cost = {}
        self.bump_probability = {}
        for n in range(self.max_tickets + 1):
            cost = 0
            prob = 0
            for k in range(n + 1):
                p_k = binom.pmf(k, n, self.show_prob)
                bumped = max(0, k - self.capacity)
                cost += p_k * bumped * self.bumping_cost
                if bumped > 0:
                    prob += p_k
            self.expected_bump_cost[n] = cost
            self.bump_probability[n] = prob
    
    def terminal_value(self, n_tickets):
        """Value at departure: revenue already collected minus expected bump cost."""
        expected_show = n_tickets * self.show_prob
        empty_seats = max(0, self.capacity - expected_show)
        empty_penalty = empty_seats * self.holding_cost
        return -self.expected_bump_cost[n_tickets] - empty_penalty


# =============================================================================
# POLICY ITERATION
# =============================================================================

def policy_iteration(mdp, verbose=True):
    """
    Policy Iteration considering:
    - Demand may or may not arrive
    - Customer may or may not accept the offered price
    """
    
    # Policy: state -> action
    policy = {s: 2 for s in mdp.states}  # Initial: always full price
    
    history = []
    
    for iteration in range(100):
        # === Policy Evaluation ===
        V = {s: 0.0 for s in mdp.states}
        
        for _ in range(500):
            V_new = {}
            
            for s in mdp.states:
                n, t = s
                
                if t == 0:  # Departure time - terminal
                    V_new[s] = mdp.terminal_value(n)
                elif n >= mdp.max_tickets:  # Can't sell more
                    V_new[s] = V.get((n, t-1), mdp.terminal_value(n))
                else:
                    action = policy[s]
                    V_new[s] = compute_action_value(mdp, s, action, V)
            
            max_diff = max(abs(V_new[s] - V.get(s, 0)) for s in mdp.states)
            V = V_new
            if max_diff < 1e-6:
                break
        
        history.append({'iteration': iteration + 1, 'V_0': V[(0, mdp.horizon)]})
        if verbose:
            print(f"Iteration {iteration + 1}: V(0, {mdp.horizon}) = ${V[(0, mdp.horizon)]:.2f}")
        
        # === Policy Improvement ===
        policy_stable = True
        new_policy = {}
        
        for s in mdp.states:
            n, t = s
            
            if t == 0 or n >= mdp.max_tickets:
                new_policy[s] = 0
            else:
                best_action = 0
                best_value = -float('inf')
                
                for a in mdp.actions:
                    val = compute_action_value(mdp, s, a, V)
                    if val > best_value:
                        best_value = val
                        best_action = a
                
                new_policy[s] = best_action
            
            if policy.get(s) != new_policy[s]:
                policy_stable = False
        
        policy = new_policy
        
        if policy_stable:
            if verbose:
                print(f"\nConverged in {iteration + 1} iterations!")
            break
    
    return policy, V, history


def compute_action_value(mdp, state, action, V):
    """Compute Q(s, a) considering demand and acceptance probabilities."""
    n, t = state
    
    if action == 0:  # Reject - no sale, just wait
        next_state = (n, t - 1)
        return V.get(next_state, mdp.terminal_value(n))
    
    # For accept actions, consider demand arrival and acceptance
    if action == 1:  # Discount
        price = mdp.price_discount
        accept_prob = mdp.accept_prob_discount
    else:  # Full price
        price = mdp.price_full
        accept_prob = mdp.accept_prob_full
    
    # Case 1: No demand arrives (prob = 1 - demand_prob)
    no_demand_value = V.get((n, t - 1), mdp.terminal_value(n))
    
    # Case 2: Demand arrives (prob = demand_prob)
    # Sub-case 2a: Customer accepts
    if n < mdp.max_tickets:
        accept_value = price + V.get((n + 1, t - 1), mdp.terminal_value(n + 1))
    else:
        accept_value = V.get((n, t - 1), mdp.terminal_value(n))
    
    # Sub-case 2b: Customer rejects our price
    reject_value = V.get((n, t - 1), mdp.terminal_value(n))
    
    demand_value = accept_prob * accept_value + (1 - accept_prob) * reject_value
    
    expected_value = (1 - mdp.demand_prob) * no_demand_value + mdp.demand_prob * demand_value
    
    return expected_value


# =============================================================================
# MONTE CARLO EVALUATION
# =============================================================================

def monte_carlo_evaluation(mdp, policy, n_simulations=10000):
    """Evaluate policy with Monte Carlo simulation."""
    
    profits = []
    tickets_sold = []
    bumped_counts = []
    
    for _ in range(n_simulations):
        n = 0  # Tickets sold
        revenue = 0
        
        # Booking period
        for t in range(mdp.horizon, 0, -1):
            if n >= mdp.max_tickets:
                break
            
            state = (n, t)
            action = policy.get(state, 0)
            
            if action == 0:  # Reject
                continue
            
            # Check if demand arrives
            if np.random.random() > mdp.demand_prob:
                continue
            
            # Determine price and acceptance
            if action == 1:
                price = mdp.price_discount
                accept_prob = mdp.accept_prob_discount
            else:
                price = mdp.price_full
                accept_prob = mdp.accept_prob_full
            
            # Customer decision
            if np.random.random() < accept_prob:
                revenue += price
                n += 1
        
        # Departure: simulate show-ups
        showed_up = np.random.binomial(n, mdp.show_prob)
        bumped = max(0, showed_up - mdp.capacity)
        bump_cost = bumped * mdp.bumping_cost
        
        # Empty seat penalty
        empty = max(0, mdp.capacity - showed_up)
        empty_cost = empty * mdp.holding_cost
        
        profit = revenue - bump_cost - empty_cost
        
        profits.append(profit)
        tickets_sold.append(n)
        bumped_counts.append(bumped)
    
    return {
        'mean_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'mean_tickets': np.mean(tickets_sold),
        'bump_probability': np.mean([b > 0 for b in bumped_counts]),
        'mean_bumped': np.mean(bumped_counts),
        'profits': profits
    }


def create_baseline_policies(mdp):
    """Create baseline policies."""
    
    # Always full price
    full_price = {s: 2 for s in mdp.states}
    
    # Always discount
    discount = {s: 1 for s in mdp.states}
    
    # No overbooking
    conservative = {}
    for s in mdp.states:
        n, t = s
        conservative[s] = 2 if n < mdp.capacity else 0
    
    # Threshold at 12 (20% overbook)
    threshold_12 = {}
    for s in mdp.states:
        n, t = s
        threshold_12[s] = 2 if n < 12 else 0
    
    return {
        'Always Full Price': full_price,
        'Always Discount': discount,
        'No Overbooking': conservative,
        'Threshold-12 (20%)': threshold_12
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AIRLINE OVERBOOKING MDP - ENHANCED VERSION")
    print("=" * 70)
    print("\nData Source: BTS Denied Boarding Data")
    print("https://data.transportation.gov/d/xyfb-hgtv\n")
    
    mdp = AirlineOverbookingMDP(
        capacity=10,
        max_overbook=5,
        horizon=20,
        price_full=300,
        discount_rate=0.20,
        bumping_cost=500,
        show_prob=0.90,
        demand_prob=0.80,
        holding_cost=20
    )
    
    print(f"\nParameters:")
    print(f"  Full price:     ${mdp.price_full}")
    print(f"  Discount price: ${mdp.price_discount} ({mdp.discount_rate:.0%} off)")
    print(f"  Bumping cost:   ${mdp.bumping_cost}")
    print(f"  Show-up prob:   {mdp.show_prob:.0%}")
    print(f"  Demand prob:    {mdp.demand_prob:.0%} per period")
    
    print("\n" + "=" * 70)
    print("RUNNING POLICY ITERATION")
    print("=" * 70 + "\n")
    
    policy, V, history = policy_iteration(mdp, verbose=True)
    
    # === Analyze Policy Structure ===
    print("\n" + "=" * 70)
    print("OPTIMAL POLICY STRUCTURE")
    print("=" * 70)
    
    print(f"\nPolicy at t={mdp.horizon} (start of booking):")
    print(f"{'Tickets':<10} {'Action':<15} {'Bump Risk':<12}")
    print("-" * 40)
    for n in range(mdp.max_tickets + 1):
        state = (n, mdp.horizon)
        action = mdp.action_names[policy.get(state, 0)]
        bump = mdp.bump_probability[n]
        print(f"{n:<10} {action:<15} {bump:.1%}")
    
    # Find threshold
    threshold = mdp.max_tickets
    for n in range(mdp.max_tickets + 1):
        if policy.get((n, mdp.horizon), 0) == 0:
            threshold = n
            break
    
    print(f"\n→ Optimal threshold: {threshold} tickets ({(threshold-mdp.capacity)/mdp.capacity*100:.0f}% overbooking)")
    
    # === Monte Carlo Evaluation ===
    print("\n" + "=" * 70)
    print("MONTE CARLO EVALUATION (10,000 simulations)")
    print("=" * 70 + "\n")
    
    mc_optimal = monte_carlo_evaluation(mdp, policy)
    baselines = create_baseline_policies(mdp)
    
    results = {'Optimal Policy': mc_optimal}
    for name, bp in baselines.items():
        results[name] = monte_carlo_evaluation(mdp, bp)
    
    print(f"{'Policy':<25} {'Profit':<12} {'Std':<10} {'Tickets':<10} {'Bump%':<10}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<25} ${r['mean_profit']:<11.2f} ${r['std_profit']:<9.2f} "
              f"{r['mean_tickets']:<10.1f} {r['bump_probability']*100:<10.1f}")
    
    # === Save Results ===
    os.makedirs("results/figures", exist_ok=True)
    
    with open("results/policy_table.csv", "w") as f:
        f.write("tickets_sold,time_remaining,action\n")
        for s in mdp.states:
            n, t = s
            f.write(f"{n},{t},{mdp.action_names[policy.get(s, 0)]}\n")
    
    with open("results/value_function.csv", "w") as f:
        f.write("tickets_sold,time_remaining,value\n")
        for s in mdp.states:
            f.write(f"{s[0]},{s[1]},{V.get(s, 0):.2f}\n")
    
    with open("results/monte_carlo_evaluation.csv", "w") as f:
        f.write("policy,mean_profit,std_profit,mean_tickets,bump_probability\n")
        for name, r in results.items():
            f.write(f"{name},{r['mean_profit']:.2f},{r['std_profit']:.2f},"
                    f"{r['mean_tickets']:.2f},{r['bump_probability']:.4f}\n")
    
    with open("results/policy_iteration_history.csv", "w") as f:
        f.write("iteration,V_0\n")
        for h in history:
            f.write(f"{h['iteration']},{h['V_0']:.2f}\n")
    
    # === Figures ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Policy comparison
    ax = axes[0, 0]
    names = list(results.keys())
    profits = [results[n]['mean_profit'] for n in names]
    colors = ['green' if 'Optimal' in n else 'steelblue' for n in names]
    ax.barh(names, profits, color=colors)
    ax.set_xlabel('Mean Profit ($)')
    ax.set_title('Policy Comparison (Monte Carlo)')
    for i, p in enumerate(profits):
        ax.text(p + 20, i, f'${p:.0f}', va='center')
    
    # 2. Value function
    ax = axes[0, 1]
    tickets = range(mdp.max_tickets + 1)
    values = [V.get((n, mdp.horizon), 0) for n in tickets]
    ax.plot(tickets, values, 'o-', linewidth=2, color='steelblue')
    ax.axvline(mdp.capacity, color='orange', linestyle='--', label='Capacity (10)')
    ax.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold})')
    ax.set_xlabel('Tickets Sold')
    ax.set_ylabel('Expected Value ($)')
    ax.set_title('Value Function at Start of Booking')
    ax.legend()
    
    # 3. Profit distribution
    ax = axes[1, 0]
    ax.hist(results['Optimal Policy']['profits'], bins=50, alpha=0.7, 
            color='green', label='Optimal')
    ax.hist(results['No Overbooking']['profits'], bins=50, alpha=0.5,
            color='orange', label='No Overbooking')
    ax.set_xlabel('Profit ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Profit Distribution')
    ax.legend()
    
    # 4. Bump risk
    ax = axes[1, 1]
    bump_probs = [results[n]['bump_probability']*100 for n in names]
    colors = ['green' if 'Optimal' in n else 'salmon' for n in names]
    ax.barh(names, bump_probs, color=colors)
    ax.set_xlabel('Bump Probability (%)')
    ax.set_title('Overbooking Risk by Policy')
    
    plt.tight_layout()
    plt.savefig("results/figures/mdp_results.png", dpi=150)
    plt.close()
    
    # === Summary ===
    best_baseline = max(baselines.keys(), key=lambda x: results[x]['mean_profit'])
    improvement = ((results['Optimal Policy']['mean_profit'] - 
                   results[best_baseline]['mean_profit']) /
                   results[best_baseline]['mean_profit'] * 100)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nOptimal Policy:")
    print(f"  Threshold:        {threshold} tickets ({(threshold-mdp.capacity)/mdp.capacity*100:.0f}% overbooking)")
    print(f"  Mean Profit:      ${results['Optimal Policy']['mean_profit']:,.2f}")
    print(f"  Std Dev:          ${results['Optimal Policy']['std_profit']:,.2f}")
    print(f"  Bump Probability: {results['Optimal Policy']['bump_probability']:.1%}")
    print(f"  Avg Tickets:      {results['Optimal Policy']['mean_tickets']:.1f}")
    print(f"\n  vs {best_baseline}: {improvement:+.1f}%")
    
    print("\nFiles saved to results/")
    print("Done!")
