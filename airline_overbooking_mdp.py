"""
Airline Overbooking MDP — Policy Iteration
===========================================
Finds optimal overbooking policy using Policy Iteration.

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
    """MDP for airline seat overbooking."""
    
    def __init__(self, capacity=10, max_tickets=15, ticket_price=300, 
                 bumping_cost=500, show_prob=0.90):
        self.capacity = capacity
        self.max_tickets = max_tickets
        self.ticket_price = ticket_price
        self.bumping_cost = bumping_cost
        self.show_prob = show_prob
        self.n_states = max_tickets + 1
        
        # Pre-compute expected bumping costs
        self.expected_bump_cost = self._compute_bump_costs()
    
    def _compute_bump_costs(self):
        """E[Bumping Cost] for each number of tickets sold."""
        costs = np.zeros(self.n_states)
        for n in range(self.n_states):
            for k in range(n + 1):
                prob = binom.pmf(k, n, self.show_prob)
                bumped = max(0, k - self.capacity)
                costs[n] += prob * bumped * self.bumping_cost
        return costs
    
    def terminal_value(self, n):
        """Terminal profit when n tickets are sold."""
        return n * self.ticket_price - self.expected_bump_cost[n]


# =============================================================================
# POLICY ITERATION
# =============================================================================

def policy_iteration(mdp, verbose=True):
    """Solve MDP using Policy Iteration."""
    
    policy = np.ones(mdp.n_states, dtype=int)  # Start: always accept
    history = []
    
    for iteration in range(100):
        # Policy Evaluation
        V = np.zeros(mdp.n_states)
        for _ in range(1000):
            V_new = np.zeros(mdp.n_states)
            for s in range(mdp.n_states):
                if policy[s] == 1 and s < mdp.max_tickets:
                    V_new[s] = mdp.ticket_price + V[s + 1]
                else:
                    V_new[s] = mdp.terminal_value(s)
            if np.max(np.abs(V - V_new)) < 1e-6:
                break
            V = V_new
        
        history.append({'iteration': iteration + 1, 'V': V.copy()})
        if verbose:
            print(f"Iteration {iteration + 1}: V(0) = ${V[0]:.2f}")
        
        # Policy Improvement
        new_policy = np.zeros(mdp.n_states, dtype=int)
        for s in range(mdp.n_states):
            if s >= mdp.max_tickets:
                new_policy[s] = 0
            else:
                q_accept = mdp.ticket_price + V[s + 1]
                q_reject = mdp.terminal_value(s)
                new_policy[s] = 1 if q_accept > q_reject else 0
        
        if np.array_equal(policy, new_policy):
            if verbose:
                print(f"\nConverged in {iteration + 1} iterations!")
            break
        policy = new_policy
    
    return policy, V, history


def get_threshold(policy, max_tickets):
    """Find the overbooking threshold."""
    for s in range(max_tickets + 1):
        if policy[s] == 0:
            return s
    return max_tickets


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AIRLINE OVERBOOKING MDP - POLICY ITERATION")
    print("=" * 60)
    print("\nData Source: BTS Denied Boarding Data")
    print("https://data.transportation.gov/d/xyfb-hgtv\n")
    
    # Parameters from real BTS data
    mdp = AirlineOverbookingMDP(
        capacity=10,
        max_tickets=15,
        ticket_price=300,      # Avg domestic fare
        bumping_cost=500,      # BTS avg: $488.09
        show_prob=0.90         # 10% no-show rate
    )
    
    print(f"Parameters: C={mdp.capacity}, M={mdp.max_tickets}, "
          f"r=${mdp.ticket_price}, c_b=${mdp.bumping_cost}, p={mdp.show_prob}\n")
    
    # Solve
    policy, V, history = policy_iteration(mdp)
    
    # Results
    threshold = get_threshold(policy, mdp.max_tickets)
    no_overbook = mdp.terminal_value(mdp.capacity)
    improvement = (V[0] - no_overbook) / no_overbook * 100
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Optimal threshold:     {threshold} tickets")
    print(f"Overbooking rate:      {(threshold - mdp.capacity) / mdp.capacity * 100:.0f}%")
    print(f"Expected profit:       ${V[0]:,.2f}")
    print(f"No-overbook profit:    ${no_overbook:,.2f}")
    print(f"Improvement:           +{improvement:.1f}%")
    
    # Save results
    os.makedirs("results/figures", exist_ok=True)
    
    # CSV files
    with open("results/policy_table.csv", "w") as f:
        f.write("state,action,action_name\n")
        for s in range(mdp.n_states):
            f.write(f"{s},{policy[s]},{'Accept' if policy[s] == 1 else 'Reject'}\n")
    
    with open("results/value_function.csv", "w") as f:
        f.write("state,value,optimal_action\n")
        for s in range(mdp.n_states):
            f.write(f"{s},{V[s]:.2f},{'Accept' if policy[s] == 1 else 'Reject'}\n")
    
    with open("results/sensitivity_analysis.csv", "w") as f:
        f.write("show_prob,threshold,overbooking_pct,expected_profit\n")
        print("\nSensitivity Analysis:")
        for p in [0.70, 0.80, 0.85, 0.90, 0.95]:
            m = AirlineOverbookingMDP(show_prob=p)
            pol, val, _ = policy_iteration(m, verbose=False)
            th = get_threshold(pol, m.max_tickets)
            f.write(f"{p},{th},{(th-m.capacity)/m.capacity*100:.0f},{val[0]:.2f}\n")
            print(f"  p={p:.2f}: threshold={th}, profit=${val[0]:,.2f}")
    
    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    axes[0,0].bar(range(mdp.n_states), V, color=['steelblue' if policy[s]==1 else 'salmon' for s in range(mdp.n_states)])
    axes[0,0].axvline(mdp.capacity-0.5, color='orange', linestyle='--', label=f'Capacity')
    axes[0,0].set_xlabel('Tickets Sold'); axes[0,0].set_ylabel('V*(s)'); axes[0,0].set_title('Value Function'); axes[0,0].legend()
    
    axes[0,1].bar(range(mdp.n_states), policy, color=['green' if p==1 else 'red' for p in policy])
    axes[0,1].axvline(mdp.capacity-0.5, color='orange', linestyle='--')
    axes[0,1].set_xlabel('Tickets Sold'); axes[0,1].set_ylabel('Action'); axes[0,1].set_title('Optimal Policy')
    
    axes[1,0].bar(range(mdp.n_states), mdp.expected_bump_cost, color='salmon')
    axes[1,0].axvline(mdp.capacity-0.5, color='orange', linestyle='--')
    axes[1,0].set_xlabel('Tickets Sold'); axes[1,0].set_ylabel('E[Cost]'); axes[1,0].set_title('Expected Bumping Cost')
    
    terminal = [mdp.terminal_value(s) for s in range(mdp.n_states)]
    axes[1,1].bar(range(mdp.n_states), terminal, color='steelblue')
    axes[1,1].axvline(mdp.capacity-0.5, color='orange', linestyle='--')
    axes[1,1].set_xlabel('Tickets Sold'); axes[1,1].set_ylabel('Value'); axes[1,1].set_title('Terminal Values')
    
    plt.tight_layout()
    plt.savefig("results/figures/mdp_results.png", dpi=150)
    plt.close()
    
    print("\nFiles saved to results/")
    print("Done!")
