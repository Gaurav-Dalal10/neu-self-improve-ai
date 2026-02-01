# Week 01 — Airline Overbooking via Policy Iteration

## 1. Problem Description and Motivation

We consider an airline revenue management problem where a carrier must decide how many tickets to sell for a flight with fixed seat capacity. Since some passengers who purchase tickets do not show up for the flight (no-shows), airlines routinely sell more tickets than available seats — a practice known as **overbooking**.

If too many passengers show up, some must be denied boarding ("bumped") and compensated. If too few tickets are sold, the flight departs with empty seats and lost revenue.

The operator faces a trade-off between:
- **generating revenue** by selling additional tickets, and
- **avoiding bumping costs** by limiting sales when overbooking risk is high.

The goal is to determine an optimal booking policy that maximizes expected profit.

### Modeling as a Markov Decision Process

This decision-making problem is naturally modeled as a Markov Decision Process (MDP) because:
- the system state (tickets sold) evolves sequentially over time,
- the controller makes sequential decisions (accept or reject each booking request),
- the future state depends only on the current state and action,
- and the objective is to maximize expected profit.

### Data Source

Parameters are calibrated using real data from the **Bureau of Transportation Statistics (BTS)**:
- Dataset: Involuntary Denied Boarding Reports (2018–2020)
- URL: https://data.transportation.gov/d/xyfb-hgtv
- Key statistic: Average compensation per bumped passenger = **$488.09**

### Scope and Assumptions

The following assumptions are made:
- Booking requests arrive sequentially (one at a time).
- Each passenger shows up independently with probability $p = 0.90$.
- The number of passengers who show up follows a Binomial distribution.
- The decision is binary: accept or reject each booking request.
- The policy is stationary and deterministic.

### Objective of the Assignment

The objectives of this assignment are to:
1. Formulate the airline overbooking problem formally as an MDP.
2. Apply policy iteration to compute an optimal booking policy.
3. Analyze the structure and behavior of the learned policy.
4. Demonstrate how the optimal overbooking threshold emerges from the model.

---

## 2. Markov Decision Process (MDP) Formulation

We model the overbooking problem as a Markov Decision Process (MDP), defined by the 5-tuple:

$$(\mathcal{S}, \mathcal{A}, P, R, \gamma)$$

where each component is formally defined below.

### 2.1 State Space

The state represents the number of tickets already sold.

$$\mathcal{S} = \{0, 1, 2, \ldots, M\}$$

A state $s_t = n$ denotes that $n$ tickets have been sold at decision epoch $t$.

### 2.2 Action Space

The action corresponds to whether to accept or reject a booking request.

$$\mathcal{A} = \{0: \text{Reject}, 1: \text{Accept}\}$$

### 2.3 Transition Probabilities

Transitions are deterministic based on the action taken:

$$P(s' \mid s, a) = \begin{cases} 1 & \text{if } a = 1 \text{ and } s' = s + 1 \\ 1 & \text{if } a = 0 \text{ and } s' = s \\ 0 & \text{otherwise} \end{cases}$$

### 2.4 Reward Function

Intermediate reward for accepting a booking:

$$R(s, a) = \begin{cases} r & \text{if } a = 1 \\ 0 & \text{if } a = 0 \end{cases}$$

Terminal reward after all booking decisions are made:

$$R_T(n) = n \cdot r - c_b \cdot \mathbb{E}[\max(0, X - C)]$$

where:
- $r$ = ticket price
- $c_b$ = bumping cost per passenger
- $C$ = aircraft capacity
- $X \sim \text{Binomial}(n, p)$ = number of passengers who show up

### 2.5 Parameters (from Real Data)

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Aircraft capacity | $C$ | 10 | — |
| Maximum tickets | $M$ | 15 | — |
| Ticket price | $r$ | $300 | Average domestic fare |
| Bumping cost | $c_b$ | $500 | BTS: $488.09 |
| Show-up probability | $p$ | 0.90 | Industry 10% no-show rate |

### 2.6 Objective

The objective is to find an optimal stationary policy $\pi^*$ that maximizes expected profit:

$$\pi^* = \arg\max_{\pi} \mathbb{E}\left[ \sum_{t=0}^{M} R(s_t, \pi(s_t)) + R_T(s_M) \right]$$

---

## 3. Solution Method: Policy Iteration

We solve the MDP using policy iteration, which alternates between policy evaluation and policy improvement.

### 3.1 Policy Evaluation

For a fixed policy $\pi$, the value function satisfies the Bellman expectation equation:

$$V^\pi(s) = R(s, \pi(s)) + \sum_{s' \in \mathcal{S}} P(s' \mid s, \pi(s)) V^\pi(s')$$

This equation is solved iteratively until convergence.

### 3.2 Policy Improvement

The policy is updated greedily using the current value function:

$$\pi_{\text{new}}(s) = \arg\max_{a \in \mathcal{A}} \left[ R(s, a) + \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \right]$$

Policy evaluation and improvement are repeated until the policy stabilizes.

---

## 4. Results and Findings

Policy iteration converged after **2 iterations**.

The learned optimal booking policy exhibits a **threshold structure**:

$$\pi^*(s) = \begin{cases} \text{Accept} & s < 15 \\ \text{Reject} & s = 15 \end{cases}$$

This indicates that the controller accepts all booking requests until 15 tickets are sold (50% overbooking for 10-seat capacity), then rejects further requests.

### Key Metrics

| Metric | Value |
|--------|-------|
| Optimal threshold | 15 tickets |
| Overbooking rate | 50% |
| Expected profit | $7,248.70 |
| Profit without overbooking | $3,000.00 |
| Improvement | +141.6% |

### Value Function Property

The value function decreases as tickets sold increases beyond capacity:

$$V(s+1) \leq V(s) \quad \text{for } s \geq C$$

This reflects increasing expected bumping costs at higher overbooking levels.

### Note on Model Limitations

The 50% overbooking rate is mathematically optimal given the parameters, but real airlines typically overbook only 5–15%. This is because the model captures only direct compensation costs. Real-world factors not modeled include:
- Reputation damage
- Customer loyalty loss
- Social media backlash
- Operational disruptions

---

## 5. Example Execution Output

The following output is produced when running the experiment (`python run_experiment.py`):

```
============================================================
AIRLINE OVERBOOKING MDP - POLICY ITERATION
============================================================

Data Source: BTS Denied Boarding Data
https://data.transportation.gov/d/xyfb-hgtv

Parameters: C=10, M=15, r=$300, c_b=$500, p=0.90

Iteration 1: V(0) = $7248.70
Iteration 2: V(0) = $7248.70

Converged in 2 iterations!

============================================================
RESULTS
============================================================
Optimal threshold:     15 tickets
Overbooking rate:      50%
Expected profit:       $7,248.70
No-overbook profit:    $3,000.00
Improvement:           +141.6%

Sensitivity Analysis:
  p=0.70: threshold=15, profit=$8,510.40
  p=0.80: threshold=15, profit=$7,957.87
  p=0.85: threshold=15, profit=$7,614.44
  p=0.90: threshold=15, profit=$7,248.70
  p=0.95: threshold=15, profit=$6,874.97

Files saved to results/
Done!
```



## References

1. Bureau of Transportation Statistics — Involuntary Denied Boarding Data  
   https://data.transportation.gov/d/xyfb-hgtv

2. US DOT 14 CFR Part 250 — Oversales Regulations
