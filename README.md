# Multi-Agent Cooperative Pathfinding with Potential-Based Reward Shaping (PBRS)

This repository implements a **multi-agent reinforcement learning** experiment for **cooperative pathfinding** on a grid. It compares:

- **Baseline**: Independent tabular Q-Learning  
- **Centralized joint DQN**  
- **Hybrid**: Decentralized policy-gradient actors + centralized DQN critic  

Each method is evaluated **with and without Potential-Based Reward Shaping (PBRS)**. The code is set up for **multi-seed experiments**, CSV summaries, and convergence analysis.

---

## 🧩 Problem & Environment

The environment is a small **grid-world** where multiple agents must **cooperate** to reach their respective goals.

Key details (as implemented in `GridWorldMA`):

- Grid size: **6 × 6**
- Number of agents: **2**
- Obstacles:
  - Random interior obstacles with `obstacle_fraction = 0.08`
- Episode limit:
  - `max_steps = 70` steps per episode
- Actions per agent:
  - Stay, Up, Down, Left, Right (5 discrete actions)
- Rewards:
  - Step cost: `-0.05 * n_agents`
  - Bump into wall/obstacle: `-0.05` per bump
  - Collision between agents: `-1.0` per collision
  - Reaching goal: `+10.0` (once per agent)
- Episodes end when:
  - All agents reach their goals, or
  - `max_steps` is reached

The state encodes:

- The grid (flattened)
- Agent positions (normalized)
- Goal positions (normalized)

---

## 🧠 Algorithms

All three algorithms assume **2 agents** and operate on the same environment.

### 1. Baseline: Independent Tabular Q-Learning

Function: `baseline_q_learning(...)`

- One tabular Q-table per agent.
- Each agent acts ε-greedily using its own Q-table.
- State encoding for each agent includes:
  - Own position
  - Other agent’s position
  - Own goal

Used as a **simple baseline** to show how hard the task is without function approximation and with sparse-ish rewards.

---

### 2. Centralized Joint DQN

Function: `train_centralized_dqn(...)`

- A single DQN sees the **centralized state**:
  - Full grid + both agents’ positions + both goals.
- Outputs Q-values over **joint actions**:
  - For 2 agents with 5 actions each → 25 joint actions.
- Uses:
  - Replay buffer
  - Target network with soft updates (τ = 0.01)
  - ε-greedy exploration over joint actions

This is a strong fully centralized baseline.

---

### 3. Hybrid: Decentralized PG Actors + Centralized DQN Critic

Function: `train_hybrid(...)`

- **Actors**:
  - One policy (actor network) per agent.
  - Each actor sees:
    - Grid
    - Own position
    - Other agent(s)’ positions
    - Own goal
  - Outputs a categorical distribution over actions and samples from it.
- **Critic**:
  - Centralized DQN over the full state and joint actions (same architecture as centralized DQN).
  - Used to compute an **advantage-like signal** for the actors.
- Actors are trained via a policy-gradient style update with:
  - Advantage normalization
  - Entropy regularization (for exploration)

This gives **decentralized execution** with **centralized training**.

---

## ⚙️ Potential-Based Reward Shaping (PBRS)

PBRS is implemented directly inside the environment (`GridWorldMA.step`).

- Potential function:

  - \(\Phi(s) = - \sum_i \text{ManhattanDistance}(\text{agent}_i, \text{goal}_i)\)  
  - i.e., negative total Manhattan distance from all agents to their goals.

- Shaped reward:

  \[
  r' = r + \text{potential\_coef} \cdot \big(\gamma_\Phi \Phi(s') - \Phi(s)\big)
  \]

Where:

- `potential_coef` controls shaping strength (β).
- `gamma_for_potential` (γₚ) is used inside the shaping term.
- This form is **potential-based reward shaping**, which preserves the optimal policy but can greatly speed up learning.

Two configurations are used:

- **PBRS ON**: `potential_coef = 0.35`
- **PBRS OFF**: `potential_coef = 0.0`

---

## 📊 Experimental Setup

Global settings (from the script):

- Seeds: **[3, 7, 11, 19, 23]**
- Maximum episodes per run: `MAX_TOTAL_EPISODES = 3200`
- Moving average window for success: `WINDOW = 40`
- Success threshold: `TARGET_SUCCESS = 0.80`

For each configuration (PBRS ON / PBRS OFF) and each seed:

- The script runs:
  - `baseline_q_learning(...)`
  - `train_centralized_dqn(...)`
  - `train_hybrid(...)`
- For each method and seed, it computes:
  - **Episodes to reach 0.80 success** (based on moving average).
  - **Final success rate** (moving average at the end).

Results across seeds are aggregated using `aggregate_stats(...)`, producing:

- Mean and std of episodes to threshold
- Mean and std of final success rate
- Number of seeds

Summaries are saved as:

- `summary_pbrs_on.csv`
- `summary_pbrs_off.csv`

in the working directory.

---

## 📈 Key Results

Below are the **average episodes (over seeds)** required for each method to reach a **success rate of 0.80** (moving average over 40 episodes).

### PBRS ON (`potential_coef = 0.35`)

| Method                              | Episodes to 0.80 success (mean) |
|------------------------------------|----------------------------------|
| Baseline Q-Learning                | 1376.0                           |
| Centralized DQN                    | **109.6**                        |
| Hybrid (Decentralized + Central)   | 392.8                            |

### PBRS OFF (`potential_coef = 0.0`)

| Method                              | Episodes to 0.80 success (mean) |
|------------------------------------|----------------------------------|
| Baseline Q-Learning                | ∞ (did not converge)             |
| Centralized DQN                    | 2514.8                           |
| Hybrid (Decentralized + Central)   | 1543.6                           |

### Interpretation

- **PBRS massively accelerates learning** for both DQN and Hybrid.
- **Centralized DQN + PBRS** is the most sample-efficient configuration in this setup.
- The **Hybrid** method is:
  - Slower than centralized DQN with PBRS,
  - But significantly faster than baseline and more robust than unshaped training.
- The **baseline without PBRS** fails to reach the target success within the maximum episode limit.

---

## 🧪 What the Script Actually Does

The main entry point is at the bottom of the file:

```python
if __name__ == "__main__":
    # Quick single-run demo (for plots):
    # base, hyb = run_single_experiment(BASE_ENV, seed=3)

    # Full thesis suite (multi-seed + ablations):
    results = run_thesis_suite()
