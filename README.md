# Multi-Agent Cooperative Pathfinding with PBRS

This project implements and compares multiple **reinforcement learning** approaches for a **multi-agent cooperative pathfinding** task, with a focus on the impact of **Potential-Based Reward Shaping (PBRS)**.

We compare:

- **Baseline**: Tabular Q-learning
- **Centralized DQN**
- **Hybrid**: Decentralized Actor–Critic with a Centralized DQN Critic  
- Each **with and without PBRS**

The main result: **PBRS dramatically accelerates learning**, especially for the centralized DQN.

---

## 📌 Problem Overview

- Environment: grid-world cooperative pathfinding
- Objective: multiple agents must reach their goals **without collisions**
- Setting: **cooperative** multi-agent RL (agents share a common reward / objective)

> 🔧 Adjust this section to match your exact environment (grid size, number of agents, etc.)

Example (replace with your actual setup):

- Grid size: `N x N` (e.g., 8×8 or 10×10)
- Number of agents: e.g., 3
- Obstacles: static or randomly placed per episode
- Episode ends when:
  - All agents reach their goals, or
  - A maximum number of steps is reached
- Success metric: all agents reach their goals without collisions

---

## 🧠 Methods

### 1. Baseline: Tabular Q-Learning

- Independent / joint state representation (depending on your design)
- Standard Q-learning update:
  
  \[
  Q(s, a) \leftarrow Q(s, a) + \alpha \big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\big]
  \]

Used as a **simple baseline** to illustrate sample inefficiency and difficulty without shaping.

---

### 2. Centralized DQN

- Single DQN that sees a **centralized representation** of the environment:
  - Full grid / all agents’ states
- Outputs joint actions or per-agent actions (depending on implementation)
- Uses:
  - Replay buffer
  - ε-greedy exploration
  - Target network

This model performs best when combined with PBRS.

---

### 3. Hybrid: Decentralized Actor–Critic + Centralized DQN Critic

- **Actors**:
  - One policy per agent (decentralized execution)
  - Each actor conditions on its **local observations**
- **Critic**:
  - Centralized DQN-based critic (uses global state / joint observations)
  - Provides value estimates / targets for updating the actors

This hybrid design offers a middle ground: more scalable than a fully centralized policy, but more informed than purely decentralized Q-learning.

> ✏️ You can add a diagram (PNG) showing:
> - Global state going into centralized critic
> - Local observations going into decentralized actors

---

## 🎯 Potential-Based Reward Shaping (PBRS)

We use **potential-based reward shaping** to accelerate learning without changing the optimal policy.

General shaping function:

\[
F(s, s') = \gamma \, \Phi(s') - \Phi(s)
\]

where:

- \(\Phi(s)\) is a **potential function** over states,
- \(\gamma\) is the discount factor.

Example potential function (adjust to your design):

- \(\Phi(s)\) = negative sum of Manhattan distances from each agent to its goal  
  (so higher potential = agents closer to their goals)

Shaped reward:

\[
r' = r + \text{potential\_coef} \cdot F(s, s')
\]

In this project, we tested configurations such as:

- `potential_coef = 0.35` (PBRS ON)
- `potential_coef = 0.0` (PBRS OFF)

---

## 📊 Results

We measure **how many episodes it takes** for each method to reach a target **success rate = 0.80**.

### PBRS ON (`potential_coef = 0.35`)

| Method                                        | Episodes to 0.80 success |
|----------------------------------------------|---------------------------|
| Baseline Q-learning                          | 1376.0                    |
| Centralized DQN                              | **109.6**                 |
| Hybrid (Decentralized Actor–Critic + DQN)    | 392.8                     |

### PBRS OFF (`potential_coef = 0.0`)

| Method                                        | Episodes to 0.80 success |
|----------------------------------------------|---------------------------|
| Baseline Q-learning                          | ∞ (did not converge)      |
| Centralized DQN                              | 2514.8                    |
| Hybrid (Decentralized Actor–Critic + DQN)    | 1543.6                    |

### Key Takeaways

- **PBRS significantly improves learning efficiency**
  - With PBRS ON, both **centralized DQN** and **hybrid** models reach the success threshold **much faster**.
- **Centralized DQN + PBRS is the strongest setup**
  - Converges in ~**109.6 episodes**, compared to **2514.8** without PBRS.
- **Hybrid model is a solid middle ground**
  - Faster than the baseline, with decentralized execution and some centralized coordination via the critic.
- **Baseline without PBRS fails to converge**
  - Illustrates how hard the task is under sparse or unshaped rewards.

> Note: Replace these numbers with updated results if you rerun with more seeds, different hyperparameters, etc.

---

## 📈 Plots & Visualizations

You can include (recommended):

- **Learning curves**: success rate vs episodes for each method
- **PBRS ON vs OFF** comparisons
- **GIF demos** of agents moving in the grid-world

Example (once you generate them):

```text
assets/
  curves_pbrs_on_off.png
  dqn_vs_hybrid.png
  demo_pbrs_on.gif
  demo_pbrs_off.gif
