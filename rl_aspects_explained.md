# Reinforcement Learning Theory & Application in RouterRL

This guide explains the Reinforcement Learning (RL) concepts, formulations, and hyperparameters implemented in **RouterRL** to bridge the gap between academic theory and practical code.

---

## 1. The MDP (Markov Decision Process) Formulation

An RL agent learns by interacting with an environment. This interaction is mathematically formalized as a **Markov Decision Process (MDP)**, which consists of four core elements: $\{S, A, P, R\}$.

```
    +-----------------------------------------------+
    |                                               |
    |                   Environment                 |
    |                                               |
    +----|--------------------------------------^---+
         | State (s_t)                          | Action (a_t)
         | Reward (r_t)                         |
    +----v--------------------------------------|---+
    |                                               |
    |                      Agent                    |
    |                                               |
    +-----------------------------------------------+
```

### State Space ($S$)
The state $s_t$ represents the information the agent uses to make a decision at step $t$. Our state is a **28-dimensional vector**:
* **1D Normalized Budget**: $\frac{\text{Remaining Tokens}}{10000}$. Tells the agent if it is flush with tokens or close to early depletion.
* **27D Query & Context Features**: Difficulty, length, syntactic patterns (math, code, urls), and context history. 
* *Markov Assumption*: We assume this 28D vector captures all necessary historical information needed to make the optimal routing decision at step $t$.

### Action Space ($A$)
The action space is the set of possible choices the agent can make. We use a **Discrete Action Space of size 5**:
* Actions $0-3$ represent candidate LLMs (Qwen3-0.6B to Qwen3-30B-Instruct).
* Action $4$ represents a **Reject** (or Drop) action, which consumes $0$ tokens but outputs $0$ quality score.

### Transition Dynamics ($P$)
The transition probability $P(s_{t+1} \mid s_t, a_t)$ defines the physics of our world—how the environment changes after the agent acts:
1. **Budget Decrement**: If a model is queried, the remaining token budget is reduced by the output length of that model's response.
2. **Turn Advancement**: The environment steps forward to the next turn in the conversation.
3. **Global Stream Transitions**: In **Global Stream Mode**, if a conversation ends but the budget remains, the environment transitions to the first turn of a *new conversation*, preserving the remaining budget.

---

## 2. The Reward Function & the Role of Beta ($\beta$)

The reward $R(s, a)$ is the numerical feedback signal the agent tries to maximize. In RouterRL, the reward at step $t$ is:

$$R_t = \text{Score}_t - (\beta \times \text{Cost}_t)$$

### The Trade-off Weight ($\beta$)
$\beta$ is a hyperparameter that represents the **cost penalty coefficient** (token price):
* **$\beta = 0.0$ (Quality-Only)**: The agent has no penalty for spending tokens. It will route queries aggressively to the premium model to secure a $10/10$ judge score.
* **$\beta = 0.1$ (Cost-Frugal)**: The cost penalty is very high. Spending 400 tokens costs $40.0$ reward points. The agent will prefer cheap models or choose to Reject queries to avoid the massive cost penalty.
* **$\beta = 0.01$ (Balanced)**: The agent balances quality and cost, selecting premium models only when the expected boost in quality is larger than the cost penalty.

### The Survival Constraint (`depletion_penalty`)
What stops a $\beta = 0.0$ agent from simply routing every query to the most expensive model?
* **Budget Depletion**: If the cost of the chosen action exceeds the remaining budget, the episode terminates early.
* **Depletion Penalty**: The agent receives a massive penalty of **`-10.0`** (default) for every remaining query in the 100-step batch.
* Even with $\beta=0.0$, the agent behaves conservatively because it knows that dying early triggers a massive sequence of negative penalties, which ruins its total cumulative reward.

---

## 3. PPO (Proximal Policy Optimization) Training Mechanics

We use **PPO**, a state-of-the-art Policy Gradient algorithm. PPO trains two neural networks simultaneously:

1. **The Actor (Policy Network $\pi_\theta(a \mid s)$)**: Maps the state vector to a probability distribution over the 5 actions. This network makes the routing decisions.
2. **The Critic (Value Network $V_\phi(s)$)**: Predicts the expected cumulative future reward from the current state. This network acts as a baseline to evaluate the Actor's decisions.

```
                         +-----------------+
                         |   State Vector  |
                         +--------|--------+
                                  |
                 +----------------+----------------+
                 |                                 |
        +--------v--------+               +--------v--------+
        |   Actor Net     |               |   Critic Net    |
        |   (Policy)      |               |   (Value)       |
        +--------|--------+               +--------|--------+
                 |                                 |
                 v                                 v
        Action Probabilities              Predicted Future Reward
       [0.1, 0.2, 0.4, 0.2, 0.1]                 (e.g., +4.2)
```

### The Rollout Update Cycle
PPO does not update the networks after every single step. Instead, it works in cycles:
1. **Rollout Collection**: The agent interacts with the environment for **`n_steps`** (default: `2048`) steps, collecting experiences.
2. **Advantage Calculation**: It calculates the **Advantage** $A_t$ for each action:
   $$A_t = \text{Actual Reward} - \text{Expected Reward predicted by Critic}$$
   * Positive advantage means the action performed better than expected; negative means worse.
3. **Optimization Epochs**: It updates the networks over **`n_epochs`** (default: `10`) epochs using mini-batches of size **`batch_size`** (default: `64`).
4. **PPO Clipping**: PPO constraints the update step using a clipping function (e.g. `0.2`). This limits how much the policy can change in one go, preventing destructive policy updates.

---

## 4. Hyperparameters Demystified

When customizing `train.py`, you can modify these key hyperparameters:

### `gamma` ($\gamma$ - Discount Factor)
* **What it is**: A factor between `0` and `1` determining how much the agent values immediate rewards vs. future rewards.
* **Effect**: 
  * $\gamma = 0.99$ (default): The agent is **farsighted**. It cares about the future turns and will save budget early to survive long-term.
  * $\gamma = 0.5$: The agent is **myopic** (short-sighted). It only cares about the current turn and turns in the immediate future, which leads to early budget depletion.

### `ent_coef` (Entropy Coefficient)
* **What it is**: A penalty that rewards the policy for being random/unpredictable.
* **Effect**: Promotes **exploration** early in training. A value of `0.01` ensures the agent continues to try different routing choices, preventing it from getting stuck in a local minimum (such as always choosing cheap models).

### `lr` (Learning Rate)
* **What it is**: The step size for gradient descent updates.
* **Effect**: A standard value of `3e-4` is balanced. If too high, training will be unstable; if too low, training will be extremely slow.

---

## 5. Reading TensorBoard Curves

When monitoring training in TensorBoard (`http://localhost:6006/`), check these core charts:

### `rollout/ep_rew_mean` (Episode Reward Mean)
* **Concept**: The average score of your agent per episode.
* **Pattern**: It should start low (highly negative due to early depletion penalties) and curve **steadily upwards** before leveling off.

### `rollout/ep_len_mean` (Episode Length Mean)
* **Concept**: How many queries the agent answered before termination (max: 100).
* **Pattern**: It should rise over time. In early epochs, the agent runs out of budget quickly (length $\approx 25$). As it learns to budget, this number should rise (e.g., to $50+$ or $70+$).

### `train/entropy_loss`
* **Concept**: The randomness of your policy.
* **Pattern**: This curve should start highly negative (high randomness/exploration) and **move upwards towards zero** (confident decision-making/exploitation).

### `train/value_loss`
* **Concept**: The error in the Critic network's predictions.
* **Pattern**: Usually spikes early, then **declines and flattens out**, indicating the critic has successfully learned to predict the expected value of different budgets.
