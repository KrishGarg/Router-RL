# RouterRL: Reinforcement Learning-based LLM Router

RouterRL is a budget-aware reinforcement learning system that learns how to allocate a fixed token budget (global limit of 10,000 tokens) across a stream of incoming queries by routing them to different LLMs (from cheap/small to expensive/strong models) or choosing to reject them when resources are critically low.

Our objective is to maximize overall system performance by balancing two competing goals:
1. **Average response quality** (maximizing user satisfaction).
2. **Number of queries successfully answered** (maximizing system capacity).

---

## 🚀 Performance Benchmarks (Global Stream Mode)

Here are the results evaluated over 200 stream episodes (representing 20,000 queries) under a global token budget of **10,000 tokens** per 100-query batch:

| Policy / Agent | Type | Avg Quality | Queries Answered | Survival Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Always_Cheap** | Heuristic | 2.880 | 55.5% | 0.5% |
| **Always_Strong** | Heuristic | 2.378 | 24.8% | 0.0% |
| **Threshold_Heuristic** | Heuristic | 2.590 | 28.7% | 0.0% |
| **ppo_router_beta_0.01 (RL)** | **RL Agent** | **3.268** | **67.6%** | **5.5%** |

### Column Explanations:
* **Avg Quality**: The average LLM-as-a-judge score (0 to 10) awarded to responses across all 100 queries in the stream. Queries that were unanswered (due to early budget depletion) or rejected count as `0.0`.
* **Queries Answered**: The fraction of the 100 queries in the stream that the policy successfully routed to an LLM before running out of budget.
* **Survival Rate**: The percentage of 100-query episodes that completed without running out of tokens (budget depletion).

---

## 1. Project Methodology

We model the routing task as a **Markov Decision Process (MDP)** implemented in a custom Gymnasium environment:

### State Space, $S$ (28-dimensional vector)
* **Normalized Remaining Budget** (1 dimension): $\frac{\text{Remaining Budget}}{\text{Max Budget}}$.
* **Difficulty / Routing Gap** (1 dimension): The performance difference between the strongest and weakest models ($`\text{Score}_{\text{strong}} - \text{Score}_{\text{weak}}`$).
* **Linguistic & Contextual Features** (26 dimensions): Query length, math/code/URL flags, prior context size, conversational dependency flags, and lexical overlaps.

### Action Space, $A$ (Discrete space of size 5)
* `0`: Route to **Qwen3-0.6B** (Cheapest, lowest quality)
* `1`: Route to **Ministral-3-8B** (Cheap-Medium)
* `2`: Route to **Qwen3-30B-A3B** (Medium-Expensive)
* `3`: Route to **Qwen3-30B-A3B-Instruct** (Most expensive, highest quality)
* `4`: **Reject / Drop query** (Zero token cost, zero quality score)

### Reward Function, $`R(s_t, a_t)`$
$$\text{Reward}_t = \text{Score}_t - (\beta \times \text{Cost}_t)$$
Where:
* $\text{Score}_t$ is the LLM-as-a-judge score (0-10) for the selected model.
* $\text{Cost}_t$ is the response length (1 word = 1 token).
* $\beta$ is the cost penalty multiplier (adjustable parameter).
* **Budget Depletion Penalty**: If the model cost exceeds the remaining budget, the request fails. The episode terminates immediately with a heavy penalty of **`-10.0`** and subsequent queries are left unanswered (score `0.0`).

### Episode Structure (Global Stream Mode)
Rather than resetting the budget after a single conversation, **Global Stream Mode** simulates a continuous query queue:
* An episode runs for a batch of **100 consecutive queries** spanning multiple different conversations.
* The 10,000-token budget carries over between conversations.
* The episode terminates early if the budget hits 0.

---

## 2. Setup & Execution Instructions

Follow these instructions to set up the repository from scratch:

### Prerequisites
Make sure you have Python 3.8+ installed.

### Step 1: Clone and Set Up Virtual Environment
Clone the repository, open your terminal in the root directory, and create a virtual environment:
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate
```

### Step 2: Install Dependencies
Install all required packages from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Step 3: Preprocess and Align the Dataset
Download features, model responses, and judge scores from the Hugging Face hub, align them, calculate token lengths, and output a clean table:
```powershell
python preprocess_data.py
```

### Step 4: Run Heuristic Baselines Benchmarking
Evaluate simple rules (Always Cheap, Always Strong, Random, Threshold) to establish comparison benchmarks:
```powershell
python baselines.py
```
*This saves results in `data/baseline_results.json`.*

### Step 5: Train the RL Router Agent
Train a PPO (Proximal Policy Optimization) agent under a balanced cost penalty ($\beta = 0.01$):
```powershell
python train.py --beta 0.01 --timesteps 100000
```
*(You can also train a quality-focused agent with `--beta 0.0` or a frugal agent with `--beta 0.05` to explore different trade-offs).*

#### Customizing Training Parameters
`train.py` accepts command-line arguments to completely customize the environment and algorithm:
* **Environment Configurations**:
  * `--max_budget`: The global token pool size (default: `10000`).
  * `--max_steps`: Max query decisions per episode (default: `100`).
  * `--depletion_penalty`: Penalty when budget is exhausted early (default: `-10.0`).
  * `--shuffle`: Shuffle conversations for training exploration (choices: `True`, `False`, default: `True`).
  * `--global_stream`: Toggles the global continuous stream (choices: `True`, `False`, default: `True`).
* **PPO Hyperparameters**:
  * `--lr`: Optimizer learning rate (default: `3e-4`).
  * `--batch_size`: Mini-batch size for gradient updates (default: `64`).
  * `--n_steps`: Number of rollout steps collected before learning (default: `2048`).
  * `--n_epochs`: Training epochs per rollout update (default: `10`).
  * `--gamma`: Discount factor for future rewards (default: `0.99`).
  * `--ent_coef`: Entropy coefficient to control exploration randomness (default: `0.01`).

*Example running with custom discount factor and low depletion penalty:*
```powershell
python train.py --beta 0.0 --depletion_penalty -2.0 --gamma 0.95 --timesteps 150000
```

### Step 6: Evaluate and Plot the Pareto Frontier
Load your trained RL models, benchmark them against the heuristics, print a comparative table, and save a Pareto frontier chart:
```powershell
python evaluate.py
```
The comparison chart will be saved at `data/quality_cost_tradeoff.png`.

---

## 3. Noteworthy Observations & Limitations

* **The Quantiy vs. Quality Trade-off**: Under a strict 10k token budget, `Always_Strong` yields a lower average quality score (**2.38**) than `Always_Cheap` (**2.88**). This happens because `Always_Strong` depletes the budget after only 24 queries, leaving the remaining 76 queries completely unanswered. 
* **The RL Advantage**: By learning when to spend tokens on the strong model (such as on coding/math tasks or high-difficulty queries) and when to conserve budget on simpler turns, the PPO agent trained with $\beta=0.01$ outperforms both heuristics—answering **67.6% of queries** and achieving an average quality score of **3.27**.
* **Deployment Limitation**: The environment includes the pre-extracted `difficulty` feature ($`Score_{strong} - Score_{weak}`$) in the state vector. In a real-time production system, this score gap is not known before calling the models. In a real deployment, we would either:
  1. Remove `difficulty` from the observation space and train the agent purely on the 26 raw features (which are instantly readable from the query text).
  2. Estimate the difficulty beforehand using a lightweight auxiliary model (like a 100M classifier).
