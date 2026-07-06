import os
import json
import numpy as np
import pandas as pd
from router_env import RouterGLEnv

# Standard baselines to benchmark
def run_baseline(env, policy_fn, num_episodes):
    """
    Runs a policy function on the environment for a fixed number of episodes and records statistics.
    If the episode terminates early due to budget depletion, we pad the remaining steps with 
    0 score and 0 cost to represent unanswered queries in the stream.
    """
    total_scores = []
    total_costs = []
    queries_attempted = 0
    queries_answered = 0
    depletion_count = 0
    rewards_per_beta = {b: [] for b in env.betas}
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        
        ep_scores = []
        ep_costs = []
        
        while not terminated:
            # Get action from policy
            action = policy_fn(obs, info, env)
            
            # Step in environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Record metrics
            queries_attempted += 1
            if not step_info["budget_depleted"]:
                queries_answered += 1
                ep_scores.append(step_info["score"])
                ep_costs.append(step_info["cost"])
            else:
                depletion_count += 1
                ep_scores.append(0.0)
                ep_costs.append(0.0)
                
            # Track rewards for all betas
            for b, r_val in step_info["rewards_all_betas"].items():
                rewards_per_beta[b].append(r_val)
                
            obs = next_obs
            
        # Pad remaining queries in the batch if depleted early
        if len(ep_scores) < env.max_steps:
            remaining_steps = env.max_steps - len(ep_scores)
            ep_scores.extend([0.0] * remaining_steps)
            ep_costs.extend([0.0] * remaining_steps)
            queries_attempted += remaining_steps
            
            # Pad rewards list with depletion penalty for all betas
            for b in env.betas:
                rewards_per_beta[b].extend([env.depletion_penalty] * remaining_steps)
            
        total_scores.extend(ep_scores)
        total_costs.extend(ep_costs)
        
    avg_quality = np.mean(total_scores) if total_scores else 0.0
    answered_ratio = queries_answered / queries_attempted if queries_attempted else 0.0
    avg_cost = np.mean(total_costs) if total_costs else 0.0
    survival_rate = 1.0 - (depletion_count / num_episodes)
    
    beta_rewards = {str(b): float(np.mean(r_list)) for b, r_list in rewards_per_beta.items()}
    
    return {
        "avg_quality": float(avg_quality),
        "avg_cost": float(avg_cost),
        "queries_answered_ratio": float(answered_ratio),
        "survival_rate": float(survival_rate),
        "beta_rewards": beta_rewards
    }

def main():
    print("=== RouterRL Heuristic Benchmarking (GLOBAL STREAM MODE) ===")
    
    # Configure betas for checking trade-offs
    betas = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    # Instantiate the environment in evaluation mode with global 10k budget
    env = RouterGLEnv(
        data_path="data/aligned_data.csv",
        max_budget=10000,
        betas=betas,
        active_beta_idx=3,  # beta = 0.01
        shuffle=False,      # Loop through conversations deterministically
        global_stream=True, # Activate global budget stream
        max_steps=100       # Target 100 queries per episode
    )
    
    # Run evaluation
    num_episodes = 200 # 200 episodes is enough to cover 20k queries (200 * 100 = 20k)
    print(f"Running evaluation over {num_episodes} episodes ({num_episodes * env.max_steps} total queries)...")
    
    # Define Policy Functions
    # 0: Qwen3-0.6B, 1: Ministral-8B, 2: Qwen3-30B-A3B, 3: Qwen3-30B-Instruct, 4: Reject
    
    # Policy 1: Always Cheap (Action 0)
    def always_cheap(obs, info, env_inst):
        return 0
        
    # Policy 2: Always Strong (Action 3)
    def always_strong(obs, info, env_inst):
        return 3
        
    # Policy 3: Random Routing (Action 0 to 3)
    def random_routing(obs, info, env_inst):
        return np.random.randint(0, 4)
        
    # Policy 4: Threshold Heuristic based on Difficulty Feature
    def threshold_heuristic(obs, info, env_inst):
        difficulty = obs[1]
        
        # Threshold logic:
        if difficulty < 3.0:
            return 0  # Qwen3-0.6B (cheapest)
        elif difficulty < 6.0:
            return 1  # Ministral-8B
        elif difficulty < 8.0:
            return 2  # Qwen3-30B-A3B
        else:
            return 3  # Qwen3-30B-Instruct (strongest)

    policies = {
        "Always_Cheap": always_cheap,
        "Always_Strong": always_strong,
        "Random": random_routing,
        "Threshold_Heuristic": threshold_heuristic
    }
    
    results = {}
    for name, policy_fn in policies.items():
        print(f"\nEvaluating policy: {name}...")
        results[name] = run_baseline(env, policy_fn, num_episodes)
        
        r = results[name]
        print(f"  Avg Quality over stream: {r['avg_quality']:.3f} / 10")
        print(f"  Avg Token Cost / step:   {r['avg_cost']:.1f}")
        print(f"  Queries Answered %:      {r['queries_answered_ratio']*100:.1f}%")
        print(f"  Episode Survival %:      {r['survival_rate']*100:.1f}%")
        
    # Save results to disk
    os.makedirs("data", exist_ok=True)
    with open("data/baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nBaseline results saved to data/baseline_results.json")

if __name__ == "__main__":
    main()
