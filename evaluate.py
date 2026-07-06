import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from router_env import RouterGLEnv

def evaluate_rl_agent(model_path, env, num_episodes):
    """
    Evaluates a trained PPO model on the environment in Global Stream Mode.
    Pads results with 0.0 if the agent runs out of budget early.
    """
    print(f"Loading and evaluating RL agent: {os.path.basename(model_path)}...")
    model = PPO.load(model_path)
    
    total_scores = []
    total_costs = []
    queries_attempted = 0
    queries_answered = 0
    depletion_count = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        
        ep_scores = []
        ep_costs = []
        
        while not terminated:
            # Predict action from RL policy
            action, _ = model.predict(obs, deterministic=True)
            
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
                
            obs = next_obs
            
        # Pad remaining queries if budget depleted early
        if len(ep_scores) < env.max_steps:
            remaining_steps = env.max_steps - len(ep_scores)
            ep_scores.extend([0.0] * remaining_steps)
            ep_costs.extend([0.0] * remaining_steps)
            queries_attempted += remaining_steps
            
        total_scores.extend(ep_scores)
        total_costs.extend(ep_costs)
        
    avg_quality = np.mean(total_scores) if total_scores else 0.0
    answered_ratio = queries_answered / queries_attempted if queries_attempted else 0.0
    avg_cost = np.mean(total_costs) if total_costs else 0.0
    survival_rate = 1.0 - (depletion_count / num_episodes)
    
    return {
        "avg_quality": float(avg_quality),
        "avg_cost": float(avg_cost),
        "queries_answered_ratio": float(answered_ratio),
        "survival_rate": float(survival_rate)
    }

def main():
    print("=== RouterRL Evaluation and Comparison ===")
    
    # 1. Load Heuristic Baselines
    baseline_path = "data/baseline_results.json"
    if not os.path.exists(baseline_path):
        print(f"Warning: Baseline results not found at {baseline_path}. Please run baselines.py first!")
        baselines = {}
    else:
        with open(baseline_path, "r") as f:
            baselines = json.load(f)
            
    # Find and evaluate trained RL models
    rl_models = [f for f in glob.glob(os.path.join("models", "ppo_router_beta_*")) if os.path.isfile(f) and not f.endswith(".json")]
    
    # Instantiate the environment in evaluation mode (deterministic, global stream)
    env = RouterGLEnv(
        data_path="data/aligned_data.csv",
        max_budget=10000,
        shuffle=False,
        global_stream=True,
        max_steps=100
    )
    
    # Evaluate over 200 episodes (same as baselines)
    num_episodes = 200
    
    results = {}
    
    # Add baselines to final results
    for name, metrics in baselines.items():
        results[name] = {
            "avg_quality": metrics["avg_quality"],
            "avg_cost": metrics["avg_cost"],
            "queries_answered_ratio": metrics["queries_answered_ratio"],
            "survival_rate": metrics["survival_rate"],
            "type": "Heuristic"
        }
        
    # Evaluate RL models
    for model_path in rl_models:
        model_name = os.path.basename(model_path)
        if model_name.endswith(".zip"):
            model_name = model_name[:-4]
        rl_metrics = evaluate_rl_agent(model_path, env, num_episodes)
        results[model_name] = {
            "avg_quality": rl_metrics["avg_quality"],
            "avg_cost": rl_metrics["avg_cost"],
            "queries_answered_ratio": rl_metrics["queries_answered_ratio"],
            "survival_rate": rl_metrics["survival_rate"],
            "type": "RL"
        }
        
    if not results:
        print("No results found to print.")
        return
        
    # Print Comparison Table
    df = pd.DataFrame.from_dict(results, orient="index")
    print("\n" + "="*80)
    print("                      ROUTING PERFORMANCE COMPARISON")
    print("="*80)
    print(df[["type", "avg_quality", "avg_cost", "queries_answered_ratio", "survival_rate"]])
    print("="*80)
    
    # Generate Pareto Plot
    print("\nGenerating Quality vs Cost Trade-off plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot Heuristics
    heuristics_df = df[df["type"] == "Heuristic"]
    if not heuristics_df.empty:
        plt.scatter(heuristics_df["avg_cost"], heuristics_df["avg_quality"], color="red", marker="X", s=150, label="Heuristics")
        for idx, row in heuristics_df.iterrows():
            plt.text(row["avg_cost"] + 2, row["avg_quality"], idx, fontsize=9, fontweight="bold", color="darkred")
            
    # Plot RL Agents
    rl_df = df[df["type"] == "RL"]
    if not rl_df.empty:
        # Sort by cost to draw a neat trade-off curve
        rl_df = rl_df.sort_values(by="avg_cost")
        plt.plot(rl_df["avg_cost"], rl_df["avg_quality"], color="blue", linestyle="--", alpha=0.6)
        plt.scatter(rl_df["avg_cost"], rl_df["avg_quality"], color="blue", marker="o", s=120, label="RL Policies")
        for idx, row in rl_df.iterrows():
            # Clean up display name (e.g. ppo_router_beta_0.01 -> RL (beta=0.01))
            display_name = idx.replace("ppo_router_beta_", "RL (beta=") + ")"
            plt.text(row["avg_cost"] + 2, row["avg_quality"] - 0.05, display_name, fontsize=9, color="blue")
            
    plt.title("Quality vs. Token Cost Trade-off (Pareto Frontier)", fontsize=14, fontweight="bold")
    plt.xlabel("Average Token Cost / Step (Lower is better)", fontsize=12)
    plt.ylabel("Average Response Quality (0-10, Higher is better)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")
    
    # Save plot
    plot_path = "data/quality_cost_tradeoff.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    
if __name__ == "__main__":
    main()
