import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from router_env import RouterGLEnv

def train():
    parser = argparse.ArgumentParser(description="Train a PPO routing agent on RouterGLEnv in Global Stream Mode")
    parser.add_argument("--beta", type=float, default=0.01, help="Cost penalty weighting (trade-off factor)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train the agent")
    parser.add_argument("--max_budget", type=int, default=10000, help="Global token budget per episode")
    parser.add_argument("--max_steps", type=int, default=100, help="Max queries per episode stream")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"\n=== Training RouterRL Agent (GLOBAL STREAM MODE) ===")
    print(f"Beta (cost weighting): {args.beta}")
    print(f"Global Token Budget:   {args.max_budget}")
    print(f"Max Steps / Episode:   {args.max_steps}")
    print(f"Total Timesteps:       {args.timesteps}")
    print(f"Learning Rate:         {args.lr}")
    print(f"Batch Size:            {args.batch_size}")
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup custom list of betas, making sure our training beta is represented
    betas_list = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    if args.beta not in betas_list:
        betas_list.append(args.beta)
    betas_list = sorted(betas_list)
    active_idx = betas_list.index(args.beta)

    # Instantiate training environment in global stream mode
    env = RouterGLEnv(
        data_path="data/aligned_data.csv",
        max_budget=args.max_budget,
        betas=betas_list,
        active_beta_idx=active_idx,
        depletion_penalty=-10.0,
        shuffle=True,          # Shuffle conversations during training for better exploration
        global_stream=True,    # Enable global stream
        max_steps=args.max_steps
    )
    
    # Seed the environment
    env.reset(seed=args.seed)

    # Initialize PPO Model
    # Since our state is a flat vector of size 28, a simple Multi-Layer Perceptron (MlpPolicy) is used
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,            # Discount factor (future turn impacts matter but are constrained by conversation length)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Entropy coefficient to encourage exploration early on
        verbose=1,
        tensorboard_log="./logs/",
        seed=args.seed
    )
    
    # Setup checkpoint saving callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path="./models/checkpoints/",
        name_prefix=f"ppo_router_beta_{args.beta}"
    )

    # Learn/Train
    print("\nStarting PPO agent training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    # Save final model
    model_name = f"ppo_router_beta_{args.beta}"
    model_path = os.path.join("models", model_name)
    model.save(model_path)
    print(f"\nTraining completed! Saved model to: {model_path}.zip")

if __name__ == "__main__":
    train()
