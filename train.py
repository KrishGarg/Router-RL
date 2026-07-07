import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from router_env import RouterGLEnv

def str2bool(v):
    """Helper function to parse booleans correctly from command line args."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train():
    parser = argparse.ArgumentParser(description="Train a PPO routing agent on RouterGLEnv")
    
    # Environment Settings
    parser.add_argument("--data_path", type=str, default="data/aligned_data.csv", help="Path to aligned data CSV")
    parser.add_argument("--max_budget", type=int, default=10000, help="Global token budget per episode")
    parser.add_argument("--max_steps", type=int, default=100, help="Max queries per episode stream")
    parser.add_argument("--depletion_penalty", type=float, default=-10.0, help="Penalty when running out of budget")
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle conversations during training")
    parser.add_argument("--global_stream", type=str2bool, default=True, help="Enable global budget stream mode")
    
    # Training Loop Settings
    parser.add_argument("--beta", type=float, default=0.01, help="Cost penalty weighting (trade-off factor)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train the agent")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # PPO Algorithm Hyperparameters
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of rollout collection steps before update")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (importance of future rewards)")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda coefficient")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clipping range")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient to encourage exploration")
    
    args = parser.parse_args()

    print(f"\n=== Training RouterRL Agent ===")
    print("--- Environment Config ---")
    print(f"  Data Path:          {args.data_path}")
    print(f"  Global Stream Mode: {args.global_stream}")
    print(f"  Token Budget:       {args.max_budget}")
    print(f"  Max Steps/Episode:  {args.max_steps}")
    print(f"  Depletion Penalty:  {args.depletion_penalty}")
    print(f"  Shuffle:            {args.shuffle}")
    
    print("--- Reward Config ---")
    print(f"  Beta (cost weight): {args.beta}")
    
    print("--- PPO Hyperparameters ---")
    print(f"  Total Timesteps:    {args.timesteps}")
    print(f"  Learning Rate:      {args.lr}")
    print(f"  Batch Size:         {args.batch_size}")
    print(f"  Rollout Steps (N):  {args.n_steps}")
    print(f"  Epochs / Update:    {args.n_epochs}")
    print(f"  Discount (Gamma):   {args.gamma}")
    print(f"  Entropy Coef:       {args.ent_coef}")
    print(f"  PPO Clip Range:     {args.clip_range}")
    print(f"  Seed:               {args.seed}")
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Setup custom list of betas, making sure our training beta is represented
    betas_list = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    if args.beta not in betas_list:
        betas_list.append(args.beta)
    betas_list = sorted(betas_list)
    active_idx = betas_list.index(args.beta)

    # Instantiate training environment
    env = RouterGLEnv(
        data_path=args.data_path,
        max_budget=args.max_budget,
        betas=betas_list,
        active_beta_idx=active_idx,
        depletion_penalty=args.depletion_penalty,
        shuffle=args.shuffle,
        global_stream=args.global_stream,
        max_steps=args.max_steps
    )
    
    # Seed the environment
    env.reset(seed=args.seed)

    # Initialize PPO Model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
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
