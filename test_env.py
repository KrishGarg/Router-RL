import numpy as np
from router_env import RouterGLEnv

def test_router_env():
    print("=== Testing RouterGLEnv ===")
    
    # Initialize env with multiple betas
    betas = [0.0, 0.001, 0.01, 0.1]
    env = RouterGLEnv(
        data_path="data/aligned_data.csv",
        max_budget=1000,  # Lower budget to trigger depletion easily for testing
        betas=betas,
        active_beta_idx=2,  # Active beta = 0.01
        depletion_penalty=-10.0,
        shuffle=True
    )
    
    # Check observation space and shape
    assert env.observation_space.shape == (28,), f"Expected shape (28,), got {env.observation_space.shape}"
    assert env.action_space.n == 5, f"Expected action space of size 5, got {env.action_space.n}"
    
    # Run 3 episodes
    for ep in range(3):
        print(f"\n--- Episode {ep + 1} ---")
        obs, info = env.reset()
        assert obs.shape == (28,), f"Observation shape mismatch on reset: {obs.shape}"
        assert "conversation_hash" in info, "Missing conversation hash in reset info"
        
        print(f"Started Conversation: {info['conversation_hash']} with {info['total_turns']} turns")
        print(f"Initial Obs (first 5 elements): {obs[:5]}")
        
        terminated = False
        turn = 0
        while not terminated:
            # Sample a random action
            action = env.action_space.sample()
            
            # Step in environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Validate next observation shape
            assert next_obs.shape == (28,), f"Observation shape mismatch on step: {next_obs.shape}"
            
            # Print turn summary safely without terminal emoji/encoding errors
            model_names = ["Q0.6B", "Mini8B", "Q30B", "Q30B_Inst", "Reject"]
            action_name = model_names[action]
            
            print(f"  Turn {turn}: Action={action_name} | Cost={step_info['cost']:.1f} | Score={step_info['score']:.1f} "
                  f"| Remaining Budget={step_info['remaining_budget']:.1f} | Reward={reward:.4f}")
            
            # Assert reward matches active beta calculation
            active_beta = betas[2]
            if step_info["budget_depleted"]:
                assert reward == -10.0, f"Expected depletion penalty -10.0, got {reward}"
                print("  [Budget Depleted! Episode terminated.]")
            else:
                expected_reward = step_info["score"] - active_beta * step_info["cost"]
                assert abs(reward - expected_reward) < 1e-5, f"Reward mismatch: expected {expected_reward}, got {reward}"
                
            # Verify all betas rewards dict is present and correct
            rewards_all = step_info["rewards_all_betas"]
            for b in betas:
                assert b in rewards_all, f"Beta {b} missing from rewards_all_betas"
                if step_info["budget_depleted"]:
                    assert rewards_all[b] == -10.0
                else:
                    expected_b_reward = step_info["score"] - b * step_info["cost"]
                    assert abs(rewards_all[b] - expected_b_reward) < 1e-5
            
            turn += 1
            obs = next_obs
            
        print(f"Episode {ep + 1} finished successfully!")
        
    print("\nAll RouterGLEnv tests passed successfully!")

if __name__ == "__main__":
    test_router_env()
