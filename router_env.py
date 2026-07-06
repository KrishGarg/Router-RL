import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class RouterGLEnv(gym.Env):
    """
    A custom Gymnasium environment for multi-turn LLM routing.
    Each episode represents a single conversation containing multiple turns.
    At each turn, the router chooses which model to handle the query or to reject it.
    The budget is global for the conversation and resets at the start of the episode.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path="data/aligned_data.csv",
        max_budget=10000,
        betas=None,
        active_beta_idx=0,
        depletion_penalty=-10.0,
        shuffle=True
    ):
        super().__init__()
        
        # Load and group dataset
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Please run preprocess_data.py first.")
            
        print(f"Loading environment data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Identify feature columns (difficulty + all feat_* columns)
        self.feature_cols = []
        if "difficulty" in df.columns:
            self.feature_cols.append("difficulty")
        feat_cols = sorted([c for c in df.columns if c.startswith("feat_")])
        self.feature_cols.extend(feat_cols)
        
        self.num_features = len(self.feature_cols)
        # Total observation size: 1 (normalized budget) + num_features = 28 features
        self.observation_dim = 1 + self.num_features
        
        # Define Action Space:
        # 0: Qwen3-0.6B
        # 1: Ministral-3-8B
        # 2: Qwen3-30B-A3B
        # 3: Qwen3-30B-A3B-Instruct
        # 4: Reject/Drop query
        self.action_space = spaces.Discrete(5)
        
        # Define Observation Space (Box space of size 28)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # Model mapping
        self.model_keys = ["qwen_06", "ministral", "qwen_30", "qwen_30_inst"]
        
        # Group data by conversation hash to facilitate sequential steps
        print("Grouping conversation logs...")
        self.conversations = {}
        for h, grp in df.groupby("conversation_hash"):
            self.conversations[h] = grp.sort_values("turn_idx").reset_index(drop=True)
            
        self.conversation_hashes = list(self.conversations.keys())
        self.shuffle = shuffle
        
        # RL hyperparameters
        self.max_budget = max_budget
        self.betas = betas if betas is not None else [0.01]
        self.active_beta_idx = active_beta_idx
        self.depletion_penalty = depletion_penalty
        
        # State variables
        self.current_conv_hash = None
        self.current_conv_df = None
        self.current_turn = 0
        self.remaining_budget = 0.0
        self.num_turns_total = 0
        self.conv_index = 0
        
        print(f"Environment initialized with {len(self.conversations)} conversations.")
        print(f"Observation dimension: {self.observation_dim} (1 budget + {self.num_features} query/context features)")
        print(f"Betas: {self.betas} (Active beta: {self.betas[self.active_beta_idx]})")

    def _get_obs(self):
        """Assembles and returns the current state vector."""
        if self.current_conv_df is None or self.current_turn >= self.num_turns_total:
            return np.zeros(self.observation_dim, dtype=np.float32)
            
        # Get query/context features for the current turn
        row = self.current_conv_df.iloc[self.current_turn]
        features = row[self.feature_cols].values.astype(np.float32)
        
        # Normalize the remaining budget
        norm_budget = np.array([self.remaining_budget / self.max_budget], dtype=np.float32)
        
        # Concatenate normalized budget and turn features
        obs = np.concatenate([norm_budget, features])
        return obs

    def reset(self, seed=None, options=None):
        """Resets the environment for a new conversation episode."""
        super().reset(seed=seed)
        
        # Select next conversation
        if self.shuffle:
            self.current_conv_hash = self.np_random.choice(self.conversation_hashes)
        else:
            self.current_conv_hash = self.conversation_hashes[self.conv_index]
            self.conv_index = (self.conv_index + 1) % len(self.conversation_hashes)
            
        self.current_conv_df = self.conversations[self.current_conv_hash]
        self.num_turns_total = len(self.current_conv_df)
        self.current_turn = 0
        self.remaining_budget = float(self.max_budget)
        
        obs = self._get_obs()
        info = {
            "conversation_hash": self.current_conv_hash,
            "total_turns": self.num_turns_total,
            "remaining_budget": self.remaining_budget
        }
        
        return obs, info

    def step(self, action):
        """
        Executes one step in the environment.
        """
        # Ensure action is within bounds
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Check if we are already out of turns (should not happen if terminated is handled)
        if self.current_turn >= self.num_turns_total:
            return self._get_obs(), 0.0, True, False, {}
            
        row = self.current_conv_df.iloc[self.current_turn]
        
        score = 0.0
        cost = 0.0
        depletion = False
        
        if action < 4:
            model_key = self.model_keys[action]
            model_score = float(row[f"{model_key}_score"])
            model_tokens = float(row[f"{model_key}_resp_tokens"])
            
            # Check budget availability
            if self.remaining_budget >= model_tokens:
                score = model_score
                cost = model_tokens
                self.remaining_budget -= model_tokens
            else:
                # Budget depletion: model cannot be queried
                score = 0.0
                cost = 0.0
                depletion = True
        else:
            # Action 4: Drop/Reject request
            score = 0.0
            cost = 0.0

        # Calculate reward for the active beta
        active_beta = self.betas[self.active_beta_idx]
        if depletion:
            reward = self.depletion_penalty
        else:
            reward = score - active_beta * cost
            
        # Calculate rewards for all betas to put in info dict
        rewards_all_betas = {}
        for b in self.betas:
            if depletion:
                rewards_all_betas[b] = self.depletion_penalty
            else:
                rewards_all_betas[b] = score - b * cost
                
        # Advance state
        self.current_turn += 1
        
        # Terminate if conversation is over OR budget is depleted
        terminated = (self.current_turn >= self.num_turns_total) or depletion
        truncated = False
        
        obs = self._get_obs()
        
        info = {
            "conversation_hash": self.current_conv_hash,
            "turn_idx": self.current_turn - 1,
            "action": action,
            "score": score,
            "cost": cost,
            "remaining_budget": self.remaining_budget,
            "rewards_all_betas": rewards_all_betas,
            "budget_depleted": depletion
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        """Optional rendering for debugging."""
        print(f"Conv: {self.current_conv_hash} | Turn: {self.current_turn}/{self.num_turns_total} | Budget: {self.remaining_budget:.1f}")
