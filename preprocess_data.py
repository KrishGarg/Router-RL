import os
import json
import pandas as pd
from huggingface_hub import hf_hub_download

# Define constants
REPO_ID = "JiaqiXUE/mmr-routing-20k"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aligned_data.csv")

MODELS = {
    "qwen_06": {"dir": "qwen3-0.6b", "name": "Qwen3-0.6B"},
    "ministral": {"dir": "ministral-8b", "name": "Ministral-3-8B"},
    "qwen_30": {"dir": "qwen3-30b-a3b", "name": "Qwen3-30B-A3B"},
    "qwen_30_inst": {"dir": "qwen3-30b-a3b-instruct", "name": "Qwen3-30B-A3B-Instruct"}
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download and parse features
    print("Downloading query features...")
    features_path = hf_hub_download(repo_id=REPO_ID, filename="data/features/qwen06b_20k.jsonl", repo_type="dataset")
    
    print("Parsing features...")
    feat_rows = []
    with open(features_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            h = data["conversation_hash"]
            t_idx = data["turn_idx"]
            difficulty = data.get("difficulty", 0)
            
            # Flatten the nested features dictionary
            flat_row = {
                "conversation_hash": h,
                "turn_idx": t_idx,
                "difficulty": difficulty
            }
            if "features" in data:
                for k, v in data["features"].items():
                    flat_row[f"feat_{k}"] = v
            feat_rows.append(flat_row)
            
    df_merged = pd.DataFrame(feat_rows)
    print(f"Loaded {len(df_merged)} feature rows.")

    # 2. Process each model's scores and responses
    for key, info in MODELS.items():
        dir_name = info["dir"]
        model_name = info["name"]
        
        print(f"\nProcessing {model_name}...")
        
        # Download scores and responses
        scores_path = hf_hub_download(repo_id=REPO_ID, filename=f"data/{dir_name}/judge_scores.jsonl", repo_type="dataset")
        resp_path = hf_hub_download(repo_id=REPO_ID, filename=f"data/{dir_name}/responses.jsonl", repo_type="dataset")
        
        # Read scores
        print(f"  Parsing scores for {model_name}...")
        score_rows = []
        with open(scores_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                score_rows.append({
                    "conversation_hash": data["conversation_hash"],
                    "turn_idx": data["turn_idx"],
                    f"{key}_score": data["score"]
                })
        df_score = pd.DataFrame(score_rows)
        
        # Read responses and calculate token lengths
        print(f"  Parsing responses for {model_name}...")
        resp_rows = []
        with open(resp_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                h = data["conversation_hash"]
                for turn in data.get("turns", []):
                    t_idx = turn["turn_idx"]
                    resp_text = turn.get("response", "")
                    
                    # Word and char count metrics
                    words = len(resp_text.split())
                    chars = len(resp_text)
                    # Token count (1 word = 1 token)
                    est_tokens = max(1, words)
                    
                    resp_rows.append({
                        "conversation_hash": h,
                        "turn_idx": t_idx,
                        f"{key}_resp_len": chars,
                        f"{key}_resp_tokens": est_tokens
                    })
        df_resp = pd.DataFrame(resp_rows)
        
        # Merge this model's scores and responses
        df_model = pd.merge(df_score, df_resp, on=["conversation_hash", "turn_idx"], how="inner")
        print(f"  Loaded {len(df_model)} score-response pairs for {model_name}.")
        
        # Merge into global dataframe
        df_merged = pd.merge(df_merged, df_model, on=["conversation_hash", "turn_idx"], how="inner")

    print("\n--- Final Merged Dataset Stats ---")
    print(f"Total aligned turn rows: {len(df_merged)}")
    print(f"Unique conversations: {df_merged['conversation_hash'].nunique()}")
    
    # Sort by conversation_hash and turn_idx to maintain sequence
    df_merged = df_merged.sort_values(by=["conversation_hash", "turn_idx"]).reset_index(drop=True)
    
    # Save to CSV
    print(f"Saving merged data to {OUTPUT_FILE}...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
