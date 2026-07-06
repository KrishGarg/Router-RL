---
license: apache-2.0
task_categories:
  - text-classification
language:
  - en
tags:
  - llm-routing
  - multi-turn
  - conversation
  - quality-prediction
size_categories:
  - 10K<n<100K
---

# MMR: Multi-turn Model Routing Dataset

A large-scale dataset for studying **LLM routing in multi-turn conversations**. Contains 20,000 multi-turn conversations from WildChat with per-turn responses and quality scores for 4 LLMs, enabling research on adaptive cost-quality tradeoff routing.

## Dataset Summary

| Stat | Value |
|------|-------|
| Conversations | 20,000 |
| Total user turns | ~81,000 |
| Languages | English |
| Source | WildChat-1M (filtered) |
| Judge model | Qwen3-235B-A22B-Instruct-2507 |

## Models

Each model's responses are independently generated and scored by the judge:

| Model | Parameters | Role | Directory |
|-------|-----------|------|-----------|
| Qwen3-0.6B | 0.6B | Weak | `qwen3-0.6b/` |
| Ministral-3-8B-Instruct-2512 | 8B | Weak | `ministral-8b/` |
| Qwen3-30B-A3B | 30B (3B active) | Weak | `qwen3-30b-a3b/` |
| Qwen3-30B-A3B-Instruct-2507 | 30B (3B active) | Strong | `qwen3-30b-a3b-instruct/` |

## Dataset Structure

```
data/
├── conversations.jsonl                    # Original WildChat conversations (20K)
├── qwen3-0.6b/
│   ├── responses.jsonl                    # Per-turn responses
│   └── judge_scores.jsonl                 # Per-turn quality scores (0-10)
├── ministral-8b/
│   ├── responses.jsonl
│   └── judge_scores.jsonl
├── qwen3-30b-a3b/
│   ├── responses.jsonl
│   └── judge_scores.jsonl
├── qwen3-30b-a3b-instruct/
│   ├── responses.jsonl
│   └── judge_scores.jsonl
└── features/
    └── qwen06b_20k.jsonl                  # 26 handcrafted routing features
```

## File Formats

### conversations.jsonl
Each line is a conversation:
- `conversation_hash`: unique identifier
- `model`: original WildChat source model
- `turn`: number of turns
- `language`: language code
- `conversation`: list of `{role, content}` message objects

### responses.jsonl
Each line is a conversation with per-turn responses:
- `conversation_hash`: identifier
- `num_turns`: total turns
- `turns`: list of `{turn_idx, user_query, response}`

### judge_scores.jsonl
Each line is a per-turn quality score:
- `conversation_hash`, `turn_idx`: identifies the turn
- `score`: integer 0-10 (quality rating by the judge)
- `reasoning`: text explanation of the score

### features/qwen06b_20k.jsonl
Each line is a per-turn feature vector (26 features):
- Query features: length, question marks, code/math/URL presence, word stats
- Context features: prior turn counts, average lengths, context size
- Dependency features: pronoun references, continuation/correction markers, self-containedness
- Labels: `weak_score`, `strong_score`, `difficulty`, `label_t1`-`label_t4` (binary at thresholds 1-4)

## Key Design Decisions

- **Context preservation**: When generating responses for turn N, the conversation context uses the **original dataset's assistant responses** for turns 0..N-1, not the model's own generated responses. This ensures all models see identical context.
- **Thinking mode disabled**: Qwen3 models had thinking mode disabled (`enable_thinking: False`) to avoid wasted tokens on `<think>` blocks.
- **Independent judge evaluation**: Each model is scored independently (0-10 scale) with reasoning, avoiding position bias. The same judge model (Qwen3-235B) evaluates all models.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{mmr2025,
  title={MMR: Multi-turn Model Routing Dataset},
  author={Jiaqi Xue},
  year={2025},
  url={https://huggingface.co/datasets/JiaqiXue/mmr-routing-20k}
}
```

## License

Apache 2.0. The conversations are sourced from [WildChat](https://huggingface.co/datasets/allenai/WildChat-1M) under its original license.
