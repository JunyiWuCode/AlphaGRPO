# DVReward: Decompositional Verifiable Reward

AlphaGRPO trains with a **Decompositional Verifiable Reward (DVReward)** that scores each generated image against per-prompt *semantic* and *quality* questions rather than a single scalar. This document covers the end-to-end flow for training on your own prompt set:

1. [Collect prompts](#step-1-collect-prompts)
2. [Decompose prompts into evaluation questions](#step-2-decompose-prompts-into-evaluation-questions) (via `scripts/decompose.py`)
3. [Drop the dataset in place](#step-3-drop-the-dataset-in-place)
4. [Configure training to use DVReward](#step-4-configure-training-to-use-dvreward)

---

## Step 1: Collect prompts

Put one prompt per line in a plain text file:

```text
# prompts.txt
A photo of a dog and a red cat.
A dense forest, an old oak tree with 'Tiger' carved into the bark.
A sunken pirate ship, 'helmet' burned into leather.
```

## Step 2: Decompose prompts into evaluation questions

`scripts/decompose.py` turns each prompt into structured *semantic* + *quality* evaluation questions via an OpenAI-compatible LLM.

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARK_API_KEY` | API key (required) | – |
| `OPENAI_API_URL` | API base URL (OpenAI-compatible) | – |
| `MODEL_NAME` | Model name | `gpt-4.1-2025-04-14` |
| `AZURE_OPENAI_API_VERSION` | Azure API version | `2024-03-01-preview` |

### Usage

```bash
export ARK_API_KEY=YOUR_API_KEY
export OPENAI_API_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
export MODEL_NAME=gpt-4.1-2025-04-14              # any strong instruction-following LLM

python scripts/decompose.py \
  --input prompts.txt \
  --output train.jsonl \
  --workers 32 --verbose
```

| Argument | Description |
|----------|-------------|
| `--input` | Input txt file, one prompt per line |
| `--output` | Output jsonl file |
| `--workers` | Number of parallel workers (default: 32) |
| `--verbose` | Print detailed processing logs |

### Output format

One JSON record per prompt:

```json
{
  "prompt": "A photo of a dog and a red cat.",
  "original_index": 0,
  "semantic_questions": [
    {"question_type": "semantic", "question": "Is there a dog?",     "is_valid": true, "tag": "Existence"},
    {"question_type": "semantic", "question": "Is there a cat?",     "is_valid": true, "tag": "Existence"},
    {"question_type": "semantic", "question": "Is the cat red?",     "is_valid": true, "tag": "Attribute"}
  ],
  "quality_questions": [
    {"question_type": "quality",  "question": "Does the dog have anatomically correct proportions?", "is_valid": true, "tag": "Anatomy"},
    {"question_type": "quality",  "question": "Is the cat's fur texture realistic?",                  "is_valid": true, "tag": "Texture"}
  ]
}
```

### Resume support

The script auto-resumes on re-run — if interrupted, simply re-run the same command and already completed prompts (matched by `original_index`) are skipped.

## Step 3: Drop the dataset in place

Create a directory under `alpha_grpo/dataset/` and provide both splits:

```
alpha_grpo/dataset/
└── my_dataset/
    ├── train.jsonl
    └── test.jsonl
```

See `alpha_grpo/dataset/alphagrpo20k/` for a working example.

## Step 4: Configure training to use DVReward

In the config function of `config/bagel.py` you plan to run, set:

```python
config.dataset   = 'my_dataset'        # matches the directory name under alpha_grpo/dataset/
config.prompt_fn = 'dvreward'          # selects TextPromptWithQuestionDataset
config.reward_fn = {"dvreward": 1.0}   # enables decompositional verifiable reward
```

At training time the reward server (Qwen-VL-based MLLM) scores each generated image by answering every semantic + quality question — no separate scalar reward model is needed. See the main README for the training launch command.
