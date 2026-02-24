# Adaptive Beam Search for Reasoning Tasks

This project supports running adaptive beam search on a variety of reasoning tasks. We provide implementations for multiple models and prompting strategies.

---

## Prompt Overview

* **CoT**: Chain-of-Thought
* **ToT**: Tree-of-Thoughts

### Arithmetic Reasoning
-   GSM8K: [CoT](/(gpt or llama)/prompts/arithmetic/gsm8k.py)
-   AQUA: [CoT](/(gpt or llama)/prompts/arithmetic/aqua.py)

### Symbolic Reasoning
-   Date Understanding: [CoT](/(gpt or llama)/prompts/symbolic/date_understanding.py)

### Commonsense Reasoning
-   CSQA: [CoT](/(gpt or llama)/prompts/commonsense/csqa.py)
-   StrategyQA: [CoT](/(gpt or llama)/prompts/commonsense/strategyqa.py)

### Algorithmic Reasoning
-   Game of 24: [ToT](/gpt/prompts/algorithmic/game_of_24.py)

---

## Setup Instructions

### 1. Install Requirements

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Prepare Environment Variables

This project may require API keys or paths to local models. Create a `.env` file in the project root and add the necessary variables.

```env
# For using OpenAI models like GPT-4
OPENAI_API_KEY="your-openai-api-key"

# For using local models like LLaMA (if applicable)
LLAMA_MODEL_PATH="/path/to/your/llama/weights"
```

### 3. Prepare the Data

Create a `data/` directory in the project root and place the dataset files inside. The expected structure is:

```
data/
├── gsm8k.json
├── aqua.json
├── date_understanding.json
├── csqa.json
├── strategyqa.json
└── game_of_24.json
```

---

## Running Experiments

Here is an example of how to run an experiment for the **GSM8K** task using the **llama2** model with adaptive beam search.

First, navigate to the project's LLaMA directory:

```bash
cd adaptive_beam_search/llama/
```

Then, run the experiment script for the desired task:

```bash
bash scripts/run_gsm8k.sh