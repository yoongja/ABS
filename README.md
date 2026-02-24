# [PAKDD 2026] Adaptive Beam Search with Shannon Entropy for Data-centric Reasoning in LLMs

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3820/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-green)](https://openai.com/)
[![LLaMA](https://img.shields.io/badge/Meta-LLaMA-purple)](https://ai.meta.com/llama/)

Welcome to the **Adaptive Beam Search** repository! This repo is official code for " Adaptive Beam Search with Shannon Entropy for Data-centric Reasoning in LLMs" , PAKDD 2026 by Yoonji Kim, Yujin Jeong, Jieun Kim.

This repository supports running adaptive beam search on a variety of reasoning tasks across multiple models and prompting strategies.

---

## 📋 Prompt Overview

* **CoT**: Chain-of-Thought
* **ToT**: Tree-of-Thoughts

### Arithmetic Reasoning
- GSM8K: [CoT](/(gpt%20or%20llama)/prompts/arithmetic/gsm8k.py)
- AQUA: [CoT](/(gpt%20or%20llama)/prompts/arithmetic/aqua.py)

### Symbolic Reasoning
- Date Understanding: [CoT](/(gpt%20or%20llama)/prompts/symbolic/date_understanding.py)

### Commonsense Reasoning
- CSQA: [CoT](/(gpt%20or%20llama)/prompts/commonsense/csqa.py)
- StrategyQA: [CoT](/(gpt%20or%20llama)/prompts/commonsense/strategyqa.py)

### Algorithmic Reasoning
- Game of 24: [ToT](/gpt/prompts/algorithmic/game_of_24.py)

---

## 🛠️ How to Use

### 1. **Clone the Repository**
```bash
git clone https://github.com/yoongja/ABS.git
cd ABS
```

### 2. **Install Requirements**

```bash
pip install -r requirements.txt
```

### 3. **Prepare Environment Variables**

Create a `.env` file in the project root and add the necessary variables:

```env
# For using OpenAI models like GPT-4
OPENAI_API_KEY="your-openai-api-key"
```

### 4. **Prepare the Data**

Create a `data/` directory in the project root and place the dataset files inside:

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

## 🚀 Running Experiments

### GPT (OpenAI)

Navigate to the GPT directory and run:

```bash
cd adaptive_beam_search/gpt/
bash scripts/run_gsm8k.sh
```

### LLaMA

Navigate to the LLaMA directory and run:

```bash
cd adaptive_beam_search/llama/
bash scripts/run_gsm8k.sh
```
