# SciRerankBench - Environment Requirements

This project requires **multiple conda environments** because different reranker models have conflicting dependencies. The original experiments used 4 separate environments.

## Environment Setup

### 1. `rerank` (Python 3.10) — Primary environment

Supports: **BGE, Jina, BCE, MXBAI (cross-encoder), MiniLM (flashrank), ColBERT, T5, GTE**

```bash
conda create -n rerank python=3.10
conda activate rerank
pip install -r requirements_rerank.txt
```

### 2. `rankify` (Python 3.10) — Rankify-based models

Supports: **RankT5, ListT5, SPLADE, TwoLAR, LLM2Vec, RankGPT**

```bash
conda create -n rankify python=3.10
conda activate rankify
pip install -r requirements_rankify.txt
```

### 3. `bce` (Python 3.10) — BCE-specific environment

Supports: **BCE** (alternative to rerank env, uses newer BCEmbedding)

```bash
conda create -n bce python=3.10
conda activate bce
pip install -r requirements_bce.txt
```

### 4. `rearank` (Python 3.12) — Rearank agent

Supports: **Rearank**

```bash
conda create -n rearank python=3.12
conda activate rearank
pip install -r requirements_rearank.txt
```

## Model → Environment Mapping

| Model | Environment | Import |
|-------|-------------|--------|
| BGE | `rerank` | `FlagEmbedding.FlagReranker` |
| Jina | `rerank` | `sentence_transformers.CrossEncoder` |
| BCE | `bce` or `rerank` | `BCEmbedding.RerankerModel` |
| MXBAI (cross-encoder) | `rerank` | `rerankers.Reranker` |
| MiniLM (flashrank) | `rerank` | `rerankers.Reranker` |
| ColBERT | `rerank` | `rerankers.Reranker` |
| T5 | `rerank` | `rerankers.Reranker` |
| GTE | `rerank` | `transformers.AutoModelForSequenceClassification` |
| RankT5 | `rankify` | `rankify` |
| ListT5 | `rankify` | `rankify` |
| SPLADE | `rankify` | `rankify` |
| TwoLAR | `rankify` | `rankify` |
| LLM2Vec | `rankify` | `rankify` |
| RankGPT | `rankify` | `rankify` |
| Rearank | `rearank` | custom |

## Running Evaluation

Use the provided shell wrapper to automatically select the correct environment:

```bash
bash scripts/eval/run_eval.sh --model bge --subject biology --task nc --llm Qwen-72B --gpu 0
```

Or activate the correct environment manually:

```bash
conda activate rerank
python scripts/eval/eval_reranker.py --model bge --subject biology --task nc --llm Qwen-72B --gpu 0
```
