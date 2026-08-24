# SciRerankBench

Anonymous release of **SciRerankBench**: a benchmark for evaluating rerankers within Retrieval-Augmented Generation (RAG) systems in the scientific domain.

## Project Structure

```
SciRerankBench/
├── README.md                          # This file
├── config/
│   ├── models.yaml                    # Reranker model configurations
│   └── llms.yaml                      # LLM configurations
├── dataset/                           # Download from [DATA_URL] and extract here
│   ├── nc/                            # Noisy Contexts (5 relevant + 95 random)
│   ├── base/                          # Clean retrieval (100-0, all relevant)
│   ├── cc/                            # Counterfactual Contexts (90 + 10 counterfactual)
│   ├── ssli/                          # Semantically Similar but Logically Irrelevant
│   └── multihop/                      # Multi-hop reasoning (2 linked abstracts)
├── scripts/
│   ├── qa_generation/
│   │   ├── generate_single_hop.py     # LMQG-based single-hop QA generation
│   │   └── generate_multi_hop.py      # Multi-hop QA generation pipeline
│   ├── data/
│   │   ├── 01_build_vector_store.py   # Qdrant index from OpenAlex abstracts
│   │   ├── 02_build_context_pools.py  # Build 100-passage context pools per task
│   │   └── run_pipeline.sh            # Full pipeline orchestrator
│   ├── eval/
│   │   ├── eval_reranker.py           # Main evaluation script (reranker + LLM)
│   │   ├── run_eval.sh                # Per-model env dispatch
│   │   └── llm_model.py               # LLM model factory
│   └── metrics/
│       ├── compute_metrics.py         # Compute per-question mean metrics
│       └── gen_tables.py              # Generate LaTeX tables
├── requirements/                      # Environment-specific dependencies
│   ├── README.md                      # Environment setup guide
│   ├── rerank.txt                     # BGE, Jina, MXBAI, MiniLM, ColBERT, T5, GTE
│   ├── rankify.txt                    # RankT5, ListT5, SPLADE, TwoLAR, LLM2Vec, RankGPT
│   ├── bce.txt                        # BCE reranker
│   └── rearank.txt                    # Rearank agent
├── docs/
│   ├── DATA.md                        # Dataset construction details + pipeline docs
│   ├── EVAL.md                        # Evaluation protocol
│   └── METRICS.md                     # Metric definitions
└── .gitignore
```
```

## Dataset

> **Download:** The dataset is hosted externally at **[DATA_URL]** (coming soon).
>
> **Size:** ~5.7 GB, 25 JSONL files, ~58,000 Q-A-C (Question-Answer-Context) triples.
>
> **Format:** Each file is JSONL with fields `QUESTION`, `ALL_CONTEXTS` (100 passages), `SELECTED_CONTEXTS_INDICES` (top-10 by reranker), `GOLDEN_ANSWERS`, and `MODEL_RESULTS`. See [docs/DATA.md](docs/DATA.md) for details.

SciRerankBench covers **5 scientific subjects** (Biology, Chemistry, Geology, Physics, Math) with **5 task types**:

| Task | Directory | Description | Context Composition | Per-Subject |
|------|-----------|-------------|---------------------|-------------|
| NC | `{base}/nc/` | Noise resilience | 5 relevant + 95 random | ~2,490 |
| Base | `{base}/base/` | Clean retrieval | 100 relevant | ~2,490 |
| CC | `{base}/cc/` | Factual consistency | 90 candidates + 10 counterfactual | ~2,490 |
| SSLI | `{base}/ssli/` | Logical discrimination | 90 candidates + 10 semantically similar but irrelevant | ~2,490 |
| Multi-Hop | `{base}/multihop/` | Cross-document reasoning | 2 semantically linked abstracts | ~1,200–1,600 |

Each subject has its own subdirectory (`biology/`, `chemistry/`, `geology/`, `math/`, `physics/`) under each task type.

### File Naming

Files follow the pattern `rebuild_{relevant-random}_{pool-size}_{Reranker}.jsonl`:
- `rebuild_5-95_100-10_{Reranker}.jsonl` — NC task
- `rebuild_100-0_100-10_{Reranker}.jsonl` — Base task
- `rebuild_100-10_{Reranker}.jsonl` — CC, SSLI, Multi-Hop tasks

## Rerankers Evaluated

- **Dense cross-encoders**: BGE, Jina, BCE, MXBAI
- **Sparse lexical**: SPLADE
- **Lightweight**: MiniLM (FlashRank)
- **Late-interaction**: ColBERT
- **LLM-based**: RankGPT, LLM2Vec
- **Seq2Seq/Listwise**: RankT5, ListT5, T5
- **Knowledge distillation**: TwoLAR
- **Agent-based**: Rearank

## LLMs Evaluated

- Qwen series (7B, 14B, 72B)
- Llama2 series (7B, 13B, 70B)
- Other open-source LLMs

## Quick Start

### Environment Setup

Different reranker models require different conda environments due to conflicting dependencies. See [requirements/README.md](requirements/README.md) for the full mapping.

```bash
# Primary env (BGE, Jina, MXBAI, MiniLM, ColBERT, T5, GTE)
conda create -n rerank python=3.10 && conda activate rerank
pip install -r requirements/requirements_rerank.txt

# Rankify-based models (RankT5, ListT5, SPLADE, TwoLAR, LLM2Vec, RankGPT)
conda create -n rankify python=3.10 && conda activate rankify
pip install -r requirements/requirements_rankify.txt

# BCE standalone
conda create -n bce python=3.10 && conda activate bce
pip install -r requirements/requirements_bce.txt

# Rearank
conda create -n rearank python=3.12 && conda activate rearank
pip install -r requirements/requirements_rearank.txt
```

### Run Evaluation

```bash
# Using the auto-dispatch script (selects correct env)
bash scripts/eval/run_eval.sh --model bge --subject biology --task nc --llm Qwen-72B --gpu 0

# Or manually activate the env first
conda activate rerank
python scripts/eval/eval_reranker.py --model bge --subject biology --task nc --llm Qwen-72B --gpu 0

# Compute metrics on existing results
python scripts/metrics/compute_metrics.py --data_root dataset/ --metric Recall@10

# Generate LaTeX tables
python scripts/metrics/gen_tables.py --data_root dataset/
```
