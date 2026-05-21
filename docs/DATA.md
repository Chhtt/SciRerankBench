# Dataset

## Overview

SciRerankBench contains **~4,500 question-answer-context (Q-A-C) triples** across **5 scientific subjects** and **5 task types**. Each Q-A-C triple consists of a scientific question, its golden answer(s), and a pool of 100 candidate passages.

## Task Types

| Task | Directory | Description | Context Pool |
|------|-----------|-------------|--------------|
| **NC** (Noisy Contexts) | `dataset/nc/` | Noise resilience | 5 relevant (dense retrieval top-5) + 95 random abstracts |
| **Base** | `dataset/base/` | Clean retrieval baseline | 100 relevant candidates |
| **CC** (Counterfactual) | `dataset/cc/` | Factual consistency | 90 standard candidates + 10 counterfactual passages |
| **SSLI** (Semantically Similar but Logically Irrelevant) | `dataset/ssli/` | Logical discrimination | 90 standard candidates + 10 semantically similar but irrelevant passages |
| **Multi-Hop** | `dataset/multihop/` | Cross-document reasoning | 2 semantically linked abstracts |

## File Format

Each file is a JSONL with one entry per question:

```json
{
    "QUESTION": "What are perovskite optoelectronic devices utilizing lead-free perovskites?",
    "ALL_CONTEXTS": ["Passage 1 text", "Passage 2 text", ..., "Passage 100 text"],
    "SELECTED_CONTEXTS_INDICES": [0, 3, 7, 12, 15, 21, 33, 45, 67, 89],
    "GOLDEN_ANSWERS": ["solar cells, LEDs, and near-infrared photodetectors"],
    "MODEL_RESULTS": [{
        "MODEL": "Qwen-72B",
        "RERANK_SCORE": {"Recall@5": 80.0, "Recall@10": 90.0, "AP": 75.0, "MRR": 0.5},
        "GENERATE_SCORE": {"f1": 60.0, "precision": 55.0, "recall": 65.0},
        "LLM_ANSWER": "The devices include solar cells, LEDs, and photodetectors."
    }]
}
```

### Field Descriptions

- `QUESTION`: The scientific question
- `ALL_CONTEXTS`: All 100 candidate passages
- `SELECTED_CONTEXTS_INDICES`: Indices of the top-10 passages selected by the reranker (0-based for most models, 1-based for RankT5, ListT5, SPLADE, LLM2Vec, Rearank, RankGPT, TwoLAR)
- `GOLDEN_ANSWERS`: Ground-truth answer(s)
- `MODEL_RESULTS`: Contains the LLM-generated answer and evaluation metrics

## Naming Convention

Files follow the pattern: `rebuild_{config}_{topK_first}-{topK_second}_{Reranker}.jsonl`

- `5-95` for NC (5 relevant + 95 random)
- `100-0` for Base (all relevant)
- `100-10` for CC, SSLI, Multi-Hop

## Statistics

| Subject | NC | Base | CC | SSLI | Multi-Hop |
|---------|----|----|----|----|----|
| Biology | 2,499 | 2,499 | 2,496 | 2,497 | 1,246 |
| Math | 2,494 | 2,494 | 2,491 | 2,493 | 1,631 |
| Physics | 2,491 | 2,491 | 2,494 | 2,492 | 1,425 |
| Geology | 2,493 | 2,493 | 2,493 | 2,496 | 1,598 |
| Chemistry | ~2,490 | ~2,490 | ~2,490 | ~2,490 | ~1,500 |

## Data Construction

See the paper's Section 3 for full details. Briefly:

1. **Source**: OpenAlex scholarly database, filtered to 5 subjects
2. **Single-hop QA**: LMQG (t5-small-squad-qag) generates (answer, question) pairs from abstracts
3. **Multi-hop QA**: Unsupervised-Multi-hop-QA generates bridging questions across two semantically linked abstracts
4. **NC**: Dense retrieval top-5 + 95 random abstracts
5. **CC/SSLI**: Mistral-7B-Instruct-v0.2 generates counterfactual/semantically-similar distractors
