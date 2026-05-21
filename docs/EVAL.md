# Evaluation Protocol

## Reranker Evaluation

For each question with 100 candidate contexts:
1. The reranker scores and ranks all 100 passages
2. Top-10 passages are selected for the LLM
3. Reranking quality is measured against the ground-truth relevance labels

### Metrics

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **Recall@k** | `# relevant in top-k / Total relevant` | Retrieval completeness |
| **AP** | Average precision over all recall levels | Overall ranking quality |
| **MRR** | `1 / rank_of_first_relevant` | Early retrieval accuracy |

## LLM Evaluation

The LLM generates answers based on the top-10 reranked contexts:

### Prompt Template
```
You are given a question and a set of contexts. Your task is to provide a clear and concise answer to the question based on the contexts provided.

Answer based **only** on the contexts. If none are relevant, say: **No Answer Present.**

Keep your answer short — ideally within 2 sentences.

---
QUESTION: {QUESTION}
CONTEXTS: {CONTEXTS}

###
ANSWER:
```

### Metrics

| Metric | Formula | What it measures |
|--------|---------|------------------|
| **Contain Answer Score** | `|tokens(golden) ∩ tokens(context)| / |tokens(golden)|` with stopword filtering, threshold 0.6 | Whether context contains the answer |
| **Token-level Recall** | `|predicted ∩ golden| / |golden| × 100` | Answer accuracy |
| **Token-level F1** | `2 × P × R / (P + R) × 100` | Combined precision and recall |

## Standard Deviation Calculation

We use a **grouped standard deviation** methodology:
1. Split per-question scores into 3 roughly equal groups
2. Compute the mean of each group
3. Report the overall mean ± standard deviation of the 3 group means

This provides a more robust estimate of variance than per-sample standard deviation.
