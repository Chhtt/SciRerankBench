# Metrics

## Reranking Metrics

### Recall@k

Measures what fraction of relevant documents are retrieved within the top-k results.

$$\text{Recall@}k = \frac{|\{\text{relevant items in top }k\}|}{|\{\text{total relevant items}\}|} \times 100$$

We report both Recall@5 and Recall@10.

### Average Precision (AP)

Measures precision averaged over all recall levels, rewarding models that rank relevant documents higher.

$$\text{AP} = \frac{1}{N} \sum_{r=1}^{K} \text{Precision@}r \times \text{rel}(r)$$

where $N$ is the total number of relevant passages and $\text{rel}(r)$ indicates whether the passage at rank $r$ is relevant.

### Mean Reciprocal Rank (MRR)

Measures how quickly the first relevant document is retrieved.

$$\text{MRR} = \frac{1}{\text{rank}_{\text{first relevant}}}$$

## Relevance Judgment

A passage is considered relevant if it contains the golden answer, determined by **Contain Answer Score**:

$$\text{ContainAnswer}(A, C) = \frac{|\text{tokens}(A) \setminus \text{stopwords} \cap \text{tokens}(C) \setminus \text{stopwords}|}{|\text{tokens}(A) \setminus \text{stopwords}|}$$

If this score exceeds 0.6, the passage is considered to contain the answer.

## Generation Metrics

### Token-level Recall

$$\text{Recall} = \frac{|P \cap T|}{|T|} \times 100$$

where $P$ and $T$ are the tokenized predicted and true answer sets.

### Token-level F1

$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \times 100$$

## Grouped Standard Deviation

We report results as mean ± standard deviation using a **grouped** methodology:

1. Sort all per-question scores for a given configuration
2. Split into 3 roughly equal groups
3. Compute the mean of each group
4. Report the overall mean ± standard deviation of the 3 group means

This provides a more robust variance estimate than per-sample standard deviation, as it captures variation across different subsets of the data rather than individual noisy samples.
