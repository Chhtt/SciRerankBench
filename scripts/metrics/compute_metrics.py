"""
SciRerankBench - Metrics Computation

Computes per-reranker, per-subject, per-task metrics from evaluation results.
Uses grouped standard deviation: split scores into 3 groups, compute
group means, then report mean +/- std.

Directory structure:
    dataset/{task}/{subject}/rebuild_{config}_{Reranker}.jsonl

Usage:
    python compute_metrics.py --data_root ./dataset --metric Recall@10
    python compute_metrics.py --data_root ./dataset --metric AP --output results.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

try:
    import nltk
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

SUBJECTS_ORDER = ["biology", "chemistry", "geology", "physics", "math"]
SUBJECT_SHORT = {"biology": "Bio.", "chemistry": "Chem.", "geology": "Geo.",
                 "physics": "Phy.", "math": "Math."}

TASK_DIR_MAP = {
    "nc": "NC",
    "base": "Base",
    "cc": "CC",
    "ssli": "SSLI",
    "multihop": "Multi-Hop",
}

ONE_BASED_MODELS = {"rankt5", "listt5", "splade", "llm2vec", "rearank", "rankgpt", "twolar"}


def grouped_std(scores):
    """Compute mean and std by splitting scores into 3 groups."""
    n = len(scores)
    if n == 0:
        return 0.0, 0.0
    group_size = max(1, n // 3)
    groups = [scores[i:i + group_size] for i in range(0, n, group_size)]
    if len(groups) > 3:
        groups = groups[:3]
    group_means = [sum(g) / len(g) for g in groups if g]
    if not group_means:
        return 0.0, 0.0
    mean = sum(group_means) / len(group_means)
    std = (sum((x - mean) ** 2 for x in group_means) / len(group_means)) ** 0.5
    return mean, std


def tokenize(text):
    return text.split()


def compute_contain_answer(answer, context, threshold=0.6):
    """Soft semantic matching with stopword filtering."""
    if not answer or not answer[0]:
        return 0.0
    ans_tokens = tokenize(answer[0].lower())
    ctx_tokens = set(tokenize(context.lower()))
    filtered = [t for t in ans_tokens if t not in STOP_WORDS]
    if not filtered:
        return 0.0
    return sum(1 for t in filtered if t in ctx_tokens) / len(filtered)


def compute_rerank_metrics(golden_answers, context_list):
    """Compute reranking metrics for a single question."""
    relevant_indices = []
    first_hit_rank = None
    hits = 0
    precision_sum = 0.0

    for i, ctx in enumerate(context_list):
        if compute_contain_answer(golden_answers, ctx) >= 0.6:
            relevant_indices.append(i)
            hits += 1
            precision_sum += hits / (i + 1)
            if first_hit_rank is None:
                first_hit_rank = i + 1

    n_total = len(relevant_indices)
    n_rel_5 = sum(1 for idx in relevant_indices if idx < 5)
    n_rel_10 = sum(1 for idx in relevant_indices if idx < 10)
    recall_5 = (n_rel_5 / n_total * 100) if n_total > 0 else 0.0
    recall_10 = (n_rel_10 / n_total * 100) if n_total > 0 else 0.0
    ap = (precision_sum / hits * 100) if hits > 0 else 0.0
    mrr = (1 / first_hit_rank) if first_hit_rank is not None else 0.0

    return {"Recall@5": recall_5, "Recall@10": recall_10, "AP": ap, "MRR": mrr}


def discover_and_evaluate(data_root):
    """Walk dataset/{task}/{subject}/rebuild_*.jsonl and compute metrics."""
    # scores[task][subject][reranker][metric] = [list of scores]
    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    data_path = Path(data_root)
    if not data_path.exists():
        print(f"Data root not found: {data_root}")
        return scores

    for task_dir in sorted(data_path.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = TASK_DIR_MAP.get(task_dir.name, task_dir.name)

        for subj_dir in sorted(task_dir.iterdir()):
            if not subj_dir.is_dir():
                continue
            subject = subj_dir.name

            for fpath in sorted(subj_dir.glob("*.jsonl")):
                fname = fpath.name
                # Extract reranker name: rebuild_5-95_100-10_BCE.jsonl -> BCE
                model_raw = fname.replace(".jsonl", "").split("_")[-1]
                is_one_based = model_raw.lower() in ONE_BASED_MODELS

                with open(fpath) as f:
                    for line in f:
                        item = json.loads(line)
                        golden = item.get("GOLDEN_ANSWERS", [])
                        all_contexts = item.get("ALL_CONTEXTS", [])
                        selected_indices = item.get("SELECTED_CONTEXTS_INDICES", [])

                        if isinstance(selected_indices, list) and selected_indices:
                            adjusted = [i - 1 if is_one_based else i for i in selected_indices]
                            selected = [all_contexts[i] for i in adjusted if 0 <= i < len(all_contexts)]
                        else:
                            selected = all_contexts[:10]

                        rerank = compute_rerank_metrics(golden, selected)
                        for metric_key, value in rerank.items():
                            scores[task_name][subject][model_raw][metric_key].append(value)

    return scores


def print_table(scores, metric="Recall@10"):
    """Print a formatted table for a given metric."""
    tasks = ["Multi-Hop", "NC", "CC", "SSLI", "Base"]

    # Collect all reranker names
    rerankers = set()
    for task in scores:
        for subject in scores[task]:
            for reranker in scores[task][subject]:
                rerankers.add(reranker)

    print(f"\n{'=' * 100}")
    print(f"Metric: {metric}")
    print(f"{'=' * 100}")

    for reranker in sorted(rerankers):
        print(f"\n  Reranker: {reranker}")
        print(f"  {'Subject':<10} {'Multi-Hop':<18} {'NC':<18} {'CC':<18} {'SSLI':<18} {'Base':<18}")
        print(f"  {'-' * 90}")

        for subject in SUBJECTS_ORDER:
            row = [SUBJECT_SHORT.get(subject, subject)]
            for task in tasks:
                vals = scores.get(task, {}).get(subject, {}).get(reranker, {}).get(metric, [])
                if vals:
                    mean, std = grouped_std(vals)
                    row.append(f"{mean:.2f} +/- {std:.2f}")
                else:
                    row.append("--")
            print(f"  {row[0]:<10} {row[1]:<18} {row[2]:<18} {row[3]:<18} {row[4]:<18} {row[5]:<18}")


def save_results(scores, output_path):
    """Save all results to JSON."""
    result = {}
    for task in scores:
        result[task] = {}
        for subject in scores[task]:
            result[task][subject] = {}
            for reranker in scores[task][subject]:
                result[task][subject][reranker] = {}
                for metric_key in scores[task][subject][reranker]:
                    vals = scores[task][subject][reranker][metric_key]
                    mean, std = grouped_std(vals)
                    result[task][subject][reranker][metric_key] = {
                        "mean": round(mean, 4),
                        "std": round(std, 4),
                        "n": len(vals),
                    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--metric", type=str, default="Recall@10",
                        choices=["Recall@5", "Recall@10", "AP", "MRR"])
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print(f"Scanning: {args.data_root}")
    scores = discover_and_evaluate(args.data_root)
    print_table(scores, metric=args.metric)

    if args.output:
        save_results(scores, args.output)


if __name__ == "__main__":
    main()
