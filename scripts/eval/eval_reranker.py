"""
SciRerankBench - Main Evaluation Script

Evaluates rerankers + LLMs on scientific QA tasks.
For each question, the reranker selects top-K contexts from 100 candidates,
then the LLM generates an answer based on those contexts.

NOTE: Different reranker models require different conda environments.
      Use scripts/eval/run_eval.sh to auto-select the correct env,
      or see docs/ENVIRONMENTS.md for the model→env mapping.

Usage:
    # Auto-dispatch (recommended):
    bash scripts/eval/run_eval.sh --model bge --subject biology --task nc --llm Qwen-72B --gpu 0

    # Manual (activate correct env first):
    conda activate rerank
    python scripts/eval/eval_reranker.py --model bge --subject biology --task nc --llm Qwen-72B --gpu 0
"""

import argparse
import json
import os
import time
from collections import defaultdict
from itertools import islice

import torch
from tqdm import tqdm

try:
    import nltk
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

# Model name mapping (display name)
TO_MODEL_NAME = {
    "bce": "BCE", "jina": "Jina", "bge": "BGE", "gpt": "GPT",
    "t5": "T5", "colbert": "ColBERT", "flashrank": "MiniLM",
    "cross-encoder": "MXBAI", "gte": "GTE", "splade": "SPLADE",
    "rankt5": "RankT5", "listt5": "ListT5", "llm2vec": "LLM2Vec",
    "rankgpt": "RankGPT", "twolar": "TwoLAR", "rearank": "Rearank",
}

SUBJECTS = ["biology", "chemistry", "geology", "physics", "math"]
TASKS = ["nc", "base", "cc", "ssli", "multihop"]

# Prompt template for LLM answer generation
INFER_PROMPT = """\
**Instructions:**

You are given a question and a set of contexts. Your task is to provide a clear and concise answer to the question based on the contexts provided.

Answer based **only** on the contexts. If none are relevant, say: **No Answer Present.**

Keep your answer short -- ideally within 2 sentences. Do not include references or restate the question.

---
QUESTION: {QUESTION}
CONTEXTS: {CONTEXTS}

###
ANSWER:
"""


def parse_args():
    parser = argparse.ArgumentParser(description="SciRerankBench Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Reranker model name")
    parser.add_argument("--subject", type=str, required=True, choices=SUBJECTS)
    parser.add_argument("--task", type=str, default="nc", choices=TASKS)
    parser.add_argument("--llm", type=str, required=True, help="LLM backend name")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--top_k_first", type=int, default=100)
    parser.add_argument("--top_k_second", type=int, default=10)
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--pool_input", action="store_true",
                        help="Read from dataset/pools/{task}/{subject}/pool.jsonl instead of pre-built files")
    parser.add_argument("--max_chunks", type=int, default=None, help="Limit chunks for testing")
    return parser.parse_args()


def get_input_path(args):
    """Get the input QA JSONL path from the dataset or pool directory."""
    if args.pool_input:
        return os.path.join(args.data_root, "pools", args.task, args.subject, "pool.jsonl")
    if args.task == "nc":
        return os.path.join(args.data_root, "nc", args.subject, f"rebuild_5-95_{args.top_k_first}-{args.top_k_second}_BCE.jsonl")
    elif args.task == "base":
        return os.path.join(args.data_root, "base", args.subject, f"rebuild_100-0_{args.top_k_first}-{args.top_k_second}_BCE.jsonl")
    elif args.task == "cc":
        return os.path.join(args.data_root, "cc", args.subject, f"rebuild_{args.top_k_first}-{args.top_k_second}_BCE.jsonl")
    elif args.task == "ssli":
        return os.path.join(args.data_root, "ssli", args.subject, f"rebuild_{args.top_k_first}-{args.top_k_second}_BCE.jsonl")
    elif args.task == "multihop":
        return os.path.join(args.data_root, "multihop", args.subject, f"rebuild_{args.top_k_first}-{args.top_k_second}_BCE.jsonl")


def get_output_path(args):
    """Get the output JSONL path."""
    model_display = TO_MODEL_NAME.get(args.model, args.model)
    if args.task == "nc":
        return os.path.join(args.data_root, "nc", args.subject,
                            f"rebuild_5-95_{args.top_k_first}-{args.top_k_second}_{model_display}.jsonl")
    elif args.task == "base":
        return os.path.join(args.data_root, "base", args.subject,
                            f"rebuild_100-0_{args.top_k_first}-{args.top_k_second}_{model_display}.jsonl")
    else:
        return os.path.join(args.data_root, args.task, args.subject,
                            f"rebuild_{args.top_k_first}-{args.top_k_second}_{model_display}.jsonl")


def read_chunks(file_path, chunk_size=100):
    """Read file in chunks for memory efficiency."""
    with open(file_path, "r") as file:
        while True:
            chunk = list(islice(file, chunk_size))
            if not chunk:
                break
            yield chunk


# ===================== Metrics =====================


def contain_answer(golden_answers, context, threshold=0.6):
    """Check if context contains the golden answer via soft token overlap."""
    if not golden_answers or not golden_answers[0]:
        return 0.0
    ans_tokens = golden_answers[0].lower().split()
    ctx_tokens = set(context.lower().split())
    filtered = [t for t in ans_tokens if t not in STOP_WORDS]
    if not filtered:
        return 0.0
    return sum(1 for t in filtered if t in ctx_tokens) / len(filtered)


def compute_rerank_metrics(golden_answers, context_list):
    """Compute AP, MRR, Recall@5, Recall@10 for reranking results."""
    relevant_indices = []
    first_hit_rank = None
    hits = 0
    precision_sum = 0.0

    for i, ctx in enumerate(context_list):
        if contain_answer(golden_answers, ctx) >= 0.6:
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

    return {
        "Recall@5": round(recall_5, 2), "Recall@10": round(recall_10, 2),
        "AP": round(ap, 2), "MRR": round(mrr, 4),
    }


def compute_f1_score(prediction, ground_truths):
    """Compute token-level F1, precision, recall."""
    final = {"f1": 0, "precision": 0, "recall": 0}
    tokens_pred = prediction.lower().split()
    tokens_true = ground_truths[0].lower().split()
    subset = len(set(tokens_true) & set(tokens_pred))
    if not tokens_true:
        return final
    precision = subset / len(tokens_pred) * 100 if tokens_pred else 0
    recall = subset / len(tokens_true) * 100
    f1 = (2 * precision * recall) / (precision + recall) * 100 if (precision + recall) > 0 else 0
    final["f1"] = round(f1, 2)
    final["recall"] = round(recall, 2)
    final["precision"] = round(precision, 2)
    return final


# ===================== Reranker Functions =====================


def rerank_passages(reranker, model_name, query, contexts_list, top_k, device):
    """Unified reranking interface for all models."""
    if model_name == "bce":
        results = reranker.rerank(query, contexts_list)
        top_k_passages = results["rerank_passages"][:top_k]
        top_k_ids = results["rerank_ids"][:top_k]
    elif model_name == "jina":
        rankings = reranker.rank(query, contexts_list, return_documents=True, show_progress_bar=False)
        top_k_passages = [r["text"] for r in rankings[:top_k]]
        top_k_ids = [r["corpus_id"] for r in rankings[:top_k]]
    elif model_name == "bge":
        query_doc_pairs = [[query, doc] for doc in contexts_list]
        scores = reranker.compute_score(query_doc_pairs)
        ranked = sorted([{"corpus_id": i, "score": scores[i], "text": contexts_list[i]} for i in range(len(scores))],
                        key=lambda x: x["score"], reverse=True)[:top_k]
        top_k_passages = [r["text"] for r in ranked]
        top_k_ids = [r["corpus_id"] for r in ranked]
    elif model_name in ["t5", "colbert", "flashrank", "cross-encoder"]:
        results = reranker.rank(query=query, docs=contexts_list)
        top_k_passages = [res.document.text for res in results.top_k(top_k)]
        top_k_ids = [res.document.doc_id for res in results.top_k(top_k)]
    elif model_name == "gte":
        reranker["model"].eval()
        with torch.no_grad():
            inputs = reranker["tokenizer"]([[query, doc] for doc in contexts_list],
                                           padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = reranker["model"](**inputs, return_dict=True).logits.view(-1).float()
        ranked = sorted([{"corpus_id": i, "score": scores[i], "text": contexts_list[i]} for i in range(len(scores))],
                        key=lambda x: x["score"], reverse=True)[:top_k]
        top_k_passages = [r["text"] for r in ranked]
        top_k_ids = [r["corpus_id"] for r in ranked]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return top_k_passages, top_k_ids


def load_reranker(model_name, device):
    """Load reranker model by name."""
    if model_name == "bce":
        from BCEmbedding import RerankerModel
        return RerankerModel(model_name_or_path="netease-youdao/bce-reranker-base_v1", device=device)
    elif model_name == "jina":
        from sentence_transformers import CrossEncoder
        return CrossEncoder("jinaai/jina-reranker-v2-base-multilingual",
                            automodel_args={"torch_dtype": "auto"}, trust_remote_code=True, device=device)
    elif model_name == "bge":
        from FlagEmbedding import FlagReranker
        return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device=device)
    elif model_name in ["t5", "colbert", "flashrank", "cross-encoder"]:
        from rerankers import Reranker
        return Reranker(model_name)
    elif model_name == "gte":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            "Alibaba-NLP/gte-reranker-modernbert-base", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-reranker-modernbert-base")
        return {"model": model.to(device), "tokenizer": tokenizer}
    else:
        raise ValueError(f"Unknown reranker model: {model_name}")


# ===================== Main =====================


def main():
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    print(f"Loading reranker: {args.model}")
    reranker = load_reranker(args.model, device)

    # Load LLM (implement your factory)
    llm_model = load_llm(args.llm, device)

    output_path = get_output_path(args)
    input_path = get_input_path(args)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        print(f"  Hint: Build context pools first with scripts/data/02_build_context_pools.py")
        return

    count = 0
    errors = []
    start_time = time.time()

    with open(output_path, "w", encoding="utf-8") as fout:
        for chunk_idx, chunk in enumerate(read_chunks(input_path, 100)):
            if args.max_chunks and chunk_idx >= args.max_chunks:
                break
            print(f"Processing chunk {chunk_idx}")
            for line in tqdm(chunk):
                item = json.loads(line)
                question = item.get("QUESTION", "")
                golden_answers = item.get("GOLDEN_ANSWERS", [])
                all_contexts = item.get("ALL_CONTEXTS", [])

                if not golden_answers:
                    continue

                try:
                    contexts_sampled, contexts_sampled_indices = rerank_passages(
                        reranker, args.model, question, all_contexts, args.top_k_second, device)
                except Exception as e:
                    errors.append(count)
                    print(f"Error reranking: {e}")
                    continue

                try:
                    prompt = INFER_PROMPT.format(QUESTION=question, CONTEXTS="\n".join(contexts_sampled))
                    llm_answer = llm_model.generate(prompt)
                except Exception as e:
                    print(f"Error generating answer: {e}")
                    continue

                f1_metric = compute_f1_score(llm_answer, golden_answers)
                rerank_metric = compute_rerank_metrics(golden_answers, contexts_sampled)

                final_result = {
                    "QUESTION": question,
                    "ALL_CONTEXTS": all_contexts,
                    "SELECTED_CONTEXTS_INDICES": contexts_sampled_indices,
                    "GOLDEN_ANSWERS": golden_answers,
                    "MODEL_RESULTS": [{
                        "MODEL": args.llm,
                        "RERANK_SCORE": rerank_metric,
                        "GENERATE_SCORE": f1_metric,
                        "LLM_ANSWER": llm_answer,
                    }],
                }
                count += 1
                fout.write(json.dumps(final_result) + "\n")

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete. {count} questions processed in {elapsed:.0f}s.")
    if errors:
        print(f"Errors: {len(errors)}")


def load_llm(llm_name, device):
    """Load LLM model. Implement your factory or use vLLM/OpenAI."""
    # See scripts/eval/llm_model.py for implementation
    raise NotImplementedError(
        f"LLM loading for '{llm_name}' not implemented. "
        "Implement your LLM factory or use scripts/eval/llm_model.py"
    )


if __name__ == "__main__":
    main()
