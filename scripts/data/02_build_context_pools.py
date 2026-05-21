"""
SciRerankBench - Build Context Pools

For each QA pair, builds a pool of 100 candidate passages using dense retrieval
(L1 FAISS index) and task-specific composition.

Tasks:
  NC:     FAISS top-5 + 95 random abstracts
  Base:   FAISS top-100
  CC:     FAISS top-90 + 10 counterfactual distractors (LangChain structured output)
  SSLI:   FAISS top-90 + 10 semantically-similar irrelevant distractors (LangChain)
  Multi-Hop: FAISS top-100 for multi-hop questions

Input:
  - dataset/index/{subject}/          (FAISS index + metadata)
  - dataset/qa_generated/single_hop/  (LMQG output)
  - dataset/qa_generated/multi_hop/   (Multi-hop QA output)

Output:
  dataset/pools/{task}/{subject}/pool.jsonl
  Each line: {QUESTION, ALL_CONTEXTS, GOLDEN_ANSWERS}

Usage:
    python scripts/data/02_build_context_pools.py --subject biology --task nc
    python scripts/data/02_build_context_pools.py --subject biology --task cc --llm mistral
"""

import argparse
import json
import os
import random
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

TASK_CONTEXT_SIZE = 100
NC_RELEVANT = 5
CC_RELEVANT = 90
SSLI_RELEVANT = 90
BASE_RELEVANT = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True,
                        choices=["biology", "chemistry", "geology", "physics", "math"])
    parser.add_argument("--task", type=str, required=True,
                        choices=["nc", "base", "cc", "ssli", "multihop"])
    parser.add_argument("--index_dir", type=str, default="./dataset/index")
    parser.add_argument("--qa_dir", type=str, default="./dataset/qa_generated")
    parser.add_argument("--output_dir", type=str, default="./dataset/pools")
    parser.add_argument("--llm", type=str, default="mistral",
                        help="LLM name for CC/SSLI distractor generation")
    parser.add_argument("--llm_base_url", type=str, default=None,
                        help="OpenAI-compatible base URL for LLM")
    parser.add_argument("--llm_api_key", type=str, default="",
                        help="API key for LLM")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


# ===================== FAISS Retrieval =====================


def load_index(subject, index_dir):
    """Load FAISS index and metadata for a subject."""
    idx_path = os.path.join(index_dir, subject, "faiss.index")
    meta_path = os.path.join(index_dir, subject, "metadata.json")
    index = faiss.read_index(idx_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return index, metadata


def search_faiss(index, query_embedding, top_k):
    """Search FAISS index for top-k similar abstracts. Returns list of abstract texts."""
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return indices[0].tolist(), scores[0].tolist()


def encode_query(query, model_name, device):
    """Encode a single query string."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return emb[0]


# ===================== Distractor Generation (CC / SSLI) =====================


def get_llm_client(args):
    """Create an OpenAI-compatible LLM client via LangChain."""
    from langchain_openai import ChatOpenAI
    client = ChatOpenAI(
        model=args.llm,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key or "dummy",
        temperature=0.7,
    )
    return client


def generate_counterfactual_distractors(question, golden_answer, retrieved_contexts, args, count=10):
    """Generate counterfactual passages using LangChain structured output."""
    from pydantic import BaseModel, Field

    class CounterfactualPassage(BaseModel):
        passage_text: str = Field(description="A plausible but factually incorrect passage about the topic")
        modification: str = Field(description="What factual claim was changed from the original")

    class CounterfactualResponse(BaseModel):
        distractors: list[CounterfactualPassage] = Field(description=f"List of {count} counterfactual passages")

    context_sample = "\n".join(retrieved_contexts[:3])
    prompt = f"""You are generating counterfactual passages for a scientific QA benchmark.

Given a question, its correct answer, and some real scientific context, generate {count} passages that:
1. Are topically relevant and sound scientifically plausible
2. Contain specific factual errors that CONTRADICT the golden answer
3. Use academic writing style similar to real abstracts

QUESTION: {question}
GOLDEN ANSWER: {golden_answer}
REAL CONTEXT (for style reference):
{context_sample}

Generate {count} counterfactual passages. Each should be 2-4 sentences, formatted like a scientific abstract excerpt."""

    from langchain_core.prompts import ChatPromptTemplate
    structured_llm = args.llm_client.with_structured_output(CounterfactualResponse)
    chain = ChatPromptTemplate.from_messages([("system", prompt)]) | structured_llm
    result = chain.invoke({})

    return [d.passage_text for d in result.distractors]


def generate_ssli_distractors(question, golden_answer, retrieved_contexts, args, count=10):
    """Generate semantically similar but logically irrelevant passages."""
    from pydantic import BaseModel, Field

    class SSLIPassage(BaseModel):
        passage_text: str = Field(description="A passage that is topically similar but does not answer the question")
        irrelevance_reason: str = Field(description="Why this passage is logically irrelevant to the question")

    class SSLIResponse(BaseModel):
        distractors: list[SSLIPassage] = Field(description=f"List of {count} semantically similar but irrelevant passages")

    context_sample = "\n".join(retrieved_contexts[:3])
    prompt = f"""You are generating distractor passages for a scientific QA benchmark.

Given a question and its correct answer, generate {count} passages that:
1. Share keywords and topics with the question (semantically similar)
2. Do NOT actually answer the question (logically irrelevant)
3. Discuss related but tangential aspects of the topic
4. Use academic writing style similar to real abstracts

QUESTION: {question}
GOLDEN ANSWER: {golden_answer}
REAL CONTEXT (for style reference):
{context_sample}

Generate {count} distractor passages. Each should be 2-4 sentences."""

    from langchain_core.prompts import ChatPromptTemplate
    structured_llm = args.llm_client.with_structured_output(SSLIResponse)
    chain = ChatPromptTemplate.from_messages([("system", prompt)]) | structured_llm
    result = chain.invoke({})

    return [d.passage_text for d in result.distractors]


# ===================== Pool Building =====================


def build_pool_for_question(index, metadata, query_emb, question, golden_answer, task_type, args, all_abstracts):
    """Build a 100-context pool for one question."""
    if task_type == "nc":
        indices, _ = search_faiss(index, query_emb, NC_RELEVANT)
        relevant = [metadata[i]["abstract"] for i in indices if i < len(metadata)]
        # Fill remaining with random
        pool_size = TASK_CONTEXT_SIZE
        remaining = pool_size - len(relevant)
        if remaining > 0:
            random_abstracts = random.sample(
                [a for a in all_abstracts if a not in relevant],
                min(remaining, len(all_abstracts) - len(relevant)),
            )
            pool = relevant + random_abstracts
        else:
            pool = relevant[:pool_size]
        # Pad if still short
        while len(pool) < pool_size:
            pool.append(random.choice(all_abstracts))
        return pool[:pool_size]

    elif task_type == "base":
        indices, _ = search_faiss(index, query_emb, BASE_RELEVANT)
        pool = [metadata[i]["abstract"] for i in indices if i < len(metadata)]
        while len(pool) < TASK_CONTEXT_SIZE:
            pool.append(random.choice(all_abstracts))
        return pool[:TASK_CONTEXT_SIZE]

    elif task_type == "cc":
        indices, _ = search_faiss(index, query_emb, CC_RELEVANT)
        retrieved = [metadata[i]["abstract"] for i in indices if i < len(metadata)]
        distractors = generate_counterfactual_distractors(
            question, golden_answer, retrieved, args, count=TASK_CONTEXT_SIZE - len(retrieved)
        )
        pool = retrieved + distractors
        while len(pool) < TASK_CONTEXT_SIZE:
            pool.append(random.choice(all_abstracts))
        return pool[:TASK_CONTEXT_SIZE]

    elif task_type == "ssli":
        indices, _ = search_faiss(index, query_emb, SSLI_RELEVANT)
        retrieved = [metadata[i]["abstract"] for i in indices if i < len(metadata)]
        distractors = generate_ssli_distractors(
            question, golden_answer, retrieved, args, count=TASK_CONTEXT_SIZE - len(retrieved)
        )
        pool = retrieved + distractors
        while len(pool) < TASK_CONTEXT_SIZE:
            pool.append(random.choice(all_abstracts))
        return pool[:TASK_CONTEXT_SIZE]

    elif task_type == "multihop":
        indices, _ = search_faiss(index, query_emb, TASK_CONTEXT_SIZE)
        pool = [metadata[i]["abstract"] for i in indices if i < len(metadata)]
        while len(pool) < TASK_CONTEXT_SIZE:
            pool.append(random.choice(all_abstracts))
        return pool[:TASK_CONTEXT_SIZE]


def load_single_hop_qa(subject, qa_dir):
    """Load LMQG-generated single-hop QA pairs."""
    qa_path = os.path.join(qa_dir, "single_hop", f"2024_{subject}_qa.jsonl")
    if not os.path.exists(qa_path):
        print(f"[SKIP] QA file not found: {qa_path}")
        return []

    questions = []
    with open(qa_path) as f:
        for line in f:
            item = json.loads(line)
            for qa in item.get("qa", []):
                if isinstance(qa, list) and len(qa) >= 2:
                    questions.append({
                        "question": qa[0],
                        "golden_answer": qa[1],
                    })
                elif isinstance(qa, dict):
                    questions.append({
                        "question": qa.get("question", ""),
                        "golden_answer": qa.get("answer", ""),
                    })
    return questions


def load_multi_hop_qa(subject, qa_dir):
    """Load multi-hop QA pairs."""
    qa_path = os.path.join(qa_dir, "multi_hop", f"2024_{subject}_qa.jsonl")
    if not os.path.exists(qa_path):
        print(f"[SKIP] Multi-hop QA file not found: {qa_path}")
        return []

    questions = []
    with open(qa_path) as f:
        for line in f:
            item = json.loads(line)
            for q in item.get("questions", []):
                questions.append({
                    "question": q.get("question", ""),
                    "golden_answer": q.get("answer", ""),
                })
    return questions


def main():
    args = parse_args()
    random.seed(42)

    # Load FAISS index
    index, metadata = load_index(args.subject, args.index_dir)
    print(f"Loaded FAISS index for {args.subject}: {index.ntotal} vectors")
    all_abstracts = [m["abstract"] for m in metadata]

    # Load QA pairs
    if args.task == "multihop":
        qa_pairs = load_multi_hop_qa(args.subject, args.qa_dir)
    else:
        qa_pairs = load_single_hop_qa(args.subject, args.qa_dir)

    print(f"Loaded {len(qa_pairs)} QA pairs")
    if not qa_pairs:
        return

    # Init LLM client for CC/SSLI
    if args.task in ("cc", "ssli"):
        args.llm_client = get_llm_client(args)
        print(f"Loaded LLM client: {args.llm}")

    # Build pools
    from sentence_transformers import SentenceTransformer
    enc_model = SentenceTransformer("BAAI/bge-m3", device=f"cuda:{args.gpu}")

    output_path = os.path.join(args.output_dir, args.task, args.subject, "pool.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(output_path, "w") as out_f:
        for qa in tqdm(qa_pairs, desc=f"Building {args.task} pool for {args.subject}"):
            question = qa["question"]
            golden_answer = qa["golden_answer"]
            if not question or not golden_answer:
                continue

            # Encode query
            query_emb = enc_model.encode(
                [question], normalize_embeddings=True, convert_to_numpy=True
            )[0]

            pool = build_pool_for_question(
                index, metadata, query_emb, question, golden_answer,
                args.task, args, all_abstracts
            )

            result = {
                "QUESTION": question,
                "ALL_CONTEXTS": pool,
                "GOLDEN_ANSWERS": [golden_answer],
            }
            out_f.write(json.dumps(result) + "\n")
            count += 1

    print(f"Built {count} context pools -> {output_path}")


if __name__ == "__main__":
    main()
