"""
SciRerankBench - Build Context Pools

For each QA pair, builds a pool of 100 candidate passages using dense retrieval
(Qdrant index via LangChain) and task-specific composition.

Tasks:
  NC:     Qdrant top-5 + 95 random abstracts
  Base:   Qdrant top-100
  CC:     Qdrant top-90 + 10 counterfactual distractors (LangChain structured output)
  SSLI:   Qdrant top-90 + 10 semantically-similar irrelevant distractors (LangChain)
  Multi-Hop: Qdrant top-100 for multi-hop questions

Input:
  - Qdrant collection: scirerank_{subject}
  - dataset/qa_generated/single_hop/  (LMQG output)
  - dataset/qa_generated/multi_hop/   (Multi-hop QA output)

Output:
  dataset/pools/{task}/{subject}/pool.jsonl
  Each line: {QUESTION, ALL_CONTEXTS, GOLDEN_ANSWERS}

Usage:
    python scripts/data/02_build_context_pools.py --subject biology --task nc
    python scripts/data/02_build_context_pools.py --subject biology --task cc --llm mistral
    python scripts/data/02_build_context_pools.py --subject biology --task nc --qdrant-url http://your-qdrant:6333
"""

import argparse
import json
import os
import random
from pathlib import Path

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
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333",
                        help="Qdrant server URL")
    parser.add_argument("--qdrant_api_key", type=str, default=None,
                        help="Qdrant API key (for cloud deployments)")
    parser.add_argument("--qa_dir", type=str, default="./dataset/qa_generated")
    parser.add_argument("--output_dir", type=str, default="./dataset/pools")
    parser.add_argument("--llm", type=str, default="mistral",
                        help="LLM name for CC/SSLI distractor generation")
    parser.add_argument("--llm_base_url", type=str, default=None,
                        help="OpenAI-compatible base URL for LLM")
    parser.add_argument("--llm_api_key", type=str, default="",
                        help="API key for LLM")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pool_size", type=int, default=None,
                        help="Override context pool size (default 100)")
    return parser.parse_args()


# ===================== Qdrant Retrieval via LangChain =====================


def get_vector_store(url, api_key, collection_name, embedding_model, device):
    """Get a Qdrant vector store via LangChain."""
    from langchain_qdrant import QdrantVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = QdrantVectorStore(
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store, embeddings


def search_vectorstore(vector_store, query_text, top_k):
    """Search vector store for top-k similar abstracts. Returns list of Document objects."""
    results = vector_store.similarity_search_with_score(query_text, k=top_k)
    return results


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


def build_pool_for_question(vector_store, query_text, question, golden_answer, task_type, args, all_abstracts):
    """Build a context pool for one question."""
    pool_size = args.pool_size or TASK_CONTEXT_SIZE

    if task_type == "nc":
        results = search_vectorstore(vector_store, query_text, NC_RELEVANT)
        relevant = [doc.page_content for doc, _ in results]
        remaining = pool_size - len(relevant)
        if remaining > 0:
            random_abstracts = random.sample(
                [a for a in all_abstracts if a not in relevant],
                min(remaining, len(all_abstracts) - len(relevant)),
            )
            pool = relevant + random_abstracts
        else:
            pool = relevant[:pool_size]
        while len(pool) < pool_size:
            pool.append(random.choice(all_abstracts))
        return pool[:pool_size]

    elif task_type == "base":
        results = search_vectorstore(vector_store, query_text, BASE_RELEVANT)
        pool = [doc.page_content for doc, _ in results]
        while len(pool) < pool_size:
            pool.append(random.choice(all_abstracts))
        return pool[:pool_size]

    elif task_type == "cc":
        results = search_vectorstore(vector_store, query_text, CC_RELEVANT)
        retrieved = [doc.page_content for doc, _ in results]
        distractors = generate_counterfactual_distractors(
            question, golden_answer, retrieved, args, count=pool_size - len(retrieved)
        )
        pool = retrieved + distractors
        while len(pool) < pool_size:
            pool.append(random.choice(all_abstracts))
        return pool[:pool_size]

    elif task_type == "ssli":
        results = search_vectorstore(vector_store, query_text, SSLI_RELEVANT)
        retrieved = [doc.page_content for doc, _ in results]
        distractors = generate_ssli_distractors(
            question, golden_answer, retrieved, args, count=pool_size - len(retrieved)
        )
        pool = retrieved + distractors
        while len(pool) < pool_size:
            pool.append(random.choice(all_abstracts))
        return pool[:pool_size]

    elif task_type == "multihop":
        results = search_vectorstore(vector_store, query_text, pool_size)
        pool = [doc.page_content for doc, _ in results]
        while len(pool) < pool_size:
            pool.append(random.choice(all_abstracts))
        return pool[:pool_size]


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


def fetch_all_abstracts(vector_store):
    """Fetch all abstracts from the Qdrant collection for random sampling."""
    from qdrant_client import QdrantClient
    abstracts = []
    client = QdrantClient(url=vector_store._client.url)
    offset = None
    limit = 1000
    while True:
        batch = client.scroll(
            collection_name=vector_store.collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = batch
        for point in points:
            abstracts.append(point.payload.get("page_content", ""))
        if next_offset is None:
            break
        offset = next_offset
    return abstracts


def main():
    args = parse_args()
    random.seed(42)

    collection_name = f"scirerank_{args.subject}"
    device = f"cuda:{args.gpu}"

    # Initialize vector store
    print(f"Connecting to Qdrant collection: {collection_name}")
    vector_store, _ = get_vector_store(
        args.qdrant_url, args.qdrant_api_key, collection_name,
        "BAAI/bge-m3", device
    )

    # Fetch all abstracts for random sampling
    print("Fetching all abstracts for random sampling...")
    all_abstracts = fetch_all_abstracts(vector_store)
    print(f"Fetched {len(all_abstracts)} abstracts")

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
    output_path = os.path.join(args.output_dir, args.task, args.subject, "pool.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    with open(output_path, "w") as out_f:
        for qa in tqdm(qa_pairs, desc=f"Building {args.task} pool for {args.subject}"):
            question = qa["question"]
            golden_answer = qa["golden_answer"]
            if not question or not golden_answer:
                continue

            pool = build_pool_for_question(
                vector_store, question, question, golden_answer,
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
