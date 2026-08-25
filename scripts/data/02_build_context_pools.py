"""
SciRerankBench - Build Context Pools

For each QA pair, builds a pool of 100 candidate passages using dense retrieval
(Qdrant index via LangChain) and task-specific composition.

Tasks:
  NC:     Qdrant top-5 + 95 random abstracts
  Base:   Qdrant top-100
  CC:     Qdrant top-90 + 10 counterfactual distractors (LLM entity rewrite of real passages)
  SSLI:   Qdrant top-90 + 10 semantically-similar irrelevant distractors (LLM rewrite + BGE embedding verification)
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


# ===================== CC: Counterfactual via Entity Rewrite =====================


def rewrite_counterfactual_passage(source_passage, question, golden_answer, llm_client):
    """Rewrite a real passage by replacing key factual entities to create a counterfactual.

    The rewritten passage preserves the academic style and structure of the original,
    but substitutes key entities (numbers, names, causal relations) with plausible
    alternatives that contradict the golden answer.
    """
    from pydantic import BaseModel, Field

    class CounterfactualPassage(BaseModel):
        passage_text: str = Field(
            description="The rewritten passage with key factual entities replaced"
        )
        entities_replaced: list[str] = Field(
            description="List of entities that were replaced"
        )

    prompt = f"""You are rewriting a scientific passage to create a counterfactual version for a QA benchmark.

Given a real scientific passage, a question, and the correct answer, REWRITE the passage by:
1. Identifying key factual entities in the passage (numbers, chemical names, causal relations, etc.)
2. Replacing them with PLAUSIBLE ALTERNATIVES that contradict the golden answer
3. Preserving the academic writing style, length, and structure of the original
4. Making the passage sound scientifically realistic

IMPORTANT: Do NOT generate a completely new passage. Only modify specific entities within the original text.

QUESTION: {question}
GOLDEN ANSWER: {golden_answer}
ORIGINAL PASSAGE:
{source_passage}

Rewrite the passage by replacing key entities. The result should read like a real scientific abstract excerpt."""

    from langchain_core.prompts import ChatPromptTemplate
    structured_llm = llm_client.with_structured_output(CounterfactualPassage)
    chain = ChatPromptTemplate.from_messages([("system", prompt)]) | structured_llm
    result = chain.invoke({})

    return result.passage_text


def generate_counterfactual_distractors(question, golden_answer, retrieved_contexts, args, count=10):
    """Generate counterfactual passages by rewriting real retrieved passages.

    Takes the first {count} passages from the retrieved contexts and rewrites
    each one by replacing key factual entities with plausible alternatives
    that contradict the golden answer.
    """
    distractors = []
    for i in range(min(count, len(retrieved_contexts))):
        source = retrieved_contexts[i]
        try:
            rewritten = rewrite_counterfactual_passage(
                source, question, golden_answer, args.llm_client
            )
            distractors.append(rewritten)
        except Exception as e:
            print(f"  [CC] Failed to rewrite passage {i}: {e}, skipping")
    return distractors


# ===================== SSLI: Semantic Rewrite + Embedding Verification =====================


def rewrite_ssli_passage(source_passage, question, golden_answer, llm_client):
    """Rewrite a real passage to be semantically similar but logically irrelevant.

    The rewritten passage shares keywords and topics with the question but
    does NOT actually answer the question — it discusses tangential aspects.
    """
    from pydantic import BaseModel, Field

    class SSLIPassage(BaseModel):
        passage_text: str = Field(
            description="The rewritten passage that shares keywords but does not answer the question"
        )

    prompt = f"""You are rewriting a scientific passage to create a distractor for a QA benchmark.

Given a real scientific passage, a question, and its correct answer, REWRITE the passage so that:
1. It shares KEYWORDS and TOPICS with the question (uses similar scientific terminology)
2. It does NOT actually answer the question — it discusses a tangential or related aspect
3. It preserves the academic writing style and length of the original passage
4. It sounds like a real scientific abstract excerpt

IMPORTANT: Rewrite the original passage, do NOT generate a completely new one. Keep the core subject matter but shift the focus away from answering the question.

QUESTION: {question}
GOLDEN ANSWER: {golden_answer}
ORIGINAL PASSAGE:
{source_passage}

Rewrite the passage to be semantically similar but logically irrelevant."""

    from langchain_core.prompts import ChatPromptTemplate
    structured_llm = llm_client.with_structured_output(SSLIPassage)
    chain = ChatPromptTemplate.from_messages([("system", prompt)]) | structured_llm
    result = chain.invoke({})

    return result.passage_text


def check_ssli_embedding(question, rewritten_text, embedding_model, threshold=0.75):
    """Verify that the rewritten passage has high cosine similarity to the question.

    Uses the BGE embedding model to compute cosine similarity between the question
    and the rewritten passage. Returns True if similarity ≥ threshold.
    """
    import numpy as np

    embeddings = embedding_model.embed_documents([question, rewritten_text])
    q_vec = np.array(embeddings[0])
    r_vec = np.array(embeddings[1])

    cosine_sim = float(np.dot(q_vec, r_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(r_vec)))
    return cosine_sim >= threshold, cosine_sim


def generate_ssli_distractors(question, golden_answer, retrieved_contexts, args, count=10):
    """Generate SSLI distractors by rewriting real passages and verifying embedding similarity.

    For each source passage from the retrieved contexts:
    1. LLM rewrites it to share keywords but not answer the question
    2. BGE embedding model verifies cosine similarity ≥ 0.75
    3. If verified, accept; otherwise retry with the next passage
    """
    distractors = []
    max_attempts = count * 3  # Allow up to 3 attempts per desired distractor
    attempt = 0

    for source in retrieved_contexts:
        if len(distractors) >= count:
            break
        if attempt >= max_attempts:
            break

        try:
            rewritten = rewrite_ssli_passage(
                source, question, golden_answer, args.llm_client
            )
            passed, sim = check_ssli_embedding(
                question, rewritten, args.embedding_model
            )
            if passed:
                distractors.append(rewritten)
            else:
                pass  # Will retry with next source passage
        except Exception as e:
            print(f"  [SSLI] Failed attempt {attempt + 1}: {e}")

        attempt += 1

    if len(distractors) < count:
        print(f"  [SSLI] Warning: only generated {len(distractors)}/{count} distractors for this question")

    return distractors


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


def fetch_all_abstracts(url, collection_name):
    """Fetch all abstracts from the Qdrant collection for random sampling."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=url)
    offset = None
    limit = 1000
    while True:
        batch = client.scroll(
            collection_name=collection_name,
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
    vector_store, embeddings = get_vector_store(
        args.qdrant_url, args.qdrant_api_key, collection_name,
        "BAAI/bge-m3", device
    )

    # Store embedding model reference for SSLI verification
    args.embedding_model = embeddings

    # Fetch all abstracts for random sampling
    print("Fetching all abstracts for random sampling...")
    all_abstracts = fetch_all_abstracts(args.qdrant_url, collection_name)
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
