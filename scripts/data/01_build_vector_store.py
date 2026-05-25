"""
SciRerankBench - Build Qdrant Vector Store

Indexes OpenAlex abstracts with BGE-M3 embeddings and stores them in a Qdrant
collection using LangChain for dense retrieval.

Input:  dataset/source/2024_{subject}_papers.jsonl
Output: Qdrant collection (default URL: http://localhost:6333)

Usage:
    python scripts/data/01_build_vector_store.py --subject biology
    python scripts/data/01_build_vector_store.py --all
    python scripts/data/01_build_vector_store.py --all --qdrant-url http://your-qdrant:6333
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None,
                        choices=["biology", "chemistry", "geology", "physics", "math", None])
    parser.add_argument("--all", action="store_true", help="Build for all subjects")
    parser.add_argument("--source_dir", type=str, default="./dataset/source")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333",
                        help="Qdrant server URL")
    parser.add_argument("--qdrant-api-key", type=str, default=None,
                        help="Qdrant API key (for cloud deployments)")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def load_abstracts(source_path):
    """Load abstracts from JSONL file."""
    abstracts = []
    metadata = []
    with open(source_path) as f:
        for line in f:
            paper = json.loads(line)
            abstract = paper.get("abstract", "")
            title = paper.get("title", "")
            if abstract and len(abstract) > 50:
                abstracts.append(abstract)
                metadata.append({
                    "title": title,
                    "abstract": abstract,
                })
    return abstracts, metadata


def build_subject(subject, args):
    """Build Qdrant collection for one subject using LangChain."""
    from sentence_transformers import SentenceTransformer
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.documents import Document

    source_path = os.path.join(args.source_dir, f"2024_{subject}_papers.jsonl")
    if not os.path.exists(source_path):
        print(f"[SKIP] Source not found: {source_path}")
        return

    collection_name = f"scirerank_{subject}"

    print(f"\n{'=' * 60}")
    print(f"Building Qdrant collection for: {subject}")
    print(f"Source: {source_path}")
    print(f"Qdrant URL: {args.qdrant_url}")

    abstracts, metadata = load_abstracts(source_path)
    print(f"Loaded {len(abstracts)} valid abstracts")

    if not abstracts:
        print("[SKIP] No valid abstracts")
        return

    # Build LangChain Documents
    docs = [
        Document(page_content=abstract, metadata={"title": meta["title"]})
        for abstract, meta in zip(abstracts, metadata)
    ]

    # Load embedding model
    device = f"cuda:{args.gpu}"
    print(f"Loading embedding model: {args.embedding_model} on {device}")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Test embed to get dimension
    test_emb = embeddings.embed_query("test")
    dim = len(test_emb)
    print(f"Embedding dimension: {dim}")

    # Create Qdrant vector store (auto-creates collection)
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        collection_name=collection_name,
        batch_size=args.batch_size,
        force_recreate=True,  # recreate collection if exists
    )

    print(f"Indexed {len(abstracts)} vectors into Qdrant collection '{collection_name}'")


def main():
    args = parse_args()
    subjects = ["biology", "chemistry", "geology", "physics", "math"] if args.all else [args.subject]
    for subj in subjects:
        build_subject(subj, args)


if __name__ == "__main__":
    main()
