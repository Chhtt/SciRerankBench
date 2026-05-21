"""
SciRerankBench - Build FAISS Vector Store

Indexes OpenAlex abstracts with BGE-M3 embeddings for dense retrieval.
Replaces the original Qdrant-based pipeline with a local FAISS index.

Input:  dataset/source/2024_{subject}_papers.jsonl
Output: dataset/index/{subject}/faiss.index + metadata.json

Usage:
    python scripts/data/01_build_vector_store.py --subject biology
    python scripts/data/01_build_vector_store.py --all
"""

import argparse
import json
import os

import faiss
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default=None,
                        choices=["biology", "chemistry", "geology", "physics", "math", None])
    parser.add_argument("--all", action="store_true", help="Build for all subjects")
    parser.add_argument("--source_dir", type=str, default="./dataset/source")
    parser.add_argument("--index_dir", type=str, default="./dataset/index")
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


def encode_abstracts(abstracts, model_name, batch_size, device):
    """Encode abstracts using BGE-M3 via sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        abstracts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


def build_index(embeddings):
    """Build FAISS IndexFlatIP (inner product, for normalized vectors = cosine)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def build_subject(subject, args):
    """Build FAISS index for one subject."""
    source_path = os.path.join(args.source_dir, f"2024_{subject}_papers.jsonl")
    if not os.path.exists(source_path):
        print(f"[SKIP] Source not found: {source_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"Building index for: {subject}")
    print(f"Source: {source_path}")

    abstracts, metadata = load_abstracts(source_path)
    print(f"Loaded {len(abstracts)} valid abstracts")

    if not abstracts:
        print("[SKIP] No valid abstracts")
        return

    device = f"cuda:{args.gpu}"
    embeddings = encode_abstracts(abstracts, args.embedding_model, args.batch_size, device)
    print(f"Embedding shape: {embeddings.shape}")

    index = build_index(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={index.d}")

    # Save
    out_dir = os.path.join(args.index_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Saved to {out_dir}/ (faiss.index + metadata.json)")


def main():
    args = parse_args()
    subjects = ["biology", "chemistry", "geology", "physics", "math"] if args.all else [args.subject]
    for subj in subjects:
        build_subject(subj, args)


if __name__ == "__main__":
    main()
