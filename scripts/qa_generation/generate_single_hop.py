"""
SciRerankBench - Single-Hop QA Generation

Uses LMQG (Language Model-based Question Generation) framework to generate
single-hop QA pairs from scientific abstracts.

Model: t5-small-squad-qag (via lmqg library)

Usage:
    python generate_single_hop.py --subject biology --data_dir ./dataset/source
"""

import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True,
                        choices=["biology", "chemistry", "geology", "physics", "math"])
    parser.add_argument("--data_dir", type=str, default="./dataset/source")
    parser.add_argument("--output_dir", type=str, default="./dataset/qa_generated/single_hop")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load LMQG model
    try:
        from lmqg import TransformersQG
    except ImportError:
        print("Please install lmqg: pip install lmqg")
        return

    model_path = "t5-small-squad-qag"
    print(f"Loading LMQG model: {model_path}")
    model = TransformersQG(language="en", model=model_path)

    input_path = os.path.join(args.data_dir, f"2024_{args.subject}_papers.jsonl")
    output_path = os.path.join(args.output_dir, f"2024_{args.subject}_qa.jsonl")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    count = 0
    with open(input_path, "r") as f, open(output_path, "w") as out_f:
        for line in tqdm(f, desc=f"Generating QA for {args.subject}"):
            paper = json.loads(line)
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            if not abstract or len(abstract) < 50:
                continue

            try:
                qa_pairs = model.generate_qa(abstract)
            except Exception as e:
                print(f"Error on abstract: {e}")
                continue

            output = {
                "title": title,
                "abstract": abstract,
                "qa": qa_pairs,
                "subject": args.subject,
            }
            out_f.write(json.dumps(output) + "\n")
            count += 1

    print(f"Generated {count} QA pairs for {args.subject}")


if __name__ == "__main__":
    main()
