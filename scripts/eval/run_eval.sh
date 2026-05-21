#!/bin/bash
# SciRerankBench - Evaluation Runner
# Automatically selects the correct conda environment for each reranker model.
#
# Usage:
#   bash scripts/eval/run_eval.sh --model bge --subject biology --task nc --llm Qwen-72B --gpu 0

set -e

# Parse arguments
MODEL=""
SUBJECT=""
TASK="nc"
LLM=""
GPU=0
DATA_ROOT="./dataset"
TOP_K_FIRST=100
TOP_K_SECOND=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2;;
        --subject) SUBJECT="$2"; shift 2;;
        --task) TASK="$2"; shift 2;;
        --llm) LLM="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        --data_root) DATA_ROOT="$2"; shift 2;;
        --top_k_first) TOP_K_FIRST="$2"; shift 2;;
        --top_k_second) TOP_K_SECOND="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$SUBJECT" ] || [ -z "$LLM" ]; then
    echo "Usage: bash scripts/eval/run_eval.sh --model <reranker> --subject <subj> --llm <llm> [--task nc|base|cc|ssli|multihop] [--gpu 0]"
    exit 1
fi

# Map model to conda environment
MODEL_LOWER=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')

case "$MODEL_LOWER" in
    bge|jina|cross-encoder|flashrank|colbert|t5|gte)
        ENV="rerank"
        ;;
    bce)
        ENV="bce"
        ;;
    rankt5|listt5|splade|twolar|llm2vec|rankgpt)
        ENV="rankify"
        ;;
    rearank)
        ENV="rearank"
        ;;
    *)
        echo "Unknown model: $MODEL — defaulting to 'rerank' env"
        ENV="rerank"
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Model:    $MODEL"
echo "Subject:  $SUBJECT"
echo "Task:     $TASK"
echo "LLM:      $LLM"
echo "GPU:      $GPU"
echo "Env:      $ENV"
echo "=========================================="

# Activate environment and run
eval "$(conda shell.bash hook)"
conda activate "$ENV"

python "$SCRIPT_DIR/eval_reranker.py" \
    --model "$MODEL" \
    --subject "$SUBJECT" \
    --task "$TASK" \
    --llm "$LLM" \
    --gpu "$GPU" \
    --data_root "$DATA_ROOT" \
    --top_k_first "$TOP_K_FIRST" \
    --top_k_second "$TOP_K_SECOND"
