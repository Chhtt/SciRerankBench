#!/bin/bash
# SciRerankBench - Data Construction Pipeline
#
# 3 steps:
#   1. Build FAISS vector store from OpenAlex abstracts
#   2. Build context pools for each task type
#   3. Run rerankers on each pool to produce final JSONL files
#
# Usage:
#   bash scripts/data/run_pipeline.sh --subject biology --tasks nc,base,cc,ssli,multihop
#   bash scripts/data/run_pipeline.sh --all --tasks nc --rerankers bge,jina
#   bash scripts/data/run_pipeline.sh --subject biology --task cc --llm mistral --llm-base-url http://localhost:8000/v1

set -e

# Defaults
SUBJECTS=()
TASKS=()
RERANKERS=()
LLM="Qwen-72B"
LLM_BASE_URL=""
LLM_API_KEY=""
GPU=0
DATA_ROOT="./dataset"
INDEX_DIR="./dataset/index"
QA_DIR="./dataset/qa_generated"
POOL_DIR="./dataset/pools"
ALL=false
RERANKERS_ALL=true  # run all rerankers by default

while [[ $# -gt 0 ]]; do
    case $1 in
        --subject) SUBJECTS=("$2"); shift 2;;
        --all) ALL=true; shift;;
        --task) TASKS=("$2"); IFS=',' read -ra TASKS <<< "$2"; shift 2;;
        --tasks) IFS=',' read -ra TASKS <<< "$2"; shift 2;;
        --reranker) RERANKERS=("$2"); RERANKERS_ALL=false; shift 2;;
        --rerankers) IFS=',' read -ra RERANKERS <<< "$2"; RERANKERS_ALL=false; shift 2;;
        --llm) LLM="$2"; shift 2;;
        --llm-base-url) LLM_BASE_URL="$2"; shift 2;;
        --llm-api-key) LLM_API_KEY="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        --data_root) DATA_ROOT="$2"; shift 2;;
        --skip-build-index) SKIP_INDEX=true; shift;;
        --skip-pools) SKIP_POOLS=true; shift;;
        --skip-reranker) SKIP_RERANKER=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if $ALL; then
    SUBJECTS=("biology" "chemistry" "geology" "physics" "math")
fi

if [ ${#SUBJECTS[@]} -eq 0 ]; then
    echo "Usage: bash scripts/data/run_pipeline.sh --subject <subj> | --all [--tasks nc,base,cc,ssli,multihop]"
    exit 1
fi

if [ ${#TASKS[@]} -eq 0 ]; then
    TASKS=("nc" "base" "cc" "ssli" "multihop")
fi

if $RERANKERS_ALL; then
    RERANKERS=("bge" "jina" "bce" "t5" "colbert" "flashrank" "cross-encoder" "gte" "rankt5" "listt5" "splade" "llm2vec" "rankgpt" "twolar" "rearank")
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EVAL_SCRIPT="$SCRIPT_DIR/eval/eval_reranker.py"

# ===================== Step 1: Build FAISS Index =====================
if [ "${SKIP_INDEX}" != "true" ]; then
    echo "=========================================="
    echo "Step 1: Building FAISS vector store"
    echo "=========================================="
    conda activate rerank
    for subj in "${SUBJECTS[@]}"; do
        python "$SCRIPT_DIR/data/01_build_vector_store.py" --subject "$subj" --gpu "$GPU" --index_dir "$INDEX_DIR"
    done
else
    echo "Step 1: Skipped (--skip-build-index)"
fi

# ===================== Step 2: Build Context Pools =====================
if [ "${SKIP_POOLS}" != "true" ]; then
    echo "=========================================="
    echo "Step 2: Building context pools"
    echo "=========================================="
    conda activate rerank
    for subj in "${SUBJECTS[@]}"; do
        for task in "${TASKS[@]}"; do
            extra_args=""
            if [[ "$task" == "cc" || "$task" == "ssli" ]]; then
                extra_args="--llm $LLM --llm-base-url $LLM_BASE_URL --llm-api-key $LLM_API_KEY"
            fi
            python "$SCRIPT_DIR/data/02_build_context_pools.py" \
                --subject "$subj" --task "$task" --gpu "$GPU" \
                --index_dir "$INDEX_DIR" --qa_dir "$QA_DIR" --output_dir "$POOL_DIR" \
                $extra_args
        done
    done
else
    echo "Step 2: Skipped (--skip-pools)"
fi

# ===================== Step 3: Run Rerankers =====================
if [ "${SKIP_RERANKER}" != "true" ]; then
    echo "=========================================="
    echo "Step 3: Running rerankers"
    echo "=========================================="

    # Map model to conda environment
    get_env_for_model() {
        local model_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
        case "$model_lower" in
            bge|jina|cross-encoder|flashrank|colbert|t5|gte)
                echo "rerank"
                ;;
            bce)
                echo "bce"
                ;;
            rankt5|listt5|splade|twolar|llm2vec|rankgpt)
                echo "rankify"
                ;;
            rearank)
                echo "rearank"
                ;;
            *)
                echo "rerank"
                ;;
        esac
    }

    for subj in "${SUBJECTS[@]}"; do
        for task in "${TASKS[@]}"; do
            pool_file="$POOL_DIR/$task/$subj/pool.jsonl"
            if [ ! -f "$pool_file" ]; then
                echo "[SKIP] Pool not found: $pool_file"
                continue
            fi

            for model in "${RERANKERS[@]}"; do
                env_name=$(get_env_for_model "$model")
                echo "----------------------------------------"
                echo "  Model: $model | Subject: $subj | Task: $task | Env: $env_name"
                echo "----------------------------------------"

                eval "$(conda shell.bash hook)"
                conda activate "$env_name"

                python "$EVAL_SCRIPT" \
                    --model "$model" \
                    --subject "$subj" \
                    --task "$task" \
                    --llm "$LLM" \
                    --gpu "$GPU" \
                    --data_root "$DATA_ROOT" \
                    --pool_input
            done
        done
    done
else
    echo "Step 3: Skipped (--skip-reranker)"
fi

echo ""
echo "=========================================="
echo "Pipeline complete"
echo "=========================================="
