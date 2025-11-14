#!/bin/bash
# GSM8K Evaluation Script for Linux/Mac
# 
# Usage: ./run_gsm8k_eval.sh [model_path] [max_problems]
#
# Example:
#   ./run_gsm8k_eval.sh models/llama3_2_1b_f32.gguf 100
#

MODEL_PATH="$1"
MAX_PROBLEMS="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${SCRIPT_DIR}/gsm8k_test.jsonl"

# Set defaults if not provided
if [ -z "$MODEL_PATH" ]; then
    echo "Error: Please provide model path as first argument"
    echo "Usage: $0 model_path [max_problems]"
    echo "Example: $0 models/llama3_2_1b_f32.gguf 100"
    exit 1
fi

if [ -z "$MAX_PROBLEMS" ]; then
    MAX_PROBLEMS="all"
    MAX_FLAG=""
else
    MAX_FLAG="--max-problems ${MAX_PROBLEMS}"
fi

# Try to find the executable
EXE_PATH=""
if [ -f "${SCRIPT_DIR}/../../build/bin/gsm8k-eval" ]; then
    EXE_PATH="${SCRIPT_DIR}/../../build/bin/gsm8k-eval"
elif [ -f "${SCRIPT_DIR}/gsm8k-eval" ]; then
    EXE_PATH="${SCRIPT_DIR}/gsm8k-eval"
else
    echo "Error: Could not find gsm8k-eval executable"
    echo "Please build the project first:"
    echo "  mkdir build"
    echo "  cd build"
    echo "  cmake .."
    echo "  cmake --build . --config Release"
    exit 1
fi

echo "====================================="
echo "GSM8K Evaluation"
echo "====================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Max Problems: ${MAX_PROBLEMS}"
echo "Executable: ${EXE_PATH}"
echo "====================================="
echo ""

# Run the evaluation
"${EXE_PATH}" -m "${MODEL_PATH}" -f "${DATASET_PATH}" ${MAX_FLAG}

