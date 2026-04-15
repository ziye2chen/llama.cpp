#!/usr/bin/env bash
#
# Qwen3-8B 批量量化格式测试脚本
#
# 这个脚本专门用于批量测试以下 4 个量化 GGUF：
#   - Q5_0
#   - Q5_K_M
#   - Q6_K
#   - Q8_0
#
# 它会依次调用已经写好的通用测试脚本：
#   examples/zeroth-order-opt/test_quant_gguf_support.sh
#
# 每个量化格式都会独立生成：
#   - 日志目录
#   - verify 输出模型
#   - finetune merge 输出模型
#   - 可选 LoRA-only 输出
#
# 用法示例：
#   bash examples/zeroth-order-opt/test_qwen3_8b_q5_q6_batch.sh \
#     --model-dir /root/models/Qwen3-8B-GGUF \
#     --output-root /root/ZO/llama.cpp/test_outputs/qwen3_8b_batch_test \
#     --run-lora-only
#
# 默认假设模型文件名为：
#   Qwen3-8B-Q5_0.gguf
#   Qwen3-8B-Q5_K_M.gguf
#   Qwen3-8B-Q6_K.gguf
#   Qwen3-8B-Q8_0.gguf
#
set -euo pipefail

MODEL_DIR="/root/models/Qwen3-8B-GGUF"
DATASET_PATH=""
WIKI_PATH=""
RUN_LORA_ONLY=0
SKIP_BUILD=0
SKIP_TRAIN=0

usage() {
    cat <<'USAGE'
用法：
  bash examples/zeroth-order-opt/test_qwen3_8b_q5_q6_batch.sh \
    [--model-dir <模型目录>] \
    [--output-root <批量输出目录>]

可选参数：
  --model-dir       Qwen3-8B GGUF 所在目录
  --output-root     批量测试输出根目录；不传时默认写到仓库下 test_outputs/qwen3_8b_batch_test
  --dataset         微调数据文件路径，默认使用仓库里的 gsm8k_test.jsonl
  --wiki            perplexity 测试文本路径，默认使用仓库里的 wiki.test.raw
  --run-lora-only   额外执行 LoRA-only 验证
  --skip-build      跳过编译
  --skip-train      跳过最小微调 + merge 验证
  -h, --help        显示帮助
USAGE
}

OUTPUT_ROOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --wiki)
            WIKI_PATH="$2"
            shift 2
            ;;
        --run-lora-only)
            RUN_LORA_ONLY=1
            shift
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数：$1" >&2
            usage
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMMON_TEST_SCRIPT="$SCRIPT_DIR/test_quant_gguf_support.sh"

if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="$REPO_ROOT/test_outputs/qwen3_8b_batch_test"
fi

if [[ -z "$DATASET_PATH" ]]; then
    DATASET_PATH="$SCRIPT_DIR/gsm8k_test.jsonl"
fi

if [[ -z "$WIKI_PATH" ]]; then
    WIKI_PATH="$SCRIPT_DIR/wiki.test.raw"
fi

if [[ ! -x "$COMMON_TEST_SCRIPT" ]]; then
    echo "错误：通用测试脚本不存在或不可执行：$COMMON_TEST_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "错误：模型目录不存在：$MODEL_DIR" >&2
    exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "错误：数据文件不存在：$DATASET_PATH" >&2
    exit 1
fi

if [[ ! -f "$WIKI_PATH" ]]; then
    echo "错误：wiki 测试文本不存在：$WIKI_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
BATCH_SUMMARY="$OUTPUT_ROOT/batch_summary.txt"

# 这里只测你要求的三个量化格式，不测 Q4_K_M。
QUANTS=(Q5_0 Q5_K_M Q6_K Q8_0)

{
    echo "==== Qwen3-8B 批量量化测试 ===="
    echo "模型目录：$MODEL_DIR"
    echo "输出根目录：$OUTPUT_ROOT"
    echo "数据文件：$DATASET_PATH"
    echo "wiki 文本：$WIKI_PATH"
    echo "测试量化：${QUANTS[*]}"
    echo
} | tee "$BATCH_SUMMARY"

for quant in "${QUANTS[@]}"; do
    MODEL_PATH="$MODEL_DIR/Qwen3-8B-${quant}.gguf"
    TEST_OUT="$OUTPUT_ROOT/${quant}"

    echo "========================================" | tee -a "$BATCH_SUMMARY"
    echo "开始测试：$quant" | tee -a "$BATCH_SUMMARY"
    echo "模型路径：$MODEL_PATH" | tee -a "$BATCH_SUMMARY"

    # 这一步是在检查当前量化模型文件是否真的存在。
    # 如果文件不存在，就直接跳过该量化格式，避免整个批量流程中断。
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "警告：缺少模型文件，跳过 $quant -> $MODEL_PATH" | tee -a "$BATCH_SUMMARY"
        echo | tee -a "$BATCH_SUMMARY"
        continue
    fi

    CMD=(bash "$COMMON_TEST_SCRIPT"
        -m "$MODEL_PATH"
        -f "$DATASET_PATH"
        -w "$WIKI_PATH"
        -o "$TEST_OUT")

    # 这一步是在决定是否把 LoRA-only 回归验证一起带上。
    if [[ "$RUN_LORA_ONLY" -eq 1 ]]; then
        CMD+=(--run-lora-only)
    fi

    # 这一步是在决定是否跳过编译。
    # 如果你已经手动编过一次，批量重复测时可以加 --skip-build 提速。
    if [[ "$SKIP_BUILD" -eq 1 ]]; then
        CMD+=(--skip-build)
    fi

    # 这一步是在决定是否跳过训练 + merge 验证。
    # 如果你只想做 verify-param-update，可以加 --skip-train。
    if [[ "$SKIP_TRAIN" -eq 1 ]]; then
        CMD+=(--skip-train)
    fi

    echo "执行命令：${CMD[*]}" | tee -a "$BATCH_SUMMARY"
    if "${CMD[@]}"; then
        echo "结果：$quant 测试成功" | tee -a "$BATCH_SUMMARY"
    else
        echo "结果：$quant 测试失败" | tee -a "$BATCH_SUMMARY"
    fi
    echo | tee -a "$BATCH_SUMMARY"
done

# 这里给出一个总提醒，告诉你批量测试完成后去哪里看结果。
{
    echo "==== 批量测试完成 ===="
    echo "汇总文件：$BATCH_SUMMARY"
    echo "各量化结果目录："
    for quant in "${QUANTS[@]}"; do
        echo "  - $OUTPUT_ROOT/$quant"
    done
    echo
    echo "建议优先查看："
    echo "  1. 每个量化目录下的 test_summary.txt"
    echo "  2. 每个量化目录下 logs/finetune_merge.log"
    echo "  3. 每个量化目录下 logs/ppl_finetuned.log"
} | tee -a "$BATCH_SUMMARY"
