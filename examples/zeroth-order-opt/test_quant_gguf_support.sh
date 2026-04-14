#!/usr/bin/env bash
#
# 量化 GGUF 支持测试脚本
#
# 这个脚本用于验证 RAdaZO 量化微调路径在以下量化格式上的行为：
#   - Q4_K_M
#   - Q5_0
#   - Q5_K_M
#   - Q6_K
#
# 它不会修改仓库代码，只会：
#   1. 编译需要的二进制
#   2. 对输入模型做“最小参数修改验证”
#   3. 对输入模型做“一轮最小微调 + merge 保存验证”
#   4. 可选执行 LoRA-only 路径验证
#
#
# 用法示例：
#   bash examples/zeroth-order-opt/test_quant_gguf_support.sh \
#     -m /path/to/model.gguf \
#     -f examples/zeroth-order-opt/gsm8k_test.jsonl \
#     -w examples/zeroth-order-opt/wiki.test.raw \
#     -o /root/ZO/llama.cpp/test_outputs/quant_gguf_test
#
# 可选：
#   --skip-build      跳过编译
#   --skip-train      跳过最小微调验证
#   --run-lora-only   额外执行 LoRA-only 验证
#
set -euo pipefail

MODEL_PATH=""
DATASET_PATH=""
WIKI_PATH=""
OUTPUT_DIR=""
SKIP_BUILD=0
SKIP_TRAIN=0
RUN_LORA_ONLY=0
THREADS="${THREADS:-8}"

usage() {
    cat <<'USAGE'
用法：
  bash examples/zeroth-order-opt/test_quant_gguf_support.sh \
    -m <模型路径> \
    -f <训练数据路径> \
    -w <perplexity测试文本路径> \
    [-o <输出目录>]

参数说明：
  -m    输入 GGUF 模型路径（Q4_K_M / Q5_0 / Q5_K_M / Q6_K 均可）
  -f    微调使用的数据文件路径，例如 gsm8k_test.jsonl
  -w    perplexity 对比使用的文本文件，例如 wiki.test.raw
  -o    所有测试输出目录；不传时默认写到仓库下 test_outputs/quant_gguf_test

可选参数：
  --skip-build      跳过编译步骤
  --skip-train      跳过最小微调 + merge 保存验证
  --run-lora-only   额外执行 LoRA-only 路径验证
  -h, --help        显示帮助
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m)
            MODEL_PATH="$2"
            shift 2
            ;;
        -f)
            DATASET_PATH="$2"
            shift 2
            ;;
        -w)
            WIKI_PATH="$2"
            shift 2
            ;;
        -o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=1
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        --run-lora-only)
            RUN_LORA_ONLY=1
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
BUILD_DIR="$REPO_ROOT/build"
VERIFY_BIN="$BUILD_DIR/bin/verify-param-update"
FINETUNE_BIN="$BUILD_DIR/bin/finetune-radazo-quant"
PPL_BIN="$BUILD_DIR/bin/llama-perplexity"
CLI_BIN="$BUILD_DIR/bin/llama-cli"
DOUBLE_QUANT_SCRIPT="$SCRIPT_DIR/verify-double-quant-loss.sh"

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$REPO_ROOT/test_outputs/quant_gguf_test"
fi

if [[ -z "$MODEL_PATH" || -z "$DATASET_PATH" || -z "$WIKI_PATH" ]]; then
    echo "错误：缺少必要参数。" >&2
    usage
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "错误：模型文件不存在：$MODEL_PATH" >&2
    exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "错误：训练数据文件不存在：$DATASET_PATH" >&2
    exit 1
fi

if [[ ! -f "$WIKI_PATH" ]]; then
    echo "错误：perplexity 文本文件不存在：$WIKI_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
ARTIFACT_DIR="$OUTPUT_DIR/artifacts"
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

BASE_NAME="$(basename "$MODEL_PATH")"
VERIFY_MODEL="$ARTIFACT_DIR/${BASE_NAME%.gguf}.verify.gguf"
FINETUNED_MODEL="$ARTIFACT_DIR/${BASE_NAME%.gguf}.finetuned.gguf"
LORA_MODEL="$ARTIFACT_DIR/${BASE_NAME%.gguf}.lora.gguf"
SUMMARY_FILE="$OUTPUT_DIR/test_summary.txt"

# 这里通过模型文件名推断量化格式，只用于日志展示。
# 真正的支持判断仍然在程序内部依赖 ggml 的 quant type。
MODEL_FORMAT="UNKNOWN"
case "$BASE_NAME" in
    *Q4_K_M*.gguf) MODEL_FORMAT="Q4_K_M" ;;
    *Q5_0*.gguf)   MODEL_FORMAT="Q5_0" ;;
    *Q5_K_M*.gguf) MODEL_FORMAT="Q5_K_M" ;;
    *Q6_K*.gguf)   MODEL_FORMAT="Q6_K" ;;
esac

run_and_log() {
    local log_file="$1"
    shift
    echo "[CMD] $*" | tee -a "$log_file"
    "$@" 2>&1 | tee -a "$log_file"
}

# 统一记录说明，方便你回头看整个测试是否成功。
{
    echo "==== 量化 GGUF 支持测试 ===="
    echo "模型：$MODEL_PATH"
    echo "推断格式：$MODEL_FORMAT"
    echo "训练数据：$DATASET_PATH"
    echo "PPL 文本：$WIKI_PATH"
    echo "输出目录：$OUTPUT_DIR"
    echo
} | tee "$SUMMARY_FILE"

# 第 0 步：进入 conda 环境。
# 测试目的：确保后续构建和运行都使用已经配置好的 llama-zo 环境。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llama-zo

# 第 1 步：编译所需目标。
# 测试目的：确认当前代码至少能成功构建关键二进制，避免后面运行时才发现编译错误。
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "[1/6] 编译测试目标：verify-param-update / finetune-radazo-quant / llama-perplexity" | tee -a "$SUMMARY_FILE"
    run_and_log "$LOG_DIR/build.log" cmake --build "$BUILD_DIR" --target verify-param-update finetune-radazo-quant llama-perplexity -- -j"$THREADS"
else
    echo "[1/6] 已跳过编译步骤" | tee -a "$SUMMARY_FILE"
fi

for required_bin in "$VERIFY_BIN" "$FINETUNE_BIN" "$PPL_BIN"; do
    if [[ ! -x "$required_bin" ]]; then
        echo "错误：缺少可执行文件：$required_bin" | tee -a "$SUMMARY_FILE"
        exit 1
    fi
done

# 第 2 步：执行最小参数修改验证。
# 测试目的：验证当前 quant type 是否能完成
#   dequant -> 参数微小修改 -> requant -> 保存
# 如果这一步失败，说明最基本的量化保存链就不通。
echo "[2/6] 执行 verify-param-update：验证最小参数修改与保存链" | tee -a "$SUMMARY_FILE"
run_and_log "$LOG_DIR/verify_param_update.log" "$VERIFY_BIN" -m "$MODEL_PATH" -o "$VERIFY_MODEL"

# 第 3 步：对比 verify 后模型与原模型的 perplexity。
# 测试目的：确认保存出来的新模型确实和原模型有可观察差异。
echo "[3/6] 对比 verify 后模型与原模型的 perplexity" | tee -a "$SUMMARY_FILE"
run_and_log "$LOG_DIR/ppl_baseline_verify.log" "$PPL_BIN" -m "$MODEL_PATH" -f "$WIKI_PATH"
run_and_log "$LOG_DIR/ppl_modified_verify.log" "$PPL_BIN" -m "$VERIFY_MODEL" -f "$WIKI_PATH"

# 第 4 步：执行最小微调 + merge 保存。
# 测试目的：验证真实训练路径下，LoRA 训练后的 merge/save 是否能在当前 quant type 上跑通。
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo "[4/6] 执行最小微调：验证训练 + merge 保存链" | tee -a "$SUMMARY_FILE"
    run_and_log "$LOG_DIR/finetune_merge.log" \
        "$FINETUNE_BIN" \
        -m "$MODEL_PATH" \
        -f "$DATASET_PATH" \
        -o "$FINETUNED_MODEL" \
        --epochs 1
else
    echo "[4/6] 已跳过最小微调 + merge 验证" | tee -a "$SUMMARY_FILE"
fi

# 第 5 步：对比 finetune 后模型与原模型的 perplexity。
# 测试目的：确认真实训练保存出来的新模型与原模型有可观察差异。
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo "[5/6] 对比 finetune 后模型与原模型的 perplexity" | tee -a "$SUMMARY_FILE"
    run_and_log "$LOG_DIR/ppl_baseline_finetune.log" "$PPL_BIN" -m "$MODEL_PATH" -f "$WIKI_PATH"
    run_and_log "$LOG_DIR/ppl_finetuned.log" "$PPL_BIN" -m "$FINETUNED_MODEL" -f "$WIKI_PATH"

    # 第 5.5 步：检查量化保存后的差异。
    # 测试目的：从二进制层面快速确认保存后的模型和原模型并不完全一样。
    if [[ -x "$DOUBLE_QUANT_SCRIPT" ]]; then
        echo "[5.5/6] 执行二进制差异检查：验证保存后模型确实发生变化" | tee -a "$SUMMARY_FILE"
        run_and_log "$LOG_DIR/double_quant_check.log" bash "$DOUBLE_QUANT_SCRIPT" "$MODEL_PATH" "$FINETUNED_MODEL"
    fi
fi

# 第 6 步：可选执行 LoRA-only 路径验证。
# 测试目的：确认不 merge 回量化 GGUF、只导出 LoRA 的路径没有被破坏。
if [[ "$RUN_LORA_ONLY" -eq 1 ]]; then
    echo "[6/6] 执行 LoRA-only 路径验证" | tee -a "$SUMMARY_FILE"
    run_and_log "$LOG_DIR/lora_only.log" \
        "$FINETUNE_BIN" \
        -m "$MODEL_PATH" \
        -f "$DATASET_PATH" \
        -o "$LORA_MODEL" \
        --epochs 1 \
        --save-lora-only

    if [[ ! -f "$LORA_MODEL" ]]; then
        echo "错误：LoRA-only 输出不存在：$LORA_MODEL" | tee -a "$SUMMARY_FILE"
        exit 1
    fi

    if [[ -x "$CLI_BIN" ]]; then
        echo "[INFO] 已生成 LoRA-only 文件，可手动进一步验证：$LORA_MODEL" | tee -a "$SUMMARY_FILE"
    fi
else
    echo "[6/6] 已跳过 LoRA-only 路径验证" | tee -a "$SUMMARY_FILE"
fi

{
    echo
    echo "==== 测试完成 ===="
    echo "输出目录：$OUTPUT_DIR"
    echo "日志目录：$LOG_DIR"
    echo "产物目录：$ARTIFACT_DIR"
    echo "verify 输出：$VERIFY_MODEL"
    if [[ "$SKIP_TRAIN" -eq 0 ]]; then
        echo "finetune 输出：$FINETUNED_MODEL"
    fi
    if [[ "$RUN_LORA_ONLY" -eq 1 ]]; then
        echo "LoRA-only 输出：$LORA_MODEL"
    fi
} | tee -a "$SUMMARY_FILE"
