#!/bin/bash
# Phase 2: File-level verification for Double Quantization Loss hypothesis.
# Compares baseline vs fine-tuned model: if SHA256 identical or diff < 0.01%,
# requantization likely wiped all LoRA updates.
#
# Usage:
#   ./verify-double-quant-loss.sh <baseline.gguf> <model_radazo_finetuned.gguf>

set -e

BASELINE="${1:-}"
FINETUNED="${2:-}"

if [[ -z "$BASELINE" || -z "$FINETUNED" ]]; then
    echo "Usage: $0 <baseline.gguf> <model_radazo_finetuned.gguf>"
    echo ""
    echo "Phase 2 verification:"
    echo "  - SHA256 hash comparison: identical = all updates wiped"
    echo "  - Binary diff: diff_bytes/total < 0.01% = almost all wiped"
    exit 1
fi

if [[ ! -f "$BASELINE" ]]; then
    echo "Error: baseline file not found: $BASELINE"
    exit 1
fi

if [[ ! -f "$FINETUNED" ]]; then
    echo "Error: fine-tuned file not found: $FINETUNED"
    exit 1
fi

echo "=== Phase 2: File-level verification ==="
echo "Baseline:   $BASELINE"
echo "Fine-tuned: $FINETUNED"
echo ""

# Hash comparison
echo "--- SHA256 hashes ---"
HASH_BASE=$(sha256sum "$BASELINE" | cut -d' ' -f1)
HASH_FINE=$(sha256sum "$FINETUNED" | cut -d' ' -f1)
echo "Baseline:   $HASH_BASE"
echo "Fine-tuned: $HASH_FINE"

if [[ "$HASH_BASE" == "$HASH_FINE" ]]; then
    echo ""
    echo "*** RESULT: Hashes IDENTICAL -> Double Quantization Loss CONFIRMED ***"
    echo "All weight updates were wiped by requantization for this quantized GGUF format."
else
    echo ""
    echo "Hashes differ. Computing binary diff..."
fi

# Binary diff (byte-level)
echo ""
echo "--- Binary diff ---"
SIZE_BASE=$(wc -c < "$BASELINE")
SIZE_FINE=$(wc -c < "$FINETUNED")
echo "Baseline size:   $SIZE_BASE bytes"
echo "Fine-tuned size: $SIZE_FINE bytes"

DIFF_BYTES=$(cmp -l "$BASELINE" "$FINETUNED" 2>/dev/null | wc -l || echo 0)
# cmp -l outputs one line per differing byte; if files identical, cmp exits 0 and outputs nothing
if cmp -s "$BASELINE" "$FINETUNED" 2>/dev/null; then
    DIFF_BYTES=0
fi

TOTAL=$SIZE_BASE
if [[ $TOTAL -gt 0 ]]; then
    RATIO="0"
    if command -v bc &>/dev/null; then
        RATIO=$(echo "scale=6; 100 * $DIFF_BYTES / $TOTAL" | bc)
    fi
    echo "Differing bytes: $DIFF_BYTES / $TOTAL"
    echo ""
    if [[ $DIFF_BYTES -eq 0 ]]; then
        echo "*** RESULT: Diff < 0.01% -> Double Quantization Loss LIKELY ***"
        echo "Almost all parameters were rounded back to original values."
    elif command -v bc &>/dev/null && [[ $(echo "$RATIO < 0.01" | bc) -eq 1 ]]; then
        echo "*** RESULT: Diff $RATIO% < 0.01% -> Double Quantization Loss LIKELY ***"
    else
        echo "Diff > 0.01%: some updates persisted (or metadata changed)."
    fi
fi

echo ""
echo "=== Phase 3: FP32 LoRA verification ==="
echo "To verify ZO training learned something, run:"
echo "  1. ./finetune-radazo-quant ... --save-lora-only -o lora_fp32.gguf"
echo "  2. ./llama-cli -m $BASELINE --lora lora_fp32.gguf -p \"<your prompt>\""
echo "  3. Compare output to baseline (no --lora)."
echo "  If FP32 LoRA changes output -> ZO effective, merge requant wiped it."
