# Quantized GGUF Support Update

## Overview

This update extends the verified quantized GGUF support in the RAdaZO quantized fine-tuning flow from the original `Q4_K_M` path to:

- `Q4_K_M`
- `Q5_0`
- `Q5_K_M`
- `Q6_K`
- `Q8_0`

The core algorithm was intentionally **not changed**.

The training path still follows the existing design:

1. Train only FP32 LoRA tensors
2. Keep base GGUF tensors read-only during training
3. During merge/save or verification:
   - dequantize the base tensor row to FP32
   - add the LoRA delta / perturbation
   - requantize back to the original tensor type

The only differences between quantization formats are the underlying `ggml` dequant/requant traits and rounding behavior.

## What Changed

### 1. Added a reusable quant support helper

File:

- [quant-gguf-support.h](/root/ZO/llama.cpp/examples/zeroth-order-opt/quant-gguf-support.h)

This new helper centralizes:

- the verified support set
- human-readable quant format names
- checks for whether a type has usable `ggml` dequant/requant traits

Verified support set:

- `GGML_TYPE_Q4_K` -> `Q4_K_M`
- `GGML_TYPE_Q5_0` -> `Q5_0`
- `GGML_TYPE_Q5_K` -> `Q5_K_M`
- `GGML_TYPE_Q6_K` -> `Q6_K`
- `GGML_TYPE_Q8_0` -> `Q8_0`

This keeps the implementation extensible:

- the merge/save logic is still generic
- adding a new verified type later only requires updating this table and adding tests/docs
- no algorithm duplication was introduced

### 2. Updated merged GGUF save path

File:

- [lora-adapter.cpp](/root/ZO/llama.cpp/examples/zeroth-order-opt/lora-adapter.cpp)

Changes:

- reused the shared quant helper
- added explicit logging for verified quant formats
- preserved the existing generic traits-driven flow
- improved diagnostics so logs are no longer implicitly `Q4_K_M`-specific

Behavior now:

- if the quant type is one of the verified formats, it is reported as such
- if the quant type is outside the verified set but still has valid `ggml` traits, the code still attempts the same generic merge path
- if the quant type lacks `to_float` / `from_float_ref`, the merge now fails clearly instead of silently pretending support

Important: the merge algorithm itself did not change.

### 3. Updated parameter verification tool

File:

- [verify-param-update.cpp](/root/ZO/llama.cpp/examples/zeroth-order-opt/verify-param-update.cpp)

Changes:

- reused the same quant helper
- reports the actual quant format being tested
- distinguishes:
  - verified quant support
  - generic fallback via traits
  - unsupported quant types with no requant support

The verification logic is still the same:

- read quantized tensor bytes
- dequantize to FP32
- apply a small perturbation
- requantize
- save the modified model

### 4. Updated quantized fine-tuning entrypoint messaging

File:

- [finetune-radazo-quant.cpp](/root/ZO/llama.cpp/examples/zeroth-order-opt/finetune-radazo-quant.cpp)

Changes:

- updated top-of-file comments
- added runtime log lines declaring the verified quantized GGUF support set
- clarified that other quant formats still rely on the same generic `ggml` traits-based merge/save flow

No CLI behavior was changed.

### 5. Updated docs and verification wording

Files:

- [README.md](/root/ZO/llama.cpp/examples/zeroth-order-opt/README.md)
- [FINETUNE_DEMO.md](/root/ZO/llama.cpp/examples/zeroth-order-opt/FINETUNE_DEMO.md)
- [GUIDANCE.txt](/root/ZO/llama.cpp/examples/zeroth-order-opt/GUIDANCE.txt)
- [verify-double-quant-loss.sh](/root/ZO/llama.cpp/examples/zeroth-order-opt/verify-double-quant-loss.sh)

Changes:

- documentation now explicitly states the verified support set:
  - `Q4_K_M`
  - `Q5_0`
  - `Q5_K_M`
  - `Q6_K`
- `Q8_0`
- wording was generalized so diagnostics no longer imply only `Q4_K_M` is supported
- example command flow remains unchanged

## What Did Not Change

The following were intentionally left unchanged:

- RAdaZO optimizer logic
- LoRA training flow
- training-time parameter update strategy
- merge/save algorithm structure
- `--save-lora-only` behavior
- command-line interface

This update is strictly a quantization-format support extension layer.

## Build Verification Already Completed

The following targets were rebuilt successfully:

- `finetune-radazo-quant`
- `verify-param-update`

Build command used:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llama-zo
cmake --build /root/ZO/llama.cpp/build --target finetune-radazo-quant verify-param-update -- -j8
```

## How To Test

### 1. Rebuild locally

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate llama-zo
cd /root/ZO/llama.cpp
cmake --build build --target finetune-radazo-quant verify-param-update -- -j8
```

### 2. Verify parameter update on each quant format

For each model format:

- `Q4_K_M`
- `Q5_0`
- `Q5_K_M`
- `Q6_K`
- `Q8_0`

run:

```bash
./build/bin/verify-param-update -m /path/to/model.gguf -o /tmp/model_modified.gguf
```

Then compare perplexity:

```bash
./build/bin/llama-perplexity -m /path/to/model.gguf -f examples/zeroth-order-opt/wiki.test.raw
./build/bin/llama-perplexity -m /tmp/model_modified.gguf -f examples/zeroth-order-opt/wiki.test.raw
```

Expected result:

- the modified model should differ from the baseline
- logs should show the detected quant format
- verified formats should be identified explicitly

### 3. Test merged fine-tuning output

Run a minimal fine-tuning pass for each format:

```bash
./build/bin/finetune-radazo-quant \
  -m /path/to/model.gguf \
  -f examples/zeroth-order-opt/gsm8k_test.jsonl \
  -o /tmp/model_finetuned.gguf \
  --epochs 1
```

Expected result:

- training completes normally
- save/merge completes
- logs show quant-format-aware merge diagnostics
- `Probe A/B/C` logs appear during merge

### 4. Compare baseline vs merged model

```bash
./build/bin/llama-perplexity -m /path/to/model.gguf -f examples/zeroth-order-opt/wiki.test.raw
./build/bin/llama-perplexity -m /tmp/model_finetuned.gguf -f examples/zeroth-order-opt/wiki.test.raw
```

Expected result:

- perplexity or output should differ
- if not, inspect the Probe C requant logs to see whether updates were rounded away

### 5. Run binary-level merge verification

```bash
examples/zeroth-order-opt/verify-double-quant-loss.sh /path/to/model.gguf /tmp/model_finetuned.gguf
```

Expected result:

- if hashes are identical, requantization wiped all updates
- if files differ, some updates persisted

### 6. Confirm LoRA-only path still works

```bash
./build/bin/finetune-radazo-quant \
  -m /path/to/model.gguf \
  -f examples/zeroth-order-opt/gsm8k_test.jsonl \
  -o /tmp/lora_fp32.gguf \
  --epochs 1 \
  --save-lora-only
```

Then:

```bash
./build/bin/llama-cli -m /path/to/model.gguf --lora /tmp/lora_fp32.gguf -p "hello"
```

Expected result:

- LoRA-only export still works exactly as before
- this path should be unaffected by quant-format changes

## Recommended Test Order

1. `verify-param-update` on one `Q4_K_M` model for regression safety
2. `verify-param-update` on `Q5_0`
3. `verify-param-update` on `Q5_K_M`
4. `verify-param-update` on `Q6_K`
5. one minimal `finetune-radazo-quant` merge/save run on each format
6. one `--save-lora-only` regression test

## Notes

- The verified support set is explicit, but the implementation still keeps the generic fallback structure.
- This means the code remains extensible without rewriting the merge/save algorithm.
- If you later want to certify another quant format, the intended workflow is:
  - add it to the verified support table
  - validate its `ggml` traits
  - run the same verification sequence
