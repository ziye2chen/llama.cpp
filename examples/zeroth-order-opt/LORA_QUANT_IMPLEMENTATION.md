# LoRA Training for Quantized GGUF Models - Implementation Guide

## Overview

This implementation provides a foundation for training LoRA (Low-Rank Adaptation) adapters on quantized GGUF models (Q4_K_M) using R-AdaZO optimization. The base quantized model remains frozen, and only the LoRA adapters are trained.

## Current Implementation Status

### ✅ Completed

1. **LoRA Adapter Structure** (`lora-adapter.h/cpp`)
   - LoRA layer definition (A and B matrices)
   - Initialization with proper weight initialization
   - Parameter collection for training
   - Basic infrastructure for LoRA management

2. **Training Framework** (`finetune-radazo-quant.cpp`)
   - Integration with R-AdaZO optimizer
   - LoRA parameter collection
   - Training loop structure
   - Dataset loading (GSM8K)

3. **Build System**
   - CMakeLists.txt updated with new target
   - Dependencies configured

### ⚠️ Limitations & TODO

#### 1. Forward Pass Integration (Critical)

**Current State**: The LoRA adapters are created and initialized, but the forward pass does not yet integrate LoRA outputs with the base model.

**Required Changes**:
- Modify `llama.cpp`'s forward pass to support LoRA
- Or implement a custom forward pass that:
  1. Dequantizes base tensors on-the-fly
  2. Computes base output: `Y_base = W_base_dequant @ X`
  3. Computes LoRA output: `Y_lora = (B @ (A @ X)) * (alpha / rank)`
  4. Combines: `Y = Y_base + Y_lora`

**Implementation Approach**:
```cpp
// In llama.cpp's forward pass (simplified concept)
for each layer:
    // Base path (existing)
    Y_base = dequantize_and_multiply(W_quantized, X);
    
    // LoRA path (new)
    if (lora_adapter.has_layer(layer_name)):
        Y_lora = lora_B @ (lora_A @ X) * (alpha / rank);
        Y = Y_base + Y_lora;
    else:
        Y = Y_base;
```

#### 2. Dequantization Functions

**Current State**: Placeholder dequantization function.

**Required**: Implement proper dequantization for Q4_K_M:
```cpp
// Need to use ggml functions like:
ggml_dequantize_row_q4_k(...)
// Or access the dequantization functions from ggml
```

#### 3. LoRA Save/Load

**Current State**: Placeholder functions.

**Required**: Implement serialization:
- Save LoRA A and B matrices to binary file
- Include metadata (rank, alpha, layer names)
- Load function to restore LoRA adapters

#### 4. Merge Tool (Post-Training)

**Required**: Create a separate tool to merge LoRA with base model:
1. Load base GGUF model
2. Load trained LoRA adapters
3. Dequantize target layers
4. Compute: `W_new = W_base + B @ A`
5. Re-quantize to Q4_K_M
6. Save new GGUF model

## Architecture

### Memory Layout

```
┌─────────────────────────────────────┐
│  Base Model (GGUF Q4_K_M)          │
│  - Quantized weights (frozen)       │
│  - Scales and quants                │
│  - Read-only during training        │
└─────────────────────────────────────┘
              │
              │ (dequantize on-the-fly)
              ▼
┌─────────────────────────────────────┐
│  LoRA Adapters (FP32)               │
│  - lora_A: [rank, in_dim]           │
│  - lora_B: [out_dim, rank]          │
│  - Trainable parameters             │
└─────────────────────────────────────┘
              │
              │ (forward pass)
              ▼
┌─────────────────────────────────────┐
│  Output: Y = Base(X) + LoRA(X)     │
└─────────────────────────────────────┘
```

### Training Flow

1. **Initialization**
   - Load quantized GGUF model (frozen)
   - Create LoRA adapters for target layers
   - Initialize A with Gaussian, B with zeros

2. **Forward Pass** (per batch)
   - Dequantize base weights (temporary)
   - Compute base output
   - Compute LoRA output
   - Combine outputs

3. **Optimization** (R-AdaZO)
   - Perturb LoRA parameters only
   - Estimate gradients via finite differences
   - Update LoRA A and B using Adam-style optimizer

4. **Save**
   - Save only LoRA adapters (not base model)
   - Base model remains unchanged

## Usage

### Building

```bash
cd llama.cpp/build
cmake --build . --target finetune-radazo-quant
```

### Running

```bash
./bin/finetune-radazo-quant \
  -m /path/to/model-Q4_K_M.gguf \
  -f /path/to/gsm8k_test.jsonl \
  -o lora_adapter.bin \
  --epochs 1
```

### Parameters

- `-m`: Path to quantized GGUF model (Q4_K_M)
- `-f`: Training data file (GSM8K JSONL format)
- `-o`: Output path for LoRA adapters
- `--epochs`: Number of training epochs

## Next Steps

1. **Implement Forward Pass Integration**
   - Modify llama.cpp or create custom forward pass
   - Integrate LoRA computation into the graph

2. **Complete Dequantization**
   - Use proper ggml dequantization functions
   - Handle Q4_K_M format correctly

3. **Implement Save/Load**
   - Binary format for LoRA adapters
   - Include metadata

4. **Create Merge Tool**
   - Separate utility to merge LoRA with base
   - Re-quantization support

5. **Testing**
   - Test with small models first
   - Verify LoRA training converges
   - Compare with full fine-tuning

## Notes

- **Memory Efficiency**: LoRA uses much less memory than full fine-tuning
- **Base Model Safety**: Original GGUF file is never modified
- **Flexibility**: Can train multiple LoRA adapters for different tasks
- **Compatibility**: Works with any quantized GGUF format (Q4_K_M, Q8_0, etc.)

## References

- LoRA Paper: "LoRA: Low-Rank Adaptation of Large Language Models"
- R-AdaZO Paper: "Refining Adaptive Zeroth-Order Optimization at Ease"
- GGUF Format: llama.cpp documentation
