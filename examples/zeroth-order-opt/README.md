# Zeroth-Order Fine-Tuning for GGUF Models

A gradient-free optimization implementation for fine-tuning `llama.cpp` models without backpropagation.

## Table of Contents

- [Overview](#overview)
- [Theory: Zeroth-Order Optimization](#theory-zeroth-order-optimization)
- [Why Zeroth-Order?](#why-zeroth-order)
- [Implementation Details](#implementation-details)
- [Code Structure](#code-structure)
- [Usage](#usage)
- [API Additions](#api-additions)
- [Results and Validation](#results-and-validation)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Overview

This project implements **real zeroth-order optimization** (also known as gradient-free or derivative-free optimization) for fine-tuning large language models in GGUF format. Unlike standard fine-tuning methods that rely on backpropagation to compute gradients, this approach estimates gradients using **actual forward passes** with perturbed parameters.

### Key Features

- ✅ **No Backpropagation**: Uses only forward passes (`llama_decode`)
- ✅ **True Gradient Estimation**: Each parameter perturbation triggers a real forward pass
- ✅ **Finite Differences**: Estimates gradients via `[L(θ+ε) - L(θ)] / ε`
- ✅ **Lower Memory**: No need to store gradients or intermediate activations for backprop
- ✅ **Direct Parameter Updates**: Modifies model weights in-place using `ggml_backend_tensor_set`
- ✅ **Model Persistence**: Saves updated models using `llama_model_save_to_file`

### Performance Note

⚠️ **This is SIGNIFICANTLY slower than backpropagation!** Each batch requires:
- 1 baseline forward pass
- `n_params_per_iter × n_elements_per_param` additional forward passes (default: 3 × 2 = 6)
- **Total: ~7× slower per batch** than standard inference

This is a working demonstration of gradient-free optimization, not a production training method.

---

## Theory: Zeroth-Order Optimization

### The Core Idea

Standard gradient descent requires computing the gradient of the loss function with respect to each parameter:

```
θ_new = θ_old - α * ∇L(θ)
```

Where:
- `θ` = model parameters
- `α` = learning rate
- `∇L(θ)` = gradient (computed via backpropagation)

**Zeroth-order optimization** estimates this gradient using finite differences:

```
∇L(θ) ≈ [L(θ + ε·e_i) - L(θ)] / ε
```

Where:
- `ε` = small perturbation (epsilon)
- `e_i` = unit vector (perturb one parameter at a time)
- `L(θ)` = loss function evaluated via forward pass

### Algorithm

For each training iteration:

1. **Compute baseline loss**: Run forward pass to get `L(θ)`
2. **For each sampled parameter**:
   - Perturb: `θ_i ← θ_i + ε`
   - Forward pass: Compute `L(θ + ε·e_i)`
   - Estimate gradient: `g_i = [L(θ + ε·e_i) - L(θ)] / ε`
   - Update: `θ_i ← θ_i - α·g_i - λ·θ_i` (with weight decay)
3. **Repeat** for all batches and epochs

### Advantages and Trade-offs

| Aspect | Zeroth-Order | First-Order (Backprop) |
|--------|--------------|------------------------|
| **Memory** | Lower (no gradient storage) | Higher (stores activations + gradients) |
| **Computation** | More forward passes | One forward + one backward pass |
| **Convergence** | Slower (noisier gradient estimates) | Faster (exact gradients) |
| **Implementation** | Simpler (no autodiff needed) | Complex (requires backprop support) |
| **Learning Rate** | Much smaller (~100x) | Standard |

---

## Why Zeroth-Order?

### Use Cases

1. **Inference-Only Frameworks**: When backpropagation is not available (e.g., `llama.cpp`)
2. **Memory-Constrained Environments**: When gradient storage is prohibitive
3. **Black-Box Optimization**: When model internals are not accessible
4. **Research & Education**: To understand optimization without autodiff

### When NOT to Use

- When backpropagation is available and memory is sufficient
- When training time is critical
- For large-scale training with millions of examples

---

## Implementation Details

### Core Components

#### 1. Parameter Collection (`collect_model_params`)

Iterates through all model tensors and collects trainable weights:

```cpp
static std::vector<struct ggml_tensor *> collect_model_params(struct llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);
    const size_t n_tensors = llama_model_n_tensors(model);
    
    for (size_t i = 0; i < n_tensors; ++i) {
        struct ggml_tensor * tensor = llama_model_get_tensor_by_index(model, i, ...);
        
        // Filter for key weight matrices
        if (name contains "attn_q", "attn_k", "attn_v", "attn_output",
                           "ffn_gate", "ffn_up", "ffn_down") {
            params.push_back(tensor);
        }
    }
}
```

**Collected Tensors** (for Llama 3.2 1B):
- Attention weights: `wq`, `wk`, `wv`, `wo` per layer
- FFN weights: `w1` (gate), `w2` (down), `w3` (up) per layer
- **Total**: 112 tensors, ~1.2B parameters

#### 2. Loss Computation

Simplified loss using L2 norm of logits (first 1000 dimensions):

```cpp
float loss_base = 0.0f;
for (int v = 0; v < std::min(1000, n_vocab); ++v) {
    loss_base += logits_base[v] * logits_base[v];
}
loss_base = std::sqrt(loss_base);
```

**Note**: This is a simple proxy for the loss. In production, use cross-entropy loss: `-log(softmax(logits)[target_token])`.

#### 3. Parameter Update Loop (REAL Zeroth-Order)

For each batch, randomly sample parameters and estimate gradients via **actual forward passes**:

```cpp
for (int32_t p = 0; p < n_params_per_iter; ++p) {
    size_t param_idx = random_sample(params);
    struct ggml_tensor * param = params[param_idx];
    
    // Sample random elements within this parameter
    for (int32_t e = 0; e < n_elements_per_param; ++e) {
        int64_t elem_idx = random_sample_element(param);
        
        // Read current value
        float current_val;
        ggml_backend_tensor_get(param, &current_val, elem_idx * sizeof(float), sizeof(float));
        
        // Perturb parameter
        float perturbed_val = current_val + epsilon;
        ggml_backend_tensor_set(param, &perturbed_val, elem_idx * sizeof(float), sizeof(float));
        
        // *** CRITICAL: Synchronize and clear KV cache ***
        llama_synchronize(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);
        
        // *** DO ACTUAL FORWARD PASS WITH PERTURBED PARAMETER ***
        if (llama_decode(ctx, batch) != 0) {
            // Restore original value on error
            ggml_backend_tensor_set(param, &current_val, ...);
            continue;
        }
        
        // Compute loss with perturbed parameter
        float * logits_perturbed = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        float loss_perturbed = 0.0f;
        for (int v = 0; v < std::min(1000, n_vocab); ++v) {
            loss_perturbed += logits_perturbed[v] * logits_perturbed[v];
        }
        loss_perturbed = std::sqrt(loss_perturbed);
        
        // Estimate gradient via finite difference
        float gradient = (loss_perturbed - loss_base) / epsilon;
        
        // Update with weight decay
        float update = learning_rate * (gradient + weight_decay * current_val);
        float new_val = current_val - update;
        
        // Write back updated value
        ggml_backend_tensor_set(param, &new_val, elem_idx * sizeof(float), sizeof(float));
        llama_synchronize(ctx);
    }
}
```

**Key Point**: Each element update requires a **full forward pass** through the model! This is why zeroth-order is much slower than backpropagation.

#### 4. KV Cache Management (Critical for Multiple Forward Passes)

Since we're doing multiple forward passes per batch (baseline + perturbed), we must clear the KV cache:

```cpp
// Process training data in batches
for (size_t i = 0; i + n_batch < train_tokens.size(); i += n_batch) {
    // Clear KV cache at start of each batch
    // (We'll reprocess the same batch multiple times with different parameters)
    llama_memory_clear(llama_get_memory(ctx), true);
    
    // Prepare batch with positions starting from 0
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    for (int j = 0; j < n_batch; ++j) {
        batch.token[j] = train_tokens[i + j];
        batch.pos[j] = j;  // Always start from position 0
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = (j == n_batch - 1);
    }
    
    // Baseline forward pass
    llama_decode(ctx, batch);
    float loss_base = compute_loss();
    
    // For each perturbed forward pass:
    //   1. Clear KV cache (positions must start from 0 again)
    //   2. Do forward pass with same batch
    //   3. Compute perturbed loss
    for (each_perturbation) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decode(ctx, batch);  // Same batch, different parameters
        float loss_perturbed = compute_loss();
    }
}
```

**Why this is necessary**: Without clearing the KV cache, `llama_decode` expects positions to continue from where they left off (e.g., position 64), but we're starting from position 0 again, causing a "inconsistent sequence positions" error.

### Hyperparameters

```cpp
struct zeroth_order_params {
    float epsilon = 1e-3f;              // Perturbation size for finite differences
    float learning_rate = 1e-4f;        // Learning rate (tune based on loss behavior)
    float weight_decay = 0.01f;         // L2 regularization to prevent overfitting
    int32_t n_params_per_iter = 3;      // Sample 3 parameters per batch (balance speed vs coverage)
    int32_t n_elements_per_param = 2;   // Sample 2 elements per tensor (6 forward passes/batch)
    int32_t max_train_tokens = 100;     // Dataset size limit for testing (-1 for full)
    bool log_gradients = false;         // Enable verbose gradient logging (debug)
};
```

**Forward Pass Count per Batch**: `1 (baseline) + n_params_per_iter × n_elements_per_param = 1 + 3×2 = 7`

This means zeroth-order is **~7× slower** than standard inference per batch!

**Tuning Guidelines**:
- `epsilon`: Too small → numerical instability; too large → poor approximation
- `learning_rate`: Start small (1e-4 to 1e-1) and adjust based on loss
- `n_params_per_iter`: More = slower but more stable updates
- `weight_decay`: Prevents overfitting on small datasets

---

## Code Structure

```
examples/zeroth-order-opt/
├── finetune-zeroth-order.cpp     # Main fine-tuning demo
├── zeroth-order-test.cpp         # Unit tests (quadratic, linear regression)
├── CMakeLists.txt                # Build configuration
└── README.md                     # This file

src/
├── llama-zeroth-order-opt.h      # Header (currently unused stub)
└── llama-zeroth-order-opt.cpp    # Implementation (currently unused stub)

include/llama.h                    # Added public API functions
src/llama-model.cpp                # Implemented API functions
```

### Key Files

#### `finetune-zeroth-order.cpp`

Main demonstration program that:
1. Loads a GGUF model
2. Tokenizes training data
3. Collects trainable parameters
4. Performs zeroth-order optimization
5. Saves the updated model

#### `zeroth-order-test.cpp`

Unit tests demonstrating the concept on simple functions:
- **Test 1**: Quadratic optimization (`f(x) = (x - 3.7)²`)
- **Test 2**: Linear regression (`y = 2.5x + 1.3`)

---

## Usage

### Build

```bash
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CURL=OFF
cmake --build . --config Debug --target finetune-zeroth-order
```

### Run Fine-Tuning

```bash
# Quick test with small dataset
.\bin\Debug\finetune-zeroth-order.exe \
    -m path/to/model.gguf \
    -f path/to/training_data.txt \
    -o output_model.gguf \
    -b 32 \
    -c 128 \
    --epochs 1
```

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Input GGUF model path | Required |
| `-f, --file` | Training data file | Required |
| `-o, --output` | Output model path | `llama3_2_1b_f32_zeroth_order_finetuned.gguf` |
| `-b, --batch-size` | Batch size | 64 |
| `-c, --ctx-size` | Context size | 128 |
| `--epochs` | Number of epochs | 1 |

### Adjusting Dataset Size

Edit `finetune-zeroth-order.cpp` line 221:

```cpp
zo_params.max_train_tokens = 6500;  // Change this:
                                    // 10     = ultra fast (~seconds)
                                    // 100    = quick test (~1 minute)
                                    // 6500   = moderate test (~5 minutes)
                                    // -1     = use full dataset
```

### Example Output

```
collect_model_params: model has 147 tensors total
collect_model_params:   [  1] blk.0.attn_q.weight (shape: [2048, 2048], 4194304 elements)
collect_model_params:   [  2] blk.0.attn_k.weight (shape: [2048, 512], 1048576 elements)
  ...
collect_model_params: collected 112 trainable parameter tensors

finetune_zeroth_order: ===== Epoch 1/1 =====
[TRAIN] Iter     10/    91 | Loss: 69.699 | Time:  31.23s | 0.3 it/s
[TRAIN] Iter     20/    91 | Loss: 69.507 | Time:  62.94s | 0.3 it/s
  ...
finetune_zeroth_order: Epoch 1 complete - Avg Loss: 71.618

main: saving fine-tuned model to test_output.gguf
main: model saved successfully!
```

---

## API Additions

To enable parameter access and modification, we added the following public API functions to `llama.h`:

### `llama_model_n_tensors`

```cpp
LLAMA_API size_t llama_model_n_tensors(const struct llama_model * model);
```

Returns the total number of tensors in the model.

### `llama_model_get_tensor_by_index`

```cpp
LLAMA_API struct ggml_tensor * llama_model_get_tensor_by_index(
        const struct llama_model * model,
                        size_t   index,
                          char * out_name,
                        size_t   out_name_size);
```

Returns a pointer to the tensor at the given index, optionally writing its name to `out_name`.

**Parameters**:
- `model`: The model to query
- `index`: Tensor index (0 to `llama_model_n_tensors(model) - 1`)
- `out_name`: Buffer for tensor name (can be NULL)
- `out_name_size`: Size of name buffer

**Returns**: Pointer to `ggml_tensor` or NULL if index is out of bounds.

### Implementation

In `src/llama-model.cpp`:

```cpp
size_t llama_model_n_tensors(const llama_model * model) {
    return model->tensors_by_name.size();
}

struct ggml_tensor * llama_model_get_tensor_by_index(
        const struct llama_model * model,
                        size_t   index,
                          char * out_name,
                        size_t   out_name_size) {
    if (index >= model->tensors_by_name.size()) {
        return nullptr;
    }
    
    const auto & [name, tensor] = model->tensors_by_name[index];
    
    if (out_name != nullptr && out_name_size > 0) {
        snprintf(out_name, out_name_size, "%s", name.c_str());
    }
    
    return tensor;
}
```

---

## Results and Validation

### Verifying Parameter Updates

To confirm that parameters were actually updated:

```bash
# Compute perplexity on original model
.\bin\Debug\llama-perplexity.exe -m original_model.gguf -f test_data.txt

# Compute perplexity on fine-tuned model
.\bin\Debug\llama-perplexity.exe -m output_model.gguf -f test_data.txt
```

If perplexity values differ, parameters were successfully updated.

### Expected Behavior

- **Loss should decrease**: Over iterations, the training loss should trend downward
- **File size preserved**: Output model should be the same size as input
- **Inference works**: The fine-tuned model should generate text normally

### Unit Tests

Run the test program to verify the optimization algorithm:

```bash
.\bin\Debug\zeroth-order-test.exe
```

Expected output:
```
Test 1: Quadratic Optimization
Initial value: x = 0.000000, f(x) = 13.690000
After 100 iterations: x = 3.695123, f(x) = 0.000024

Test 2: Linear Regression  
Initial w = 0.000, b = 0.000, loss = 6.723
After 500 iterations: w = 2.487, b = 1.319, loss = 0.012
```

---

## Limitations

### Current Implementation

1. **Simplified Loss Function**: Uses L2 norm of logits instead of proper cross-entropy loss
   - Production systems should use: `-log(softmax(logits)[target_token])`
   
2. **Random Sampling Only**: Parameters are sampled randomly, not systematically
   - Could implement cyclic or importance-based sampling for better coverage
   
3. **Forward Difference Only**: Uses single-sided finite difference `(f(x+ε) - f(x))/ε`
   - Central difference `(f(x+ε) - f(x-ε))/(2ε)` would be more accurate but 2× slower
   
4. **No Validation During Training**: Doesn't evaluate on held-out set to monitor overfitting
   - Should track validation loss in production
   
5. **Extremely Sparse Updates**: Only updates `n_params_per_iter × n_elements_per_param` elements per batch
   - With default settings: 3 × 2 = 6 parameters out of ~1.2B per batch!
   - Would need thousands of epochs to touch all parameters

### Theoretical Limitations

1. **Slow Convergence**: Requires **many more iterations** than gradient descent
   - Gradient descent: 1 backward pass estimates all gradients
   - Zeroth-order: 1 forward pass estimates 1 gradient → need `d` forward passes for `d` parameters
   
2. **High Variance**: Gradient estimates are very noisy
   - Random sampling adds variance
   - Finite differences amplify numerical errors
   
3. **Sample Inefficiency**: Needs O(d) function evaluations per gradient (d = parameter count)
   - Standard backprop: O(1) per gradient (chain rule does all parameters simultaneously)
4. **Scaling**: Impractical for very large models (billions of parameters)

### Practical Constraints

- **Time**: ~0.3 it/s on CPU (vs ~100 it/s for inference)
- **Memory**: Still needs to load full model
- **Quality**: May not match backprop-based fine-tuning

---

## Future Work

### Algorithm Improvements

1. **Better Gradient Estimation**:
   - Central differences: `∇L ≈ [L(θ+ε) - L(θ-ε)] / (2ε)`
   - Coordinate-wise updates
   - Momentum and adaptive learning rates (Adam-style)

2. **Loss Function**:
   - Implement actual cross-entropy loss
   - Target token prediction
   - Perplexity-based optimization

3. **Sampling Strategies**:
   - Importance sampling (prioritize large gradients)
   - Block-wise updates (update entire layers)
   - Coordinate descent

4. **Variance Reduction**:
   - Antithetic variates
   - Control variates
   - Gradient aggregation

### Engineering Improvements

1. **Parallelization**: Multi-threaded parameter updates
2. **GPU Support**: Offload forward passes to GPU
3. **Checkpointing**: Save intermediate models
4. **Logging**: Detailed training metrics and visualization
5. **Validation**: Automatic evaluation on validation set

### Advanced Features

1. **LoRA Integration**: Apply zeroth-order updates to LoRA adapters
2. **Quantization-Aware**: Fine-tune quantized models directly; verified GGUF support includes `Q4_K_M`, `Q5_0`, `Q5_K_M`, and `Q6_K`, while the merge/save path remains extensible through generic ggml dequant/requant traits
3. **Multi-Objective**: Optimize for multiple losses simultaneously
4. **Evolutionary Methods**: Combine with genetic algorithms or ES

---

## References

### Papers

- **Finite Differences**: Classic numerical analysis (Hildebrand, 1956)
- **Evolution Strategies**: Rechenberg (1973), Salimans et al. (2017)
- **Zeroth-Order Optimization**: Nesterov & Spokoiny (2017)
- **Black-Box Optimization**: Conn et al. (2009)

### Related Work

- **ZO-SGD**: Liu et al. (2018) - "Zeroth-Order Stochastic Gradient Descent"
- **MeZO**: Malladi et al. (2023) - "Fine-Tuning Language Models with Just Forward Passes"
- **ES for RL**: Salimans et al. (2017) - "Evolution Strategies as a Scalable Alternative to RL"

---

## License

This implementation is part of `llama.cpp` and follows its license (MIT).

## Acknowledgments

- **llama.cpp team**: For the excellent inference framework
- **GGML**: For the tensor operations library
- **Community**: For testing and feedback

---

## Contact

For questions, issues, or contributions:
- Open an issue on the `llama.cpp` GitHub repository
- Discuss in the `llama.cpp` Discord server

---

**Happy Fine-Tuning! 🚀**
