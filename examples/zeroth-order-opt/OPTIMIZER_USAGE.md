# Zeroth-Order Optimizer - Plug-and-Play Usage Guide

This document explains how to use the refactored zeroth-order optimizer as a standalone, reusable component.

## Overview

The optimizer has been refactored into a clean, plug-and-play design:

- **`zeroth-order-optimizer.h`**: Header file with the optimizer interface
- **`zeroth-order-optimizer.cpp`**: Implementation of the optimizer logic
- **`finetune-zeroth-order.cpp`**: Example usage/demo program

## Basic Usage

### 1. Include the Header

```cpp
#include "zeroth-order-optimizer.h"
```

### 2. Collect Trainable Parameters

Use the provided helper function to collect parameters from your model:

```cpp
// Collect parameters (attention and FFN weights)
auto trainable_params = collect_trainable_parameters(ctx, verbose=true);
```

This automatically filters for key weight matrices:
- Attention: `attn_q`, `attn_k`, `attn_v`, `attn_output`
- FFN: `ffn_gate`, `ffn_up`, `ffn_down`

### 3. Configure Optimizer Parameters

```cpp
zeroth_order_params zo_params;
zo_params.epsilon = 1e-3f;              // Perturbation size for gradient estimation
zo_params.learning_rate = 1e-4f;        // Learning rate
zo_params.weight_decay = 0.01f;         // L2 regularization
zo_params.n_params_per_iter = 3;        // Parameters to sample per batch
zo_params.n_elements_per_param = 2;     // Elements to update per parameter
zo_params.log_gradients = false;        // Enable verbose logging
```

**Note**: Total forward passes per batch = `1 + (n_params_per_iter × n_elements_per_param)`

### 4. Create the Optimizer

```cpp
ZerothOrderOptimizer optimizer(zo_params, trainable_params);
```

### 5. Training Loop

```cpp
for (int epoch = 0; epoch < n_epochs; ++epoch) {
    for (size_t i = 0; i + n_batch < train_tokens.size(); i += n_batch) {
        // Clear KV cache
        llama_memory_clear(llama_get_memory(ctx), true);
        
        // Prepare batch
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        // ... fill batch with tokens ...
        
        // Baseline forward pass
        llama_decode(ctx, batch);
        
        // *** SINGLE LINE: Perform all gradient estimation and updates ***
        float loss = optimizer.step(ctx, batch, n_vocab);
        
        // Free batch
        llama_batch_free(batch);
    }
}
```

### 6. Get Statistics

```cpp
LOG_INF("Total parameter updates: %lld\n", optimizer.get_total_updates());
LOG_INF("Total forward passes: %lld\n", optimizer.get_forward_passes());
```

## Complete Example

Here's a minimal working example:

```cpp
#include "llama.h"
#include "zeroth-order-optimizer.h"

int main() {
    // 1. Load model and context (standard llama.cpp code)
    llama_context * ctx = /* your context */;
    std::vector<llama_token> train_tokens = /* your data */;
    int n_vocab = /* vocab size */;
    
    // 2. Collect trainable parameters
    auto trainable_params = collect_trainable_parameters(ctx, true);
    
    // 3. Configure optimizer
    zeroth_order_params zo_params;
    zo_params.epsilon = 1e-3f;
    zo_params.learning_rate = 1e-4f;
    zo_params.n_params_per_iter = 3;
    zo_params.n_elements_per_param = 2;
    
    // 4. Create optimizer
    ZerothOrderOptimizer optimizer(zo_params, trainable_params);
    
    // 5. Training loop
    const int n_batch = 64;
    for (size_t i = 0; i + n_batch < train_tokens.size(); i += n_batch) {
        llama_memory_clear(llama_get_memory(ctx), true);
        
        llama_batch batch = llama_batch_init(n_batch, 0, 1);
        for (int j = 0; j < n_batch; ++j) {
            batch.token[j] = train_tokens[i + j];
            batch.pos[j] = j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (j == n_batch - 1);
        }
        batch.n_tokens = n_batch;
        
        llama_decode(ctx, batch);
        
        // *** SINGLE CALL: All optimization happens here ***
        float loss = optimizer.step(ctx, batch, n_vocab);
        
        printf("Loss: %.6f\n", loss);
        
        llama_batch_free(batch);
    }
    
    // 6. Statistics
    printf("Updates: %lld, Forward passes: %lld\n",
           optimizer.get_total_updates(),
           optimizer.get_forward_passes());
    
    return 0;
}
```

## API Reference

### `ZerothOrderOptimizer` Class

#### Constructor
```cpp
ZerothOrderOptimizer(
    const zeroth_order_params & params,
    const std::vector<struct ggml_tensor *> & trainable_params
)
```

#### Methods

**`float step(llama_context * ctx, llama_batch & batch, int n_vocab)`**
- Performs one optimization step on a batch
- Returns the baseline loss
- Internally handles:
  - Loss computation
  - Parameter perturbation
  - Forward passes with perturbed parameters
  - Gradient estimation via finite differences
  - Parameter updates with weight decay

**`int64_t get_total_updates() const`**
- Returns total number of parameter updates performed

**`int64_t get_forward_passes() const`**
- Returns total number of forward passes executed

### Helper Functions

**`collect_trainable_parameters(llama_context * ctx, bool verbose = true)`**
- Collects trainable weight matrices from the model
- Filters for attention and FFN weights
- Returns `std::vector<struct ggml_tensor *>`

### `zeroth_order_params` Structure

```cpp
struct zeroth_order_params {
    float epsilon;                   // Perturbation size (default: 1e-3f)
    float learning_rate;             // Learning rate (default: 1e-4f)
    float weight_decay;              // L2 regularization (default: 0.01f)
    int32_t n_params_per_iter;       // Parameters per iteration (default: 3)
    int32_t n_elements_per_param;    // Elements per parameter (default: 2)
    bool use_random_sampling;        // Use random sampling (default: true)
    uint32_t random_seed;            // Random seed (default: 42)
    bool log_gradients;              // Verbose logging (default: false)
};
```

## Tuning Guidelines

### Epsilon (`epsilon`)
- **Too small** (< 1e-5): Numerical instability, noisy gradients
- **Too large** (> 1e-1): Poor gradient approximation
- **Recommended**: Start with `1e-3` to `1e-4`

### Learning Rate (`learning_rate`)
- Much smaller than standard SGD due to noisy gradient estimates
- **Recommended**: Start with `1e-4` to `1e-3`, adjust based on loss behavior

### Parameters per Iteration (`n_params_per_iter`, `n_elements_per_param`)
- **Trade-off**: More parameters = slower but more stable updates
- **Forward passes per batch** = `1 + (n_params_per_iter × n_elements_per_param)`
- **Recommended**: Start with `n_params_per_iter=3`, `n_elements_per_param=2` (7 forward passes)

### Weight Decay (`weight_decay`)
- Prevents overfitting, especially on small datasets
- **Recommended**: `0.01` to `0.1`

## Performance Characteristics

- **Speed**: ~7× slower than standard inference per batch (with default settings)
- **Memory**: Lower than backpropagation (no gradient storage)
- **Convergence**: Slower than first-order methods, requires more epochs
- **Scalability**: Best for small models or specific layer fine-tuning

## When to Use

✅ **Good for:**
- Frameworks without backpropagation support (like `llama.cpp`)
- Memory-constrained environments
- Black-box optimization scenarios
- Research and education

❌ **Not recommended for:**
- Large-scale training (millions of examples)
- When backpropagation is available and memory is sufficient
- Time-critical training scenarios

## Build Instructions

Add to your `CMakeLists.txt`:

```cmake
add_executable(your_program
    your_program.cpp
    zeroth-order-optimizer.cpp
    zeroth-order-optimizer.h
)
target_link_libraries(your_program PRIVATE ggml llama common)
target_compile_features(your_program PRIVATE cxx_std_11)
target_include_directories(your_program PRIVATE ${CMAKE_SOURCE_DIR}/src)
```

## Example Output

```
collect_trainable_parameters: collected 112 trainable parameter tensors

Optimizer Configuration:
  epsilon = 1.00e-03 (perturbation size)
  learning_rate = 1.00e-04
  weight_decay = 1.00e-02
  n_params_per_iter = 3
  n_elements_per_param = 2
  Forward passes per batch = 7 (1 baseline + 6 perturbed)

===== Epoch 1/1 =====
[TRAIN] Iter     10/    91 | Loss: 347.387 | Time:  31.23s | 0.3 it/s

Optimizer Statistics:
  Total parameter updates: 546
  Total forward passes: 637
```

## License

This optimizer is part of the `llama.cpp` project and follows its license.

