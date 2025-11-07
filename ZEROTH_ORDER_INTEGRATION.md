# Zeroth-Order Optimization Integration Guide

## Overview

This document provides a comprehensive guide for integrating zeroth-order optimization into existing llama.cpp training workflows.

## Architecture Analysis

### Current First-Order Optimization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     opt_epoch_iter                              │
│                                                                 │
│  1. Process batches through context window                     │
│  2. Build computation graph (model.build_graph)                │
│  3. Setup optimization context:                                │
│     - ggml_opt_prepare_alloc(opt_ctx, ctx, gf, inputs, outputs)│
│     - ggml_opt_alloc(opt_ctx, train=true)                      │
│  4. Set labels (sparse → one-hot)                              │
│  5. Execute optimization:                                      │
│     - ggml_opt_eval(opt_ctx, result)                           │
│       → Forward pass (compute loss)                            │
│       → Backward pass (compute gradients via autodiff)         │
│       → Optimizer step (update parameters using gradients)     │
│  6. Invoke callback (progress reporting)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Zeroth-Order Optimization Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                opt_epoch_iter_zeroth_order                      │
│                                                                 │
│  1. Process batches through context window                     │
│  2. Build computation graph (model.build_graph)                │
│  3. Setup optimization context:                                │
│     - ggml_opt_prepare_alloc(opt_ctx, ctx, gf, inputs, outputs)│
│     - ggml_opt_alloc(opt_ctx, train=false) ← FORWARD ONLY     │
│  4. Set labels (sparse → one-hot)                              │
│  5. Collect trainable parameters from graph                    │
│  6. Compute baseline loss (forward pass)                       │
│  7. For each parameter (or sampled subset):                    │
│     a. Save original value                                     │
│     b. Perturb: param += ε                                     │
│     c. Compute perturbed loss (forward pass)                   │
│     d. Estimate gradient: grad ≈ (loss' - loss) / ε           │
│     e. Update: param = original - α * grad                     │
│  8. Invoke callback (progress reporting)                       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Differences

| Aspect | First-Order | Zeroth-Order |
|--------|-------------|--------------|
| **Graph Allocation** | `ggml_opt_alloc(opt_ctx, true)` | `ggml_opt_alloc(opt_ctx, false)` |
| **Graph Type** | GRAD or OPT | FORWARD only |
| **Gradient Computation** | Automatic differentiation | Finite differences |
| **Forward Passes** | 1 per batch | 1 + N per batch (N = params updated) |
| **Memory for Gradients** | Yes (full gradient tensors) | No |
| **Optimizer State** | Adam momenta, etc. | None (we use simple SGD) |
| **Parameter Updates** | In optimizer graph | Manual (via tensor_get/set) |

## Implementation Details

### 1. Parameter Collection

```cpp
std::vector<struct ggml_tensor *> collect_trainable_params(struct ggml_cgraph * gf) {
    std::vector<struct ggml_tensor *> params;
    for (int i = 0; i < gf->n_nodes; ++i) {
        struct ggml_tensor * node = gf->nodes[i];
        if (node->flags & GGML_TENSOR_FLAG_PARAM) {
            params.push_back(node);
        }
    }
    return params;
}
```

Parameters are identified by the `GGML_TENSOR_FLAG_PARAM` flag, which is set during model initialization via `ggml_set_param()`.

### 2. Loss Computation (Forward Only)

```cpp
// Setup for forward pass
ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, inputs, outputs);
ggml_opt_alloc(opt_ctx, /*backward =*/ false);  // ← Key difference

// Evaluate
struct ggml_opt_result result;
ggml_opt_eval(opt_ctx, &result);

// Extract loss
float loss = result.loss.back();
```

### 3. Gradient Estimation

```cpp
// Baseline loss
float loss_base = compute_forward_loss(params);

// For each parameter element
float original_value = param_data[idx];
float epsilon = 1e-4f * max(1.0f, abs(original_value));

// Perturb
param_data[idx] = original_value + epsilon;
update_parameter(param, param_data);

// Compute perturbed loss
float loss_perturbed = compute_forward_loss(params);

// Estimate gradient
float grad = (loss_perturbed - loss_base) / epsilon;
```

### 4. Parameter Update

```cpp
// SGD update with weight decay
float update = learning_rate * (grad + weight_decay * original_value);
param_data[idx] = original_value - update;

// Write back to tensor
ggml_backend_tensor_set(param, param_data, 0, n_elements * sizeof(float));
```

## Integration into llama-context.cpp

### Option 1: Add to Existing Class

Add the zeroth-order method to `llama_context` class:

```cpp
// In llama-context.h
class llama_context {
    // ... existing members ...
    
    void opt_epoch_iter_zeroth_order(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        const llama_zeroth_order_params & zo_params,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start);
};
```

### Option 2: Wrapper Function

Create a wrapper that switches between methods:

```cpp
enum llama_opt_method {
    LLAMA_OPT_METHOD_FIRST_ORDER,
    LLAMA_OPT_METHOD_ZEROTH_ORDER,
};

void llama_context::opt_epoch_iter_adaptive(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        llama_opt_method                 method,
        void *                           method_params,
        ...) {
    
    if (method == LLAMA_OPT_METHOD_ZEROTH_ORDER) {
        opt_epoch_iter_zeroth_order(
            dataset, result, tokens, labels_sparse, batch, callback,
            *(llama_zeroth_order_params*)method_params,
            idata_in_loop, ndata_in_loop, t_loop_start);
    } else {
        opt_epoch_iter(
            dataset, result, tokens, labels_sparse, batch, callback,
            true,  // train
            idata_in_loop, ndata_in_loop, t_loop_start);
    }
}
```

## Usage Example

### In Training Code (e.g., finetune.cpp)

```cpp
#include "llama-zeroth-order-opt.h"

// Initialize optimization context as usual
llama_opt_params lopt_params;
lopt_params.n_ctx_train = n_ctx;
lopt_params.optimizer_type = GGML_OPT_OPTIMIZER_TYPE_ADAMW;  // Not used in ZO
ctx->opt_init(model, lopt_params);

// Configure zeroth-order parameters
llama_zeroth_order_params zo_params = llama_zeroth_order_default_params();
zo_params.epsilon = 1e-4f;
zo_params.learning_rate = 1e-6f;  // Much smaller than first-order!
zo_params.n_params_per_iter = 1000;  // Sample 1000 params per iteration
zo_params.use_random_sampling = true;

// Run training epoch
for (int epoch = 0; epoch < n_epochs; ++epoch) {
    // Prepare dataset, batch, tokens, labels...
    
    // Use zeroth-order optimization
    ctx->opt_epoch_iter_zeroth_order(
        dataset,
        result_train,
        tokens,
        labels_sparse,
        batch,
        progress_callback,
        zo_params,
        idata,
        ndata,
        t_start);
}
```

## Performance Tuning

### 1. Epsilon Selection

The perturbation size `ε` is critical:

- **Too large**: Inaccurate gradient estimates
- **Too small**: Numerical precision issues

**Recommendation**: Use adaptive epsilon based on parameter magnitude:
```cpp
epsilon = base_epsilon * max(1.0f, abs(param_value))
```

Typical values:
- `base_epsilon`: 1e-4 to 1e-3
- For parameters in [0, 1]: ε ≈ 1e-4
- For parameters in [-10, 10]: ε ≈ 1e-3

### 2. Learning Rate

Zeroth-order methods need different learning rates than first-order:

| Model Size | First-Order LR | Zeroth-Order LR |
|------------|----------------|-----------------|
| Small (<100M params) | 1e-4 | 1e-5 to 1e-6 |
| Medium (100M-1B) | 1e-5 | 1e-6 to 1e-7 |
| Large (>1B) | 1e-6 | 1e-7 to 1e-8 |

### 3. Parameter Sampling

For large models, updating all parameters is infeasible:

```cpp
// Sample a fixed number per iteration
zo_params.n_params_per_iter = min(1000, total_params / 100);

// Or sample a percentage
zo_params.n_params_per_iter = total_params * 0.01;  // 1%
```

### 4. Batch Size

Larger batches reduce noise in loss estimates:

- **First-order**: batch_size = 8-32
- **Zeroth-order**: batch_size = 32-128 (4x larger)

## Theoretical Justification

### Gradient Approximation Error

The finite difference approximation has error:
```
|∇f(θ) - (f(θ+ε) - f(θ))/ε| ≤ O(ε·||∇²f||)
```

Where `||∇²f||` is the Hessian norm (curvature).

**Implication**: Flat regions (small curvature) have better approximations.

### Convergence Rate

For smooth, convex functions:
- **First-order (SGD)**: O(1/√T) after T iterations
- **Zeroth-order (finite diff)**: O(1/T^(1/4)) after T iterations

**Implication**: ~16x more iterations needed for same convergence.

### Sample Complexity

For n parameters:
- **First-order**: O(1) gradient evaluations per iteration
- **Zeroth-order**: O(n) or O(k) if sampling k parameters

**Implication**: Use aggressive sampling for large models.

## Limitations and Caveats

### 1. Not a Drop-in Replacement

Zeroth-order optimization requires:
- Different hyperparameters (smaller LR, larger epsilon)
- More iterations to converge
- Potentially different batch sizes

### 2. Scalability

For a model with 1B parameters:
- Full update: 1B forward passes per iteration (infeasible)
- Sampled update (0.1%): 1M forward passes per iteration (slow)
- Solution: Use very aggressive sampling or layer-wise updates

### 3. Numerical Stability

Finite differences are sensitive to:
- Floating point precision (use FP32, not FP16 for gradients)
- Loss magnitude (normalize or use relative epsilon)
- Batch-to-batch variance (increase batch size)

### 4. Model Saving

The current implementation modifies parameters in-place. To save:

```cpp
// After training, parameters are already updated in model tensors
// Save model as usual
llama_save_model(model, "output.gguf");
```

## Debugging Tips

### 1. Verify Gradient Estimates

Compare finite difference gradients with analytical gradients on a small model:

```cpp
// Compute both
float grad_analytical = /* from backprop */;
float grad_finite_diff = (loss_plus - loss_base) / epsilon;

// Should be close
assert(abs(grad_analytical - grad_finite_diff) < 1e-3);
```

### 2. Monitor Loss

Loss should decrease (eventually):

```
Epoch 0: loss = 3.456
Epoch 1: loss = 3.423  ← decreasing (good)
Epoch 2: loss = 3.401
...
```

If loss increases or fluctuates wildly:
- Reduce learning rate
- Increase batch size
- Adjust epsilon

### 3. Check Parameter Updates

Print parameter statistics:

```cpp
printf("Param min: %.6f, max: %.6f, mean: %.6f\n", 
       param_min, param_max, param_mean);
```

Values should change gradually, not explode.

## Future Extensions

### 1. SPSA (Simultaneous Perturbation)

Perturb all parameters simultaneously with random directions:

```cpp
// Generate random direction
vector<float> direction(n_params);
for (auto& d : direction) d = random_sign();  // ±1

// Perturb all at once
for (int i = 0; i < n_params; ++i) {
    params[i] += epsilon * direction[i];
}

// Single forward pass!
float loss_plus = compute_loss();

// Gradient estimate
for (int i = 0; i < n_params; ++i) {
    grad[i] = direction[i] * (loss_plus - loss_base) / epsilon;
}
```

**Advantage**: O(1) forward passes instead of O(n)!

### 2. Evolution Strategies

Use population-based methods:

```cpp
// Generate population of perturbed parameters
vector<Params> population = generate_perturbations(params, sigma, n_pop);

// Evaluate all
vector<float> losses(n_pop);
for (int i = 0; i < n_pop; ++i) {
    losses[i] = compute_loss(population[i]);
}

// Update using weighted average
params_new = weighted_sum(population, softmax(-losses));
```

### 3. Layer-wise Updates

Update one layer at a time:

```cpp
for (auto& layer : model.layers) {
    update_layer_parameters(layer, zo_params);
}
```

**Advantage**: Reduces number of parameters per iteration.

## Conclusion

Zeroth-order optimization provides a gradient-free alternative to backpropagation for fine-tuning GGUF models. While slower than first-order methods, it offers unique advantages:

- **Simplicity**: Only requires forward pass implementation
- **Memory efficiency**: No gradient storage
- **Flexibility**: Can optimize non-differentiable objectives

Use it when backpropagation is unavailable, memory is constrained, or you need to explore alternative optimization strategies.

## References

1. Nesterov & Spokoiny (2017) - Random gradient-free minimization
2. Liu et al. (2018) - Zeroth-order optimization for deep learning
3. Spall (1992) - SPSA algorithm
4. Salimans et al. (2017) - Evolution strategies for RL



