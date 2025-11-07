# Analysis of GGUF Optimization and Zeroth-Order Implementation

## Current Optimization Flow Analysis

### 1. **opt_epoch_iter Function** (llama-context.cpp:2110-2217)

The current first-order optimization process works as follows:

#### Data Flow:
1. **Batch Processing**: Iterates through context window in batches (n_ctx chunks)
2. **Graph Building**: For each ubatch:
   - Creates computation graph via `model.build_graph()`
   - Allocates compute context for gradients
3. **Optimization Setup**:
   - `ggml_opt_prepare_alloc()`: Prepares graph, inputs, outputs
   - `ggml_opt_alloc(opt_ctx, train)`: Allocates graph based on mode
     - When `train=true`: Allocates GRAD or OPT graph (backward pass)
     - When `train=false`: Allocates FORWARD graph only
4. **Label Setup**: Converts sparse labels to one-hot encoding
5. **Evaluation**: `ggml_opt_eval()` computes forward + backward + optimizer step
6. **Callback**: Reports progress

#### Key Components:
- **Parameters**: Model tensors marked with `GGML_TENSOR_FLAG_PARAM` 
- **Gradients**: Automatically computed via automatic differentiation
- **Optimizers**: AdamW or SGD update parameters using gradients
- **Loss**: Cross-entropy loss computed from outputs and labels

### 2. **ggml_opt_alloc Function** (ggml-opt.cpp:724-779)

This function determines which computation graph to allocate:

- **FORWARD**: Only forward pass (inference mode)
- **GRAD**: Forward + gradient computation (accumulate gradients)
- **OPT**: Forward + gradient + optimizer step (update parameters)

The backward pass is triggered by the `backward` parameter. When true:
- Computes gradients via automatic differentiation
- Accumulates gradients over opt_period
- Applies optimizer step to update parameters

### 3. **Parameter Updates**

In the current implementation:
1. **Forward**: Compute loss = f(params)
2. **Backward**: Compute ∇loss = ∂f/∂params (via autodiff)
3. **Optimizer**: params_new = optimizer_step(params, ∇loss)

The actual parameter update happens in the optimizer step within `ggml_opt_eval()`:
- AdamW: Updates using momentum and adaptive learning rates
- SGD: Simple gradient descent with weight decay

---

## Zeroth-Order Optimization Design

### Concept

Zeroth-order optimization estimates gradients using only function evaluations (forward passes), without backpropagation:

**Finite Difference Approximation**:
```
∂f/∂θᵢ ≈ (f(θ + εeᵢ) - f(θ)) / ε
```

Where:
- θ is parameter vector
- eᵢ is unit vector for parameter i
- ε is perturbation size (e.g., 1e-4)

### Advantages:
- No backward pass needed
- Memory efficient (no gradient storage)
- Can optimize non-differentiable operations

### Disadvantages:
- More forward passes required (2 per parameter)
- Slower convergence
- Sensitive to noise

### Implementation Strategy

We'll implement **Coordinate-wise Finite Differences with SGD**:

1. For each parameter tensor:
   - Perturb each element by +ε
   - Compute forward pass and loss
   - Estimate gradient: grad ≈ (loss_perturbed - loss_base) / ε
   - Update parameter: param -= learning_rate * grad

2. **Optimization**: 
   - Only perturb a random subset of parameters per iteration
   - Use larger batch sizes to reduce noise
   - Adaptive ε based on parameter magnitude



