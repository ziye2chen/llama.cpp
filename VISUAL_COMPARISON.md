# Visual Comparison: First-Order vs Zeroth-Order Optimization

## 🎯 Side-by-Side Algorithm Comparison

### First-Order Optimization (Current Implementation)

```
┌─────────────────────────────────────────────────────────┐
│                 ONE TRAINING ITERATION                   │
│                                                          │
│  Input: tokens, labels, model parameters θ              │
│                                                          │
│  ┌────────────────────────────────────────┐             │
│  │  1. BUILD COMPUTATION GRAPH            │             │
│  │     gf = model.build_graph()           │             │
│  │     - Embeddings                       │             │
│  │     - Transformer layers               │             │
│  │     - Output logits                    │             │
│  └────────────────────────────────────────┘             │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────┐             │
│  │  2. ALLOCATE FOR TRAINING              │             │
│  │     ggml_opt_alloc(opt_ctx, train=TRUE)│             │
│  │     → Allocates BACKWARD graph         │             │
│  │     → Allocates gradient tensors       │             │
│  │     → Allocates optimizer state        │             │
│  └────────────────────────────────────────┘             │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────┐             │
│  │  3. EVALUATE                           │             │
│  │     ggml_opt_eval(opt_ctx, result)     │             │
│  │                                        │             │
│  │     FORWARD PASS:                      │             │
│  │     ┌──────────────────────────────┐   │             │
│  │     │ logits = forward(θ, tokens) │   │             │
│  │     │ loss = CE(logits, labels)   │   │             │
│  │     └──────────────────────────────┘   │             │
│  │              │                         │             │
│  │              ▼                         │             │
│  │     BACKWARD PASS: ← AUTOMATIC!        │             │
│  │     ┌──────────────────────────────┐   │             │
│  │     │ grad_θ = ∂loss/∂θ            │   │             │
│  │     │   (automatic differentiation)│   │             │
│  │     └──────────────────────────────┘   │             │
│  │              │                         │             │
│  │              ▼                         │             │
│  │     OPTIMIZER STEP:                    │             │
│  │     ┌──────────────────────────────┐   │             │
│  │     │ m = β₁·m + (1-β₁)·grad       │   │             │
│  │     │ v = β₂·v + (1-β₂)·grad²      │   │             │
│  │     │ θ_new = θ - α·m/√(v+ε)       │   │             │
│  │     └──────────────────────────────┘   │             │
│  └────────────────────────────────────────┘             │
│                    │                                     │
│                    ▼                                     │
│  Output: Updated parameters θ_new                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Cost: 1 forward + 1 backward = O(N) where N = #params
Memory: O(3N) for gradients + optimizer state
```

---

### Zeroth-Order Optimization (New Implementation)

```
┌─────────────────────────────────────────────────────────┐
│                 ONE TRAINING ITERATION                   │
│                                                          │
│  Input: tokens, labels, model parameters θ              │
│                                                          │
│  ┌────────────────────────────────────────┐             │
│  │  1. BUILD COMPUTATION GRAPH            │             │
│  │     gf = model.build_graph()           │             │
│  │     - Embeddings                       │             │
│  │     - Transformer layers               │             │
│  │     - Output logits                    │             │
│  └────────────────────────────────────────┘             │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────┐             │
│  │  2. ALLOCATE FOR INFERENCE ONLY        │             │
│  │     ggml_opt_alloc(opt_ctx, train=FALSE)│            │
│  │     → Allocates FORWARD graph only     │             │
│  │     → NO gradient tensors              │             │
│  │     → NO optimizer state               │             │
│  └────────────────────────────────────────┘             │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────┐             │
│  │  3. COMPUTE BASELINE LOSS              │             │
│  │     ggml_opt_eval(opt_ctx, result)     │             │
│  │                                        │             │
│  │     FORWARD PASS ONLY:                 │             │
│  │     ┌──────────────────────────────┐   │             │
│  │     │ logits = forward(θ, tokens) │   │             │
│  │     │ loss₀ = CE(logits, labels)  │   │             │
│  │     └──────────────────────────────┘   │             │
│  └────────────────────────────────────────┘             │
│                    │                                     │
│                    ▼                                     │
│  ┌────────────────────────────────────────┐             │
│  │  4. MANUAL GRADIENT ESTIMATION         │             │
│  │                                        │             │
│  │  Sample k parameters to update         │             │
│  │  For each parameter θᵢ:                │             │
│  │                                        │             │
│  │    ┌─────────────────────────────┐    │             │
│  │    │ a. θᵢ' = θᵢ + ε             │    │             │
│  │    │ b. loss' = forward(θ')      │    │ ← FORWARD   │
│  │    │ c. grad_θᵢ ≈ (loss'-loss₀)/ε│    │             │
│  │    │ d. θᵢ = θᵢ - α·grad_θᵢ       │    │             │
│  │    └─────────────────────────────┘    │             │
│  │                                        │             │
│  └────────────────────────────────────────┘             │
│                    │                                     │
│                    ▼                                     │
│  Output: Updated parameters θ_new                       │
│                                                          │
└─────────────────────────────────────────────────────────┘

Cost: (k+1) forward passes = O(k) where k = #sampled params
Memory: O(0) - no gradient storage needed!
```

---

## 📊 Detailed Comparison Table

| Feature | First-Order (Current) | Zeroth-Order (New) |
|---------|----------------------|-------------------|
| **Graph Allocation** | `ggml_opt_alloc(opt_ctx, TRUE)` | `ggml_opt_alloc(opt_ctx, FALSE)` |
| **Graph Type** | GRAD or OPT | FORWARD only |
| **Forward Passes** | 1 per iteration | k+1 per iteration (k = params sampled) |
| **Backward Pass** | ✅ Yes (automatic) | ❌ No |
| **Gradient Computation** | Automatic differentiation | Finite differences |
| **Gradient Formula** | ∂L/∂θ (exact) | ≈ (L(θ+ε) - L(θ))/ε |
| **Memory for Gradients** | O(N) | O(0) |
| **Optimizer State** | O(2N) for AdamW | O(0) |
| **Parameter Update** | Automatic (in graph) | Manual (tensor get/set) |
| **Convergence Rate** | O(1/√T) | O(1/T^(1/4)) |
| **Iterations Needed** | T | ~5-10T |
| **Best Learning Rate** | 1e-5 | 1e-6 to 1e-7 |
| **Best Batch Size** | 8-32 | 32-128 |
| **Implementation** | Complex (autodiff) | Simple (loops) |
| **Updates GGUF?** | ✅ Yes | ✅ Yes |

---

## 🔍 Code Comparison

### First-Order: What `ggml_opt_eval` Does Internally

```cpp
// Pseudo-code for first-order optimization
void ggml_opt_eval(opt_ctx, result) {
    // Forward pass
    compute_graph(forward_graph);
    loss = get_tensor_value(loss_tensor);
    
    // Backward pass (automatic)
    compute_graph(backward_graph);  // ← Computes all gradients via autodiff
    
    // Optimizer step (if opt_period reached)
    if (opt_mode) {
        for (param in parameters) {
            grad = get_gradient(param);
            
            // AdamW update
            m[param] = beta1 * m[param] + (1-beta1) * grad;
            v[param] = beta2 * v[param] + (1-beta2) * grad^2;
            m_hat = m[param] / (1 - beta1^t);
            v_hat = v[param] / (1 - beta2^t);
            
            param_data -= alpha * m_hat / (sqrt(v_hat) + epsilon);
            param_data -= weight_decay * param_data;
        }
    }
    
    result->loss = loss;
}
```

### Zeroth-Order: What Our Implementation Does

```cpp
// Zeroth-order optimization
void opt_epoch_iter_zeroth_order(...) {
    // Forward pass for baseline
    ggml_opt_alloc(opt_ctx, train=false);  // ← Forward only!
    ggml_opt_eval(opt_ctx, result);
    loss_base = result->loss.back();
    
    // Collect parameters
    params = collect_trainable_params(graph);
    
    // Sample subset (or use all)
    sampled_params = random_sample(params, n_params_per_iter);
    
    // Manual gradient estimation
    for (param in sampled_params) {
        n_elements = param->nelements();
        param_data = read_tensor(param);
        
        // Sample elements to perturb
        for (elem_idx in sample(n_elements)) {
            original = param_data[elem_idx];
            
            // Perturb
            param_data[elem_idx] = original + epsilon;
            write_tensor(param, param_data);
            
            // Forward pass
            ggml_opt_alloc(opt_ctx, train=false);
            ggml_opt_eval(opt_ctx, result);
            loss_perturbed = result->loss.back();
            
            // Estimate gradient
            grad = (loss_perturbed - loss_base) / epsilon;
            
            // SGD update
            param_data[elem_idx] = original - learning_rate * grad;
        }
        
        // Write updated parameter back
        write_tensor(param, param_data);
    }
}
```

---

## 🎨 Visual Flow Diagrams

### First-Order Flow

```
Input Data (tokens, labels)
         |
         ▼
    [Build Graph] ──────────────────┐
         |                          │
         ▼                          │
    [Allocate]                      │
    train=TRUE ───► Creates:        │ Same
         |           - Forward graph│ graph
         |           - Backward graph│ structure
         |           - Opt graph    │
         ▼                          │
    [Evaluate] ◄────────────────────┘
         |
         ├──► Forward: loss
         ├──► Backward: ∂loss/∂θ (automatic!)
         └──► Optimize: θ_new
         
         ▼
    Updated Model Parameters
```

### Zeroth-Order Flow

```
Input Data (tokens, labels)
         |
         ▼
    [Build Graph] ──────────────────┐
         |                          │
         ▼                          │
    [Allocate]                      │
    train=FALSE ──► Creates:        │ Reused
         |           - Forward graph│ multiple
         |           ONLY!          │ times
         ▼                          │
    [Eval Base Loss] ◄──────────────┤
         |                          │
         ▼                          │
    [For each param θᵢ]             │
         |                          │
         ├─► [Perturb θᵢ]           │
         ├─► [Eval Loss'] ◄─────────┤ Same
         ├─► [Grad ≈ (L'-L)/ε]      │ forward
         └─► [Update θᵢ]            │ graph
                                    │
         ▼                          │
    [Next param] ───────────────────┘
         
         ▼
    Updated Model Parameters
```

---

## 📈 Performance Characteristics

### Computational Cost vs Model Size

```
Forward Passes per Iteration:

First-Order:  ████ 1 pass
              (constant, regardless of model size)

Zeroth-Order: ████████████████████████████████████████ N passes
(full update)  (N = number of parameters)
              (scales linearly with model size)

Zeroth-Order: ████████████ k passes  
(sampled)      (k = sampled parameters, e.g., 1000)
              (constant if k is fixed)


Model Size:
Small (10K params)    │ First ≈ Zeroth (sampled)
Medium (1M params)    │ First < Zeroth (sampled)
Large (100M params)   │ First << Zeroth (sampled)
Huge (7B params)      │ First <<< Zeroth (need aggressive sampling!)
```

### Memory Usage vs Model Size

```
Memory Overhead:

Model Parameters:    ████████████████████████████████ N (baseline)

First-Order Total:   ████████████████████████████████████████████████████████████ 
                     N (model) + N (gradients) + 2N (optimizer state) = 4N

Zeroth-Order Total:  ████████████████████████████████
                     N (model) + 0 (no gradients) = N

Savings: 3N (75% reduction!)
```

---

## 🧮 Mathematical Comparison

### Gradient Computation

**First-Order (Exact)**:
```
∂L/∂θᵢ = lim[ε→0] (L(θ + ε·eᵢ) - L(θ)) / ε

Computed via chain rule (backpropagation):
∂L/∂θᵢ = ∂L/∂f · ∂f/∂g · ... · ∂h/∂θᵢ

Properties:
- Exact (up to numerical precision)
- O(N) complexity via reverse-mode autodiff
- Requires storing intermediate activations
```

**Zeroth-Order (Approximate)**:
```
∂L/∂θᵢ ≈ (L(θ + ε·eᵢ) - L(θ)) / ε

Error bound:
|∂L/∂θᵢ - (L(θ+ε·eᵢ) - L(θ))/ε| ≤ O(ε·||∇²L||)

Properties:
- Approximate (error depends on ε and curvature)
- O(N) forward passes (or O(k) if sampling)
- No intermediate storage needed
```

### Parameter Update Rules

**First-Order (AdamW)**:
```
mᵢ = β₁·mᵢ + (1-β₁)·∂L/∂θᵢ
vᵢ = β₂·vᵢ + (1-β₂)·(∂L/∂θᵢ)²
m̂ᵢ = mᵢ / (1-β₁ᵗ)
v̂ᵢ = vᵢ / (1-β₂ᵗ)
θᵢ ← θᵢ - α·m̂ᵢ/√(v̂ᵢ+ε) - λ·θᵢ
```

**Zeroth-Order (SGD)**:
```
θᵢ ← θᵢ - α·(L(θ+ε·eᵢ) - L(θ))/ε - λ·θᵢ

Simplified:
θᵢ ← θᵢ - α·ĝᵢ - λ·θᵢ

where ĝᵢ is the finite-difference estimate
```

---

## 🎯 When to Use Each Method

### Use First-Order When:
```
✅ Standard fine-tuning scenarios
✅ Have sufficient memory (can store gradients)
✅ Need fast convergence
✅ Model has millions/billions of parameters
✅ Production training pipelines
✅ Backpropagation is well-tested and working
```

### Use Zeroth-Order When:
```
✅ Backpropagation unavailable/broken
✅ Extremely memory-constrained environments
✅ Model has non-differentiable operations
✅ Research into gradient-free methods
✅ Educational purposes (teaching optimization)
✅ Verifying gradient implementations
✅ Optimizing discrete/combinatorial objectives
```

---

## 💡 Key Insight

The fundamental difference is:

**First-Order**: 
- Tells the model "here's exactly how wrong each parameter is" (∂L/∂θ)
- Fast but requires complex machinery (autodiff)

**Zeroth-Order**:
- Asks the model "what happens if I wiggle this parameter?" (L(θ+ε) - L(θ))
- Slow but conceptually simple

Both ultimately do the same thing: **update GGUF model parameters to reduce loss**.

The choice is a **tradeoff** between:
- **Speed vs Simplicity**
- **Memory vs Computation**
- **Exact vs Approximate**

---

## 📝 Summary

| Aspect | First-Order | Zeroth-Order |
|--------|------------|--------------|
| **Core Idea** | Use calculus | Approximate calculus |
| **Implementation** | `backward=true` | `backward=false` + loops |
| **Cost** | 1 forward + 1 backward | k+1 forward |
| **Memory** | 4× model size | 1× model size |
| **Speed** | ⚡⚡⚡ Fast | ⚡ Slower |
| **Complexity** | 🔴 Complex | 🟢 Simple |
| **Updates GGUF?** | ✅ Yes | ✅ Yes |

**Both methods successfully update GGUF model parameters!**



