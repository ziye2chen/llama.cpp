## GGUF Tensors and Per-Element Zeroth-Order Gradient Descent

This report explains, in simple language, how we go from **GGUF tensors** (with different shapes) to **per-element gradient descent** using zeroth-order optimization (ZO) and R-AdaZO in `llama.cpp`.

It connects three main ideas:

- **Tensor storage**: how tensors are stored in GGUF models (shapes vs memory)
- **Per-element updates**: how we pick and perturb a single element inside a tensor
- **Gradient direction**: how a single global loss tells us which way to move that element

---

## 1. How GGUF Tensors Are Stored

### 1.1 Logical Shapes vs Physical Storage

When you load a GGUF model, you see many tensors like:

- `token_embd.weight` → shape `[128256, 2048]`
- `output_norm.weight` → shape `[2048]`
- `blk.0.attn_q.weight` → shape `[2048, 2048]`
- `blk.0.attn_k.weight` → shape `[2048, 512]`
- `blk.0.ffn_gate.weight` → shape `[2048, 8192]`
- ...

These shapes are the **logical view** – how we *think* about the data (vectors, matrices).

Inside memory, however, **every tensor is just a flat 1D array of floats**:

- `blk.0.attn_q.weight` → 2048 × 2048 = **4,194,304 floats in a row**
- `blk.0.attn_k.weight` → 2048 × 512  = **1,048,576 floats in a row**
- `blk.0.ffn_gate.weight` → 2048 × 8192 = **16,777,216 floats in a row**
- `output_norm.weight` → 2048 floats in a row

So even though we think in 2D, in memory each tensor looks like:

```text
param[0], param[1], param[2], ..., param[n_elements - 1]
```

### 1.2 Getting the Number of Elements

In code, we use `ggml_nelements(param)` to get the number of elements in a tensor:

```cpp
struct ggml_tensor * param = ...;              // e.g. blk.0.attn_q.weight
int64_t n_elements = ggml_nelements(param);    // e.g. 4'194'304
```

This works for any shape because internally it multiplies the dimensions.

Once we know `n_elements`, we can index elements from `0` to `n_elements - 1` using a **single linear index**.

---

## 2. Reading and Writing a Single Element

### 2.1 Reading One Element

To read a single element from a tensor, we use `ggml_backend_tensor_get` with a **byte offset**:

```cpp
int64_t elem_idx = 123456;       // which element we want
float current_val = 0.0f;

ggml_backend_tensor_get(
    param,                       // tensor (drawer)
    &current_val,                // destination
    elem_idx * sizeof(float),    // byte offset
    sizeof(float));              // size (4 bytes)
```

This works the same whether `param` is:

- `[2048, 2048]` (attention weight matrix),
- `[2048]` (normalization vector), or
- `[128256, 2048]` (embedding matrix).

### 2.2 Writing One Element

After computing an update, we write the new value back with `ggml_backend_tensor_set`:

```cpp
float new_val = ...;  // value computed by the optimizer

ggml_backend_tensor_set(
    param,                       // tensor
    &new_val,                    // source
    elem_idx * sizeof(float),    // byte offset
    sizeof(float));              // size (4 bytes)

llama_synchronize(ctx);          // ensure write is committed
```

**Key point:** tensor *shape* never appears here. We only need:

- the tensor pointer (`param`),
- the linear index (`elem_idx`),
- the byte offset (`elem_idx * sizeof(float)`).

The GGML backend takes care of where this lives physically.

---

## 3. The Loss Is Global, Not Per-Element

The model has **one scalar loss per batch**, not one loss per tensor element.

For a given batch:

1. We run a forward pass with the **current** parameters:
   ```cpp
   llama_decode(ctx, batch);
   float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
   float loss_base = compute_loss(logits, n_vocab); // e.g. 250.3
   ```
2. `loss_base` is a **global measure** of how good the model’s output is for that whole batch.

When we perturb a single element, we run another forward pass and get a **new global loss**. The difference between these two losses tells us how that one element affects the **overall** model performance.

---

## 4. Zeroth-Order Gradient for One Element

### 4.1 Simple Finite Difference View

Consider a single element `θ_i` (for example, element `123456` in `blk.0.attn_q.weight`).

We can estimate its gradient using **finite differences**:

```text
∂L/∂θ_i ≈ (L(θ_i + ε) - L(θ_i)) / ε
```

Algorithm:

1. Run forward with current value `θ_i` → get `loss_base = L(θ_i)`.
2. Change only `θ_i` to `θ_i + ε`.
3. Run forward again → get `loss_perturbed = L(θ_i + ε)`.
4. Estimate gradient:
   ```cpp
   float gradient = (loss_perturbed - loss_base) / epsilon;
   ```

Interpretation:

- If `gradient > 0`: increasing `θ_i` made the loss **worse** → we should **decrease** `θ_i`.
- If `gradient < 0`: increasing `θ_i` made the loss **better** → we should **increase** `θ_i`.

Even though we change only one element, the loss is still a **global** number, so each gradient describes how that element affects the overall model behavior.

### 4.2 R-AdaZO’s Random Direction Estimation

R-AdaZO uses a slightly more advanced estimator:

```text
∇L(θ) ≈ (L(θ + μ·u) - L(θ)) · u / μ
```

Where:

- `u` is a random direction vector (one value per element),
- `μ` is a small perturbation magnitude,
- we average over `n_samples` random `u`’s to reduce noise.

For a single element `θ_i`:

1. Sample `u_i` (the perturbation direction for this element).
2. Set `θ_i' = θ_i + μ·u_i` and run forward to get `L(θ + μ·u)`.
3. Compute:
   ```cpp
   float grad_sample = (loss_perturbed - loss_base) * u_val / mu;
   ```
4. Average over multiple samples:
   ```cpp
   estimated_gradient = sum(grad_sample) / n_samples;
   ```

This gives a **noisy but unbiased** estimate of the gradient for that element, again derived from global loss changes.

---

## 5. Gradient Descent on a Single Element

Once we have an estimated gradient for element `θ_i`, we apply standard gradient descent:

```cpp
float learning_rate = params_.lr;
float new_val = current_val - learning_rate * estimated_gradient;
```

- If `estimated_gradient > 0` → `new_val < current_val` (we move left).
- If `estimated_gradient < 0` → `new_val > current_val` (we move right).

Finally, we write `new_val` back to the tensor at index `elem_idx` with `ggml_backend_tensor_set`.

R-AdaZO refines this step by using **Adam-style adaptive updates**:

```cpp
state.m[elem_idx] = beta1 * state.m[elem_idx] + (1.0f - beta1) * estimated_gradient;
state.v[elem_idx] = beta2 * state.v[elem_idx] + (1.0f - beta2) * state.m[elem_idx] * state.m[elem_idx];

float m_hat = state.m[elem_idx] / (1.0f - std::pow(beta1, global_step));
float v_hat = state.v[elem_idx] / (1.0f - std::pow(beta2, global_step));

float update = params_.lr * m_hat / (std::sqrt(v_hat) + params_.eps);
float new_val = current_val - update;
```

But conceptually, it is still **gradient descent on a single scalar**, using a better estimate of both direction and step size.

---

## 6. How Shape Differences Are Handled

Different tensors have different shapes, for example:

- `blk.0.attn_q.weight` → `[2048, 2048]` → 4,194,304 elements
- `blk.0.attn_k.weight` → `[2048, 512]` → 1,048,576 elements
- `output_norm.weight` → `[2048]` → 2,048 elements

In the optimizer, we treat them all in the same way:

1. Use `ggml_nelements(param)` to get `n_elements`.
2. Sample a random linear index `elem_idx` in `[0, n_elements)`.
3. Read, perturb, estimate gradient, and update **that one index**.

We never need to compute row/column positions. The GGML tensor stores the shape if we need it, but the optimizer logic only needs:

- the tensor pointer,
- the total number of elements,
- a linear index.

This makes the code **shape-agnostic** and works for any GGUF model that exposes its tensors.

---

## 7. Big Picture Intuition

- A GGUF model is like a cabinet full of drawers (tensors).
- Each drawer contains a long line of numbers (tensor elements).
- Zeroth-order optimization:
  - picks one drawer,
  - picks one element inside that drawer,
  - nudges that element a little,
  - checks whether the **overall** model got better or worse,
  - and then moves that element in the direction that improves the **global** loss.
- R-AdaZO does this more intelligently by:
  - averaging over multiple random perturbations,
  - keeping per-element momentum and variance,
  - adapting the step size for each individual element.

So your mental model is correct:

> We are doing **gradient descent on individual scalar elements** inside tensors, but the gradient for each element is computed from how that element changes the **global loss** of the entire model.

The fact that tensors have different shapes does not change the core logic, because all shapes reduce to flat arrays when we read/write elements through GGML.

{
  "cells": [],
  "metadata": {
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}