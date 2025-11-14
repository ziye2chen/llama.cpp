# Fine-Tuning Types Comparison

This document explains the different types of fine-tuning and where our zeroth-order optimizers fit.

## Fine-Tuning Types

### 1. Full Parameter Fine-Tuning (Full FT)

**What it is:**
- Updates **ALL** model parameters
- Directly modifies original weights
- Most expressive, most memory-intensive

**Parameters updated:**
- ✅ All attention weights
- ✅ All FFN weights
- ✅ All embeddings
- ✅ All layer norms
- ✅ All biases
- ✅ Everything!

**Memory:** Very high (need gradients for all parameters)

**Our implementation:** ❌ **NOT this** - We only update ~76% of parameters

---

### 2. LoRA (Low-Rank Adaptation)

**What it is:**
- Adds **low-rank adapter matrices** (A and B)
- Keeps original weights **frozen**
- Computes: `W_new = W_original + B × A`
- Only trains A and B (much smaller)

**Parameters updated:**
- ✅ Only adapter matrices A and B
- ❌ Original weights stay frozen

**Memory:** Low (only gradients for adapters)

**Our implementation:** ❌ **NOT this** - We directly modify original weights

**Example:**
```
Original: W (2048×2048) = 4M parameters
LoRA:     A (2048×r), B (r×2048) where r=8
          = 2×2048×8 = 32K parameters (0.8% of original!)
```

---

### 3. Selective/Partial Fine-Tuning

**What it is:**
- Updates **only selected parameters**
- Directly modifies original weights (like full FT)
- But excludes certain components

**Parameters updated:**
- ✅ Selected layers/components
- ❌ Other layers/components stay frozen

**Memory:** Medium (gradients only for selected parameters)

**Our implementation:** ✅ **THIS IS WHAT WE DO!**

**What we update:**
- ✅ Attention weights (Q, K, V, O) - ~64 tensors
- ✅ FFN weights (gate, up, down) - ~48 tensors
- ❌ Embeddings (frozen)
- ❌ Layer norms (frozen)
- ❌ Biases (frozen)
- ❌ Other components (frozen)

**Total:** ~112 tensors out of 147 (~76% of parameters)

---

## Comparison Table

| Type | Parameters Updated | Memory | Expressiveness | Our Implementation |
|------|-------------------|--------|----------------|-------------------|
| **Full FT** | 100% (all) | Very High | Highest | ❌ No |
| **LoRA** | ~0.1-1% (adapters) | Low | Medium | ❌ No |
| **Selective FT** | ~50-80% (selected) | Medium | High | ✅ **Yes** |
| **Our ZO Methods** | ~76% (attention + FFN) | Low-Medium* | High | ✅ **Yes** |

*Lower than gradient-based because we don't store gradients, but still need optimizer state for R-AdaZO

---

## Why Selective Fine-Tuning?

### Advantages

1. **Reduced Memory**
   - Only need optimizer state for selected parameters
   - Still significant savings vs. full FT

2. **Focused Adaptation**
   - Attention and FFN are the most important for task adaptation
   - Embeddings and norms are often better kept frozen

3. **Computational Efficiency**
   - Fewer parameters to update → fewer forward passes needed
   - Critical for zeroth-order methods!

4. **Empirical Success**
   - Many papers show selective FT works almost as well as full FT
   - Attention + FFN capture most of the adaptation

### What We Exclude and Why

| Component | Why Frozen |
|-----------|------------|
| **Embeddings** | Usually task-agnostic, large memory footprint |
| **Layer Norms** | Often better kept from pretraining |
| **Biases** | Small impact, can be updated separately if needed |
| **Position Embeddings** | Usually don't need adaptation |

---

## Code Evidence

### What Gets Collected

```cpp
// From radazo-optimizer.cpp:collect_trainable_parameters_radazo()

// Attention weights (per layer)
if (name.find("attn_q.weight") != std::string::npos ||
    name.find("attn_k.weight") != std::string::npos ||
    name.find("attn_v.weight") != std::string::npos ||
    name.find("attn_output.weight") != std::string::npos ||
    // ... FFN weights ...
```

### Direct Weight Modification

```cpp
// From radazo-optimizer.cpp:update_parameter_adam()

// Read current value from ORIGINAL tensor
ggml_backend_tensor_get(param, &current_val, ...);

// Compute update
float new_val = current_val - update;

// Write back to ORIGINAL tensor (not a LoRA adapter!)
ggml_backend_tensor_set(param, &new_val, ...);
```

**Key point:** We're modifying `param` directly, which is the original model weight tensor, not a LoRA adapter.

---

## Converting to Other Types

### To Full Fine-Tuning

Modify `collect_trainable_parameters_radazo()` to include ALL tensors:

```cpp
// Instead of filtering, collect everything:
for (size_t i = 0; i < n_tensors; ++i) {
    struct ggml_tensor * tensor = llama_model_get_tensor_by_index(...);
    if (tensor != nullptr) {
        params.push_back(tensor);  // Add ALL tensors
    }
}
```

**Warning:** This will significantly increase memory and compute!

### To LoRA

Would require major refactoring:

1. Create adapter matrices A and B for each weight
2. Modify forward pass to compute `W_original + B×A`
3. Only update A and B, keep W_original frozen
4. Store adapters separately, merge on save

**Current implementation:** Not designed for this (would need significant changes)

---

## Summary

**Our zeroth-order fine-tuning is:**

✅ **Selective/Partial Fine-Tuning**
- Updates ~76% of parameters (attention + FFN)
- Directly modifies original weights
- Excludes embeddings, layer norms, biases

❌ **NOT LoRA**
- No adapter matrices
- No low-rank decomposition
- Direct weight modification

❌ **NOT Full Fine-Tuning**
- Not all parameters updated
- Selective subset only

**Why selective?**
- Balances expressiveness and efficiency
- Focuses on most important parameters
- Reduces computational cost (critical for ZO methods!)
- Works well empirically

---

## References

- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **Selective Fine-Tuning**: Common practice in efficient fine-tuning literature
- **Our Implementation**: Zeroth-order selective fine-tuning for inference-only frameworks

