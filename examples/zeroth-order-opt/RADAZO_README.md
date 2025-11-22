# R-AdaZO: Refining Adaptive Zeroth-Order Optimization

Implementation of R-AdaZO (Refining Adaptive Zeroth-Order Optimization) for GGUF models in `llama.cpp`.

**Paper**: "Refining Adaptive Zeroth-Order Optimization at Ease" (arXiv:2502.01014)

## Overview

R-AdaZO is an advanced zeroth-order optimizer that combines:
1. **Adam-style adaptive learning rates** with exponential moving averages
2. **Multiple random perturbations** per gradient estimation (variance reduction)
3. **Key innovation**: Uses momentum-smoothed gradients for second moment estimation

This reduces gradient variance significantly compared to basic zeroth-order methods while maintaining the benefits of gradient-free optimization.

## Key Differences from Basic Zeroth-Order

| Feature | Basic ZO | R-AdaZO |
|---------|----------|---------|
| **Learning Rate** | Fixed | Adaptive (per-parameter) |
| **Gradient Estimation** | Single perturbation | Multiple samples (default: 2) |
| **Variance** | High | Lower (averaged samples) |
| **Second Moment** | Uses raw gradient: `v = β₂v + (1-β₂)g²` | Uses momentum: `v = β₂v + (1-β₂)m²` |
| **Convergence** | Slower, noisy | Faster, more stable |
| **Memory** | Minimal | Stores first/second moments per parameter |

### The Key Innovation

Standard ZO-Adam:
```
m = β₁ * m + (1 - β₁) * g        # First moment (momentum)
v = β₂ * v + (1 - β₂) * g²       # Second moment (uses raw gradient)
θ = θ - α * m / (√v + ε)         # Parameter update
```

**R-AdaZO** (one line change!):
```
m = β₁ * m + (1 - β₁) * g        # First moment (momentum)
v = β₂ * v + (1 - β₂) * m²       # Second moment (uses MOMENTUM!)
θ = θ - α * m / (√v + ε)         # Parameter update
```

By using the smoothed momentum `m` instead of raw gradient `g` for the second moment, R-AdaZO dramatically reduces variance in the adaptive learning rate.

## Algorithm

For each parameter θ:

1. **Sample multiple random directions** (n_samples times):
   ```
   For k = 1 to n_samples:
       u_k = random unit vector
       f_+ = loss(θ + μ * u_k)    # Forward pass with perturbation
       f   = loss(θ)               # Baseline forward pass
       g_k = (f_+ - f) * u_k / μ  # Gradient estimate
   ```

2. **Average gradient estimates**:
   ```
   g = (1/n_samples) * Σ g_k
   ```

3. **Update momentum (first moment)**:
   ```
   m = β₁ * m + (1 - β₁) * g
   ```

4. **Update second moment (R-AdaZO innovation)**:
   ```
   v = β₂ * v + (1 - β₂) * m²    # Uses m, not g!
   ```

5. **Apply adaptive update**:
   ```
   θ = θ - α * m / (√v + ε)
   ```

## Usage

### Basic Example

```cpp
#include "radazo-optimizer.h"

// 1. Collect trainable parameters
auto trainable_params = collect_trainable_parameters_radazo(ctx, true);

// 2. Configure R-AdaZO
radazo_params config;
config.lr = 1e-3f;           // Learning rate (higher than basic ZO)
config.beta1 = 0.9f;         // First moment decay
config.beta2 = 0.999f;       // Second moment decay
config.mu = 5e-3f;           // Perturbation magnitude
config.n_samples = 2;        // Multiple samples per gradient

// 3. Create optimizer
RAdaZOOptimizer optimizer(config, trainable_params);

// 4. Training loop
for (each batch) {
    llama_decode(ctx, batch);  // Baseline forward pass
    float loss = optimizer.step(ctx, batch, n_vocab);  // R-AdaZO update
}

// 5. Get statistics
printf("Updates: %lld, Forward passes: %lld\n",
       optimizer.get_total_updates(),
       optimizer.get_forward_passes());
```

### Running the Demo

```bash
# Build
cd llama.cpp/build
cmake .. -DLLAMA_CURL=OFF
cmake --build . --config Debug --target finetune-radazo

# Run with small dataset
.\bin\Debug\finetune-radazo.exe \
    -m path/to/model.gguf \
    -f path/to/training_data.txt \
    -o output_radazo.gguf \
    -b 32 \
    -c 128 \
    --epochs 1
```

## Configuration Parameters

```cpp
struct radazo_params {
    float lr;                        // Learning rate (default: 1e-3)
    float beta1;                     // First moment decay (default: 0.9)
    float beta2;                     // Second moment decay (default: 0.999)
    float eps;                       // Numerical stability (default: 1e-8)
    float mu;                        // Perturbation magnitude (default: 5e-3)
    int32_t n_samples;               // Samples per gradient (default: 2)
    int32_t n_params_per_iter;       // Parameters per batch (default: 3)
    bool full_tensor_gradient;       // Perturb/update entire tensor (default: true)
    bool log_gradients;              // Verbose logging (default: false)
};
```

### Tuning Guidelines

#### Learning Rate (`lr`)
- R-AdaZO can use **higher learning rates** than basic ZO (10-100x)
- Thanks to adaptive per-parameter scaling
- **Recommended**: Start with `1e-3` to `1e-2`
- Monitor loss - if unstable, reduce by 10x

#### Momentum Parameters (`beta1`, `beta2`)
- `beta1 = 0.9`: Controls momentum smoothing
  - Higher = more smoothing, less responsive
  - Lower = less smoothing, more responsive
- `beta2 = 0.999`: Controls adaptive learning rate smoothing
  - Should be close to 1.0 for stable adaptive rates
- **Recommended**: Use defaults (0.9, 0.999) initially

#### Perturbation Magnitude (`mu`)
- Controls how far to perturb parameters
- **Too small** (< 1e-4): Numerical instability
- **Too large** (> 1e-1): Poor gradient approximation
- **Recommended**: `5e-3` to `1e-2`

#### Number of Samples (`n_samples`)
- **Key R-AdaZO parameter!**
- More samples = lower variance, slower
- **Trade-off**: Each sample requires 1 forward pass
- **Recommended**: Start with 2-3 samples
- Increase if gradients are very noisy (up to 5-10)

#### Forward Pass Budget

Total forward passes per batch:
```
FP = 1 (baseline) + n_params_per_iter × n_samples
```

Example with defaults (3 params, 2 samples):
```
FP = 1 + 3 × 2 = 7 forward passes per batch
```

## Performance Characteristics

### Computational Cost

- **Basic ZO**: ~7x slower than inference (1 baseline + 6 perturbed forward passes)
- **R-AdaZO (n_samples=2)**: ~7x slower (1 baseline + 6 perturbed forward passes, but each pass perturbs an entire tensor)
- **Trade-off**: Same number of passes, but better gradient quality → fewer total epochs needed

### Memory Usage

- **Gradient storage**: None (gradient-free!)
- **Optimizer state**: 2 float arrays per parameter (m and v)
  - First moment (m): ~same size as parameters
  - Second moment (v): ~same size as parameters
- **Total**: ~2x parameter size (still much less than backprop with activation storage)

### Convergence Speed

Empirical results (from paper):
- **vs. Basic ZO**: 2-5x faster convergence
- **vs. First-order (Adam)**: 5-10x slower convergence
- **Key**: Adaptive learning rates help navigate loss landscape more efficiently

## Implementation Details

### Gradient Estimation

For each tensor, we now generate `n_samples` random **full-length** perturbations:

```cpp
const int64_t n_elements = ggml_nelements(param);
std::vector<float> base(n_elements);
ggml_backend_tensor_get(param, base.data(), 0, n_elements * sizeof(float));

std::vector<float> grad(n_elements, 0.0f);
std::vector<float> perturbed(n_elements);

for (int s = 0; s < n_samples; ++s) {
    auto direction = get_perturbation(n_elements, rng_()); // unit vector
    
    for (int64_t i = 0; i < n_elements; ++i) {
        perturbed[i] = base[i] + mu * direction[i];
    }
    ggml_backend_tensor_set(param, perturbed.data(), 0, n_elements * sizeof(float));
    llama_synchronize(ctx);
    
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_decode(ctx, batch);
    
    float loss_plus = compute_loss();
    float scale = (loss_plus - loss_base) / mu;
    for (int64_t i = 0; i < n_elements; ++i) {
        grad[i] += scale * direction[i];
    }
    
    ggml_backend_tensor_set(param, base.data(), 0, n_elements * sizeof(float));
    llama_synchronize(ctx);
}

for (float & g : grad) {
    g /= n_samples;
}
```

### Adam-Style Update

With R-AdaZO's key modification:

```cpp
// Update first moment (momentum)
m = beta1 * m + (1 - beta1) * grad;

// Update second moment (uses momentum!)
v = beta2 * v + (1 - beta2) * m * m;  // Key: m² not grad²

// Bias correction
m_hat = m / (1 - beta1^t);
v_hat = v / (1 - beta2^t);

// Parameter update
param = param - lr * m_hat / (sqrt(v_hat) + eps);
```

## API Reference

### `RAdaZOOptimizer` Class

#### Constructor
```cpp
RAdaZOOptimizer(
    const radazo_params & params,
    const std::vector<struct ggml_tensor *> & trainable_params
)
```

#### Methods

**`float step(llama_context * ctx, llama_batch & batch, int n_vocab)`**
- Performs one R-AdaZO optimization step
- Returns baseline loss
- Handles all gradient estimation and parameter updates internally

**`int64_t get_total_updates() const`**
- Returns total number of parameter element updates

**`int64_t get_forward_passes() const`**
- Returns total number of forward passes executed

### Helper Functions

**`collect_trainable_parameters_radazo(llama_context * ctx, bool verbose = true)`**
- Collects attention and FFN weight matrices
- Returns `std::vector<struct ggml_tensor *>`

## Advantages and Limitations

### ✅ Advantages

1. **Better convergence** than basic zeroth-order methods
2. **Adaptive learning rates** per parameter (like Adam)
3. **Variance reduction** via multiple samples
4. **No backpropagation** required (works in inference-only frameworks)
5. **Lower memory** than gradient-based methods (no activation storage)
6. **Stable updates** even with noisy gradients

### ⚠️ Limitations

1. **Slower than backprop** (requires many forward passes)
2. **Higher memory than basic ZO** (stores momentum states)
3. **Still sample inefficient** compared to gradient descent
4. **Hyperparameter sensitive** (lr, mu, n_samples need tuning)
5. **Not suitable for very large models** (billions of parameters)

## When to Use R-AdaZO

### ✅ Good for:
- Frameworks without backpropagation (like `llama.cpp`)
- When gradient computation is not available
- Memory-constrained scenarios (vs. backprop, not vs. basic ZO)
- Research on gradient-free optimization
- Fine-tuning small parts of large models

### ❌ Not recommended for:
- When backpropagation is available and fast
- Very large-scale training (millions of examples)
- Time-critical training scenarios
- When memory is extremely limited (use basic ZO instead)

## Comparison with Other Methods

| Method | Forward Passes/Update | Memory | Convergence | Use Case |
|--------|----------------------|--------|-------------|----------|
| **SGD** | 1 | Low | Fast | Standard training |
| **Adam** | 1 | Medium | Fast | Standard training |
| **Basic ZO** | 7 | Minimal | Slow | Inference-only frameworks |
| **R-AdaZO** | 7 | Low-Medium | Medium | Better ZO with adaptive rates |

## Example Output

```
R-AdaZO Configuration:
  lr = 1.00e-03 (learning rate)
  beta1 = 0.900 (first moment decay)
  beta2 = 0.999 (second moment decay)
  mu = 5.00e-03 (perturbation magnitude)
  n_samples = 2 (multiple perturbations per gradient!)

===== Epoch 1/1 =====
[TRAIN] Iter      3/      3 | Loss: 326.546600 | Time:   8.45s | 0.4 it/s

R-AdaZO Optimizer Statistics:
  Total parameter updates: 18
  Total forward passes: 39
  Average forward passes per update: 2.17
```

## References

1. **Paper**: "Refining Adaptive Zeroth-Order Optimization at Ease"  
   Authors: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai  
   arXiv:2502.01014, 2025

2. **MeZO**: Memory-efficient zeroth-order optimizer (baseline)
3. **ZO-Adam**: Zeroth-order Adam (R-AdaZO's predecessor)

## Files

```
examples/zeroth-order-opt/
├── radazo-optimizer.h           # R-AdaZO optimizer header
├── radazo-optimizer.cpp         # R-AdaZO optimizer implementation
├── finetune-radazo.cpp          # Demo program
└── RADAZO_README.md             # This file
```

## Building

Add to `CMakeLists.txt`:

```cmake
set(TARGET finetune-radazo)
add_executable(${TARGET}
    finetune-radazo.cpp
    radazo-optimizer.cpp
    radazo-optimizer.h
)
target_link_libraries(${TARGET} PRIVATE ggml llama common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
```

## License

Part of `llama.cpp` project, follows the same license.

