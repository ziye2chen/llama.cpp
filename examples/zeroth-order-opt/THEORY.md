# Zeroth-Order Optimization Theory

A mathematical deep-dive into gradient-free optimization for neural network training.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Gradient Estimation Methods](#gradient-estimation-methods)
4. [Convergence Analysis](#convergence-analysis)
5. [Comparison with Standard Methods](#comparison-with-standard-methods)
6. [Practical Considerations](#practical-considerations)

---

## Introduction

### What is Zeroth-Order Optimization?

**Zeroth-order optimization** refers to optimization methods that only require function evaluations (zeroth-order information), not derivatives. This contrasts with:

- **First-order methods**: Use gradient (∇f)
- **Second-order methods**: Use Hessian (∇²f)

### Why "Zeroth-Order"?

The term comes from the **Taylor series expansion**:

```
f(x + δ) = f(x) + ∇f(x)ᵀδ + (1/2)δᵀ∇²f(x)δ + ...
           ↑        ↑              ↑
        zeroth    first         second
```

Zeroth-order methods only use `f(x)` values.

---

## Mathematical Foundation

### Problem Setup

We want to minimize a loss function:

```
min L(θ)
 θ
```

Where:
- `θ ∈ ℝᵈ` are the model parameters (d dimensions)
- `L: ℝᵈ → ℝ` is the loss function
- We can only evaluate `L(θ)` (no access to ∇L)

### Standard Gradient Descent

First-order gradient descent:

```
θₜ₊₁ = θₜ - α∇L(θₜ)
```

Where:
- `α` is the learning rate
- `∇L(θₜ)` is computed via backpropagation

**Problem**: Computing ∇L requires:
1. Storing intermediate activations
2. Backward pass through network
3. O(d) memory for gradients

---

## Gradient Estimation Methods

### 1. Forward Finite Difference

**Idea**: Approximate the gradient using function evaluations.

For a single coordinate i:

```
∂L/∂θᵢ ≈ [L(θ + εeᵢ) - L(θ)] / ε
```

Where:
- `εeᵢ` is a small perturbation in direction i
- `eᵢ` is the i-th unit vector

**Algorithm**:
```python
def forward_difference(θ, ε):
    gradient = zeros(d)
    L_base = forward_pass(θ)
    
    for i in range(d):
        θ_perturbed = θ.copy()
        θ_perturbed[i] += ε
        L_perturbed = forward_pass(θ_perturbed)
        gradient[i] = (L_perturbed - L_base) / ε
    
    return gradient
```

**Cost**: O(d) forward passes per iteration

**Error**: O(ε) + O(ε²) (first-order accurate)

### 2. Central Finite Difference

**More accurate** version using symmetric perturbations:

```
∂L/∂θᵢ ≈ [L(θ + εeᵢ) - L(θ - εeᵢ)] / (2ε)
```

**Algorithm**:
```python
def central_difference(θ, ε):
    gradient = zeros(d)
    
    for i in range(d):
        θ_plus = θ.copy()
        θ_plus[i] += ε
        L_plus = forward_pass(θ_plus)
        
        θ_minus = θ.copy()
        θ_minus[i] -= ε
        L_minus = forward_pass(θ_minus)
        
        gradient[i] = (L_plus - L_minus) / (2 * ε)
    
    return gradient
```

**Cost**: O(2d) forward passes per iteration

**Error**: O(ε²) (second-order accurate)

### 3. Random Direction Estimation

**Efficiency trick**: Estimate gradient in a random direction.

```
∇L(θ) ≈ (d/ε)[L(θ + εu) - L(θ)]u
```

Where `u ~ Uniform(S^(d-1))` is a random unit vector.

**Algorithm**:
```python
def random_direction(θ, ε):
    u = random_unit_vector(d)
    L_base = forward_pass(θ)
    L_perturbed = forward_pass(θ + ε * u)
    return (d / ε) * (L_perturbed - L_base) * u
```

**Cost**: O(2) forward passes per iteration

**Bias**: E[estimate] ≈ ∇L(θ) (unbiased in expectation)

**Used in**: Evolution Strategies (ES)

### 4. Our Implementation (Sparse Coordinate Sampling)

**Practical approach** for large neural networks:

```
For each iteration:
  1. Sample k parameters randomly (k << d)
  2. For each sampled parameter i:
       a. Sample m elements within that parameter
       b. Perturb each element by ε
       c. Estimate gradient via forward difference
       d. Update: θᵢ ← θᵢ - α·ĝᵢ
```

**Parameters in our demo**:
- k = 10 (n_params_per_iter)
- m = 5 (n_elements_per_param)
- ε = 1e-2 (epsilon)
- α = 1e-1 (learning_rate)

**Cost**: O(k·m·2) ≈ 100 forward passes per iteration

---

## Convergence Analysis

### Convergence Rate

For smooth, L-Lipschitz functions:

**Standard gradient descent**:
```
L(θₜ) - L(θ*) ≤ O(1/t)
```

**Zeroth-order (random direction)**:
```
𝔼[L(θₜ) - L(θ*)] ≤ O(d/t)
```

**Key observation**: d-factor slowdown due to gradient estimation variance.

### Convergence Conditions

For convergence, we need:

1. **Learning rate**: `Σ αₜ = ∞` and `Σ αₜ² < ∞`
   - Example: `αₜ = 1/√t`

2. **Epsilon**: Must satisfy `ε → 0` as `t → ∞`
   - But not too fast: `ε ≫ machine_precision`

3. **Smoothness**: L must be L-Lipschitz continuous

### Optimal Epsilon

The approximation error is:

```
|∇̂L - ∇L| ≈ Cε + D/ε²
           ↑     ↑
        bias  variance
```

**Optimal choice**: `ε ≈ O(σ/√d)` where σ is noise level

Typical range: `1e-4 to 1e-2` for neural networks

---

## Comparison with Standard Methods

### Memory Complexity

| Method | Memory for Gradients | Memory for Activations |
|--------|---------------------|------------------------|
| **Backprop** | O(d) | O(L·B) per layer |
| **Zeroth-Order** | O(1) | O(L·B) (for forward only) |

Where:
- d = number of parameters
- L = number of layers
- B = batch size

### Time Complexity (per iteration)

| Method | Forward Passes | Backward Passes | Total Operations |
|--------|----------------|-----------------|------------------|
| **SGD** | 1 | 1 | O(d) |
| **ZO-SGD (coordinate)** | d | 0 | O(d²) |
| **ZO-SGD (random)** | 2 | 0 | O(d) |
| **Our method** | k·m | 0 | O(k·m·d_param) |

### Convergence Speed

For reaching ε-accuracy:

```
Standard SGD:        O(1/ε²) iterations
Zeroth-Order (full): O(d/ε²) iterations
Zeroth-Order (rand): O(d/ε²) iterations (in expectation)
```

### Quality of Solution

**Empirical observations**:
- ZO can match first-order on some tasks
- Performance gap increases with:
  - Model size (larger d)
  - Task complexity
  - Training data size

---

## Practical Considerations

### 1. Choosing Epsilon

**Too small**: Numerical instability
```
ε < √machine_precision → catastrophic cancellation
```

**Too large**: Poor approximation
```
|∇̂L - ∇L| ≈ O(ε)
```

**Rule of thumb**:
```python
ε = max(1e-7, 1e-3 * ||θ||)
```

### 2. Adaptive Learning Rate

Instead of fixed α, use adaptive methods:

**Adam-style update**:
```python
m = β₁ * m + (1-β₁) * gradient    # momentum
v = β₂ * v + (1-β₂) * gradient²   # variance
θ -= α * m / (√v + ε_adam)
```

### 3. Variance Reduction

**Antithetic variates**:
```python
# Instead of:
g = (f(θ+ε) - f(θ)) / ε

# Use:
g = (f(θ+ε) - f(θ-ε)) / (2ε)  # central difference
```

**Multiple samples**:
```python
gradient = mean([estimate_gradient(θ, seed=i) for i in range(K)])
```

### 4. Coordinate Selection

**Random sampling** (our approach):
```python
indices = random.sample(range(d), k)
```

**Importance sampling**:
```python
# Prioritize parameters with large magnitude
probs = |θ|² / Σ|θ|²
indices = sample(range(d), k, p=probs)
```

**Block-wise**:
```python
# Update entire layers at once
for layer in model.layers:
    update_layer(layer)
```

### 5. Regularization

**Weight decay** (L2 regularization):
```python
θ ← θ - α·∇L(θ) - λ·θ
```

Prevents overfitting on small datasets.

**Typical values**: λ ∈ [1e-5, 1e-1]

### 6. Batch Size Effects

**Small batches** (1-32):
- High variance in loss
- Noisy gradient estimates
- Better generalization?

**Large batches** (64-512):
- More stable loss
- Better gradient estimates
- May overfit

**Our choice**: 32-64 (balance)

---

## Advanced Topics

### Evolution Strategies (ES)

**Natural Evolution Strategies**:

```
θₜ₊₁ = θₜ + α·∇θ 𝔼[L(θ + σε)]
```

Where ε ~ N(0, I).

**Advantages**:
- Highly parallelizable
- No need for memory
- Works with discrete/black-box functions

**OpenAI ES** (2017):
- Trained RL agents with only forward passes
- Competitive with A3C on Atari

### Simultaneous Perturbation (SPSA)

**Key idea**: Perturb all coordinates at once with random signs.

```
gradient = [L(θ + ε·Δ) - L(θ - ε·Δ)] / (2ε) · (1/Δ)
```

Where Δᵢ ~ Bernoulli({-1, +1})

**Cost**: Only 2 function evaluations!

**Used in**: Control systems, simulation optimization

### Bayesian Optimization

**Model the function** with a Gaussian Process:

```
L(θ) ~ GP(μ(θ), k(θ, θ'))
```

**Acquisition function** guides sampling:
```
θₙₑₓₜ = argmax EI(θ) or UCB(θ)
```

**Best for**: Low-dimensional problems (d < 100)

---

## Summary Table

| Property | Backprop | ZO (Coordinate) | ZO (Random) | Our Method |
|----------|----------|-----------------|-------------|------------|
| **Forward passes/iter** | 1 | d+1 | 2 | k·m |
| **Backward passes/iter** | 1 | 0 | 0 | 0 |
| **Gradient memory** | O(d) | O(1) | O(1) | O(1) |
| **Convergence rate** | O(1/t) | O(1/t) | O(d/t) | O(1/t) |
| **Parallelizable** | Limited | Yes | Yes | Yes |
| **Implementation** | Complex | Simple | Simple | Simple |

---

## Mathematical Proofs

### Theorem 1: Unbiasedness of Random Direction

**Claim**: E[ĝ] = ∇L(θ) where ĝ = (d/ε)[L(θ+εu) - L(θ)]u

**Proof**:
```
E[ĝ] = E[(d/ε)[L(θ+εu) - L(θ)]u]
     = (d/ε)E[[L(θ+εu) - L(θ)]u]           (linearity)
     = (d/ε)E[L(θ+εu)·u]                   (E[L(θ)u] = 0)
     
By Taylor expansion:
L(θ+εu) ≈ L(θ) + ε∇L(θ)ᵀu

So:
E[ĝ] = (d/ε)E[(L(θ) + ε∇L(θ)ᵀu)·u]
     = (d/ε)E[L(θ)·u] + d·E[∇L(θ)ᵀu·u]
     = 0 + d·∇L(θ)ᵀE[u·u]
     = d·∇L(θ)ᵀ·(1/d)I                    (E[uuᵀ] = I/d for uniform u)
     = ∇L(θ)
```

### Theorem 2: Convergence Rate

Under standard assumptions (L-smoothness, convexity):

```
𝔼[L(θₜ) - L(θ*)] ≤ (L||θ₀-θ*||² + σ²d·Σαₜ²) / (2·Σαₜ)
```

For αₜ = 1/√t:
```
𝔼[L(θₜ) - L(θ*)] = O(d·log(t)/√t)
```

---

## References

### Foundational Papers

1. **Kiefer & Wolfowitz (1952)**: "Stochastic Estimation of the Maximum of a Regression Function"
   - First ZO algorithm (FDSA)

2. **Spall (1992)**: "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation"
   - SPSA algorithm

3. **Nesterov & Spokoiny (2017)**: "Random Gradient-Free Minimization of Convex Functions"
   - Theoretical foundations

### Recent ML Applications

4. **Liu et al. (2018)**: "ZO-SGD: Zeroth-Order Stochastic Gradient Descent"
   - First application to deep learning

5. **Salimans et al. (2017)**: "Evolution Strategies as a Scalable Alternative to Reinforcement Learning"
   - OpenAI ES for RL

6. **Malladi et al. (2023)**: "Fine-Tuning Language Models with Just Forward Passes (MeZO)"
   - ZO for LLM fine-tuning

### Books

7. **Conn, Scheinberg, Vicente (2009)**: "Introduction to Derivative-Free Optimization"
   - Comprehensive textbook

---

## Exercises

### Exercise 1: Implement Central Difference

Modify the code to use central differences instead of forward differences. Measure the accuracy improvement.

### Exercise 2: Optimal Epsilon

For the quadratic function f(x) = (x-3)², plot the approximation error vs. ε. Find the optimal ε experimentally.

### Exercise 3: Variance Analysis

Compare the variance of gradient estimates between:
- Coordinate sampling (k=10)
- Random directions (same cost)

### Exercise 4: Momentum

Add momentum to the zeroth-order optimizer:
```python
v = 0.9 * v + gradient
θ -= α * v
```

Does it help convergence?

---

**End of Theory Document**

