# Zeroth-Order Fine-Tuning - Quick Start Guide

A 5-minute guide to get started with gradient-free fine-tuning.

## What is This?

**Zeroth-order optimization** = Fine-tuning neural networks **without backpropagation**, using only forward passes.

```
Standard: θ ← θ - α·∇L(θ)          [requires backprop]
Zeroth:   ∇L ≈ [L(θ+ε) - L(θ)]/ε  [only forward passes]
```

**Why?**
- ✅ Lower memory (no gradient storage)
- ✅ Works with inference-only frameworks
- ❌ Slower (more forward passes needed)

---

## Installation

```bash
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CURL=OFF
cmake --build . --config Debug --target finetune-zeroth-order
```

---

## Quick Test (30 seconds)

```bash
# Create tiny test file
echo "This is a test sentence for fine-tuning the model." > test.txt

# Run fine-tuning
.\bin\Debug\finetune-zeroth-order.exe \
    -m path/to/model.gguf \
    -f test.txt \
    -o output.gguf \
    -b 8 \
    -c 64 \
    --epochs 1
```

**Expected**: Completes in ~30 seconds

---

## Real Fine-Tuning

```bash
.\bin\Debug\finetune-zeroth-order.exe \
    -m llama3_2_1b_f32.gguf \
    -f wiki.test.raw \
    -o finetuned_model.gguf \
    -b 32 \
    -c 128 \
    --epochs 1
```

**Parameters**:
- `-b 32`: Batch size (larger = slower but more stable)
- `-c 128`: Context size (must fit in RAM)
- `--epochs 1`: Number of passes through data

**Time**: ~5 minutes for 6500 tokens

---

## Adjusting Dataset Size

Edit `examples/zeroth-order-opt/finetune-zeroth-order.cpp` line 221:

```cpp
zo_params.max_train_tokens = 6500;  // Options:
                                    //    10 = 10 seconds
                                    //   100 = 1 minute  
                                    //  6500 = 5 minutes
                                    //    -1 = full dataset
```

Then rebuild:
```bash
cmake --build . --config Debug --target finetune-zeroth-order
```

---

## Tuning Hyperparameters

Edit `examples/zeroth-order-opt/finetune-zeroth-order.cpp` lines 216-220:

```cpp
zo_params.epsilon = 1e-2f;              // Perturbation size
zo_params.learning_rate = 1e-1f;        // Update step size
zo_params.weight_decay = 0.1f;          // Regularization
zo_params.n_params_per_iter = 10;       // Parameters per batch
zo_params.n_elements_per_param = 5;     // Elements per parameter
```

**Guidelines**:

| Parameter | Too Small | Good Range | Too Large |
|-----------|-----------|------------|-----------|
| `epsilon` | Numerical errors | `1e-4` to `1e-2` | Poor approximation |
| `learning_rate` | No learning | `1e-4` to `1e-1` | Divergence |
| `n_params_per_iter` | Slow updates | `5` to `50` | Too slow |

---

## Verifying It Worked

### Check File Size
```bash
# Should be ~same size
ls -lh original_model.gguf finetuned_model.gguf
```

### Test Inference
```bash
.\bin\Debug\llama-cli.exe \
    -m finetuned_model.gguf \
    -p "The quick brown fox"
```

### Compare Perplexity
```bash
# Original
.\bin\Debug\llama-perplexity.exe -m original.gguf -f test.txt

# Fine-tuned
.\bin\Debug\llama-perplexity.exe -m finetuned.gguf -f test.txt
```

**If perplexity changed**: ✅ Parameters were updated!

---

## Understanding the Output

```
collect_model_params: model has 147 tensors total
collect_model_params: collected 112 trainable parameter tensors
  ↑ Found all weight matrices (attn_q, attn_k, attn_v, ffn_*, etc.)

[TRAIN] Iter 10/91 | Loss: 69.699 | Time: 31.23s | 0.3 it/s
        ↑     ↑        ↑              ↑              ↑
     mode  progress   loss value   elapsed     speed

finetune_zeroth_order: Epoch 1 complete - Avg Loss: 71.618
                                                ↑ Should decrease over time
```

**Good signs**:
- Loss trends downward
- No crashes or NaN values
- Model file created successfully

**Bad signs**:
- Loss increases or oscillates wildly → reduce learning rate
- Very slow (< 0.1 it/s) → reduce batch size or context
- Out of memory → reduce `-c` context size

---

## Common Issues

### 1. "Out of memory"
```bash
# Reduce context size
-c 64   # instead of 128
```

### 2. "Loss is NaN"
```cpp
// Reduce learning rate in code
zo_params.learning_rate = 1e-4f;  // was 1e-1f
```

### 3. "Too slow"
```cpp
// Reduce sampling in code
zo_params.n_params_per_iter = 5;     // was 10
zo_params.n_elements_per_param = 3;  // was 5
```

### 4. "Parameters not updating"
- Check that you're using the correct output file
- Verify perplexity actually changed
- Make sure `max_train_tokens > 0`

---

## What's Actually Happening?

```python
# Pseudocode of the algorithm
for each training batch:
    loss_base = forward_pass(model, batch)
    
    for param in random_sample(parameters, k=10):
        for element in random_sample(param, m=5):
            # Estimate gradient
            param[element] += epsilon
            loss_perturbed = forward_pass(model, batch)
            gradient = (loss_perturbed - loss_base) / epsilon
            
            # Update
            param[element] -= learning_rate * gradient
```

**Key insight**: We're approximating gradients using the difference in loss after tiny parameter changes.

---

## Next Steps

### For Learning:
1. Read [`THEORY.md`](THEORY.md) for mathematical details
2. Run [`zeroth-order-test.exe`](zeroth-order-test.cpp) to see the algorithm on simple functions
3. Experiment with hyperparameters

### For Production:
1. Implement proper cross-entropy loss (currently placeholder)
2. Add validation set evaluation
3. Use central differences for better accuracy
4. Add momentum/Adam optimizer
5. Parallelize parameter updates

### For Research:
1. Compare convergence with standard fine-tuning
2. Test on different model sizes
3. Explore importance sampling strategies
4. Combine with LoRA or other PEFT methods

---

## Resources

- **Full Documentation**: [`README.md`](README.md)
- **Theory**: [`THEORY.md`](THEORY.md)
- **Code**: [`finetune-zeroth-order.cpp`](finetune-zeroth-order.cpp)
- **Unit Tests**: [`zeroth-order-test.cpp`](zeroth-order-test.cpp)

---

## One-Liners

```bash
# Test on simple functions
.\bin\Debug\zeroth-order-test.exe

# Quick fine-tune (30 sec)
echo "Test data" > test.txt && .\bin\Debug\finetune-zeroth-order.exe -m model.gguf -f test.txt -o out.gguf -b 8 -c 64 --epochs 1

# Compare models
.\bin\Debug\llama-perplexity.exe -m model1.gguf -f data.txt && .\bin\Debug\llama-perplexity.exe -m model2.gguf -f data.txt

# Generate with fine-tuned model
.\bin\Debug\llama-cli.exe -m finetuned.gguf -p "Your prompt here" -n 50
```

---

**That's it! You're now fine-tuning without backpropagation. 🎉**

Questions? Check the full [`README.md`](README.md) or open an issue.

