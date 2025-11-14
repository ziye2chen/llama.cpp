# GSM8K Evaluation - Quick Start Guide

This guide will help you quickly set up and run GSM8K evaluation on your GGUF models.

## Prerequisites

- A built llama.cpp project
- A GGUF model file
- The `gsm8k_test.jsonl` dataset (already included in this directory)

## Step 1: Build the Project

### Windows (Visual Studio)

```cmd
cd D:\Github\llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

The executable will be at: `build\bin\Release\gsm8k-eval.exe`

### Windows (MinGW)

```cmd
cd D:\Github\llama.cpp
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
cmake --build . --config Release
```

### Linux/Mac

```bash
cd ~/llama.cpp
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
```

The executable will be at: `build/bin/gsm8k-eval`

## Step 2: Run Evaluation

### Quick Test (First 10 Problems)

Test your setup with just 10 problems to make sure everything works:

**Windows:**
```cmd
cd examples\evaluation
..\..\build\bin\Release\gsm8k-eval.exe -m D:\Github\llama.cpp\llama3_2_1b_f32.gguf -f gsm8k_test.jsonl --max-problems 10 -v
```

**Linux/Mac:**
```bash
cd examples/evaluation
../../build/bin/gsm8k-eval -m ../../llama3_2_1b_f32.gguf -f gsm8k_test.jsonl --max-problems 10 -v
```

### Full Evaluation (All 1,319 Problems)

**Windows:**
```cmd
cd examples\evaluation
..\..\build\bin\Release\gsm8k-eval.exe -m D:\Github\llama.cpp\llama3_2_1b_f32.gguf -f gsm8k_test.jsonl
```

Or using the batch script:
```cmd
run_gsm8k_eval.bat D:\Github\llama.cpp\llama3_2_1b_f32.gguf
```

**Linux/Mac:**
```bash
cd examples/evaluation
../../build/bin/gsm8k-eval -m ../../llama3_2_1b_f32.gguf -f gsm8k_test.jsonl
```

Or using the shell script:
```bash
./run_gsm8k_eval.sh ../../llama3_2_1b_f32.gguf
```

### Evaluate Subset (e.g., 100 Problems)

**Windows:**
```cmd
run_gsm8k_eval.bat D:\Github\llama.cpp\llama3_2_1b_f32.gguf 100
```

**Linux/Mac:**
```bash
./run_gsm8k_eval.sh ../../llama3_2_1b_f32.gguf 100
```

## Step 3: Interpret Results

The output will show:

```
[1319/1319] Accuracy: 45.26% (597/1319) | Time: 1234.5s | ETA: 0.0s

=====================================
GSM8K Evaluation Results
=====================================
Model: llama3_2_1b_f32.gguf
Dataset: gsm8k_test.jsonl
Total problems: 1319
Correct: 597
Incorrect: 722
Accuracy: 45.26%
Total time: 1234.56 seconds
Average time per problem: 0.94 seconds
=====================================
```

### Expected Accuracy Ranges

- **Small models (1-3B)**: 5-30% - Basic arithmetic, struggles with multi-step reasoning
- **Medium models (7-13B)**: 15-35% - Can handle simple problems, inconsistent on complex ones
- **Large models (30-70B)**: 40-60% - Good performance, handles most problems
- **Very large models (>100B)**: 70-95% - Excellent performance, near human-level

Note: These are rough estimates. Actual performance depends heavily on the model's training data and whether it was specifically trained on math problems.

## Common Issues

### Issue: "failed to load model"

**Solution**: Check the model path is correct
```cmd
dir D:\Github\llama.cpp\*.gguf
```

### Issue: "failed to load dataset"

**Solution**: Make sure you're in the right directory
```cmd
cd examples\evaluation
dir gsm8k_test.jsonl
```

### Issue: Very slow evaluation

**Solutions**:
1. Use GPU acceleration: `-ngl 33` (adjust based on your GPU)
2. Reduce context size: `-c 1024`
3. Test on subset first: `--max-problems 10`

### Issue: Out of memory

**Solutions**:
1. Reduce GPU layers: `-ngl 0` (CPU only)
2. Reduce context size: `-c 512`
3. Use a smaller quantized model

### Issue: Very low accuracy (< 5%)

**Possible causes**:
1. Model is not instruction-tuned (base models perform poorly)
2. Model is too small (< 1B parameters)
3. Model was not trained on math/reasoning tasks

**Check model output**:
```cmd
gsm8k-eval.exe -m model.gguf -f gsm8k_test.jsonl --max-problems 5 -v
```

Look at the generated outputs. If the model is producing:
- Random text → Model might be corrupted
- Refusing to answer → Model needs different prompting
- Nonsensical math → Model is too small or not math-capable

## Advanced Usage

### Custom Prompting

If your model uses a specific chat template, you may want to modify the prompt format in `gsm8k-eval.cpp` at line ~323:

```cpp
// Current prompt:
std::string prompt = "Q: " + problem.question + "\nA: Let's solve this step by step.\n";

// For chat models, you might want:
std::string prompt = "[INST] " + problem.question + " [/INST]\n";
```

### Adjusting Generation Parameters

In `gsm8k-eval.cpp`, you can modify:
- `n_predict` (line 29): Max tokens to generate (default: 512)
- `temperature` (line 33): Generation temperature (default: 0.0 for greedy)
- Context window size
- Sampling parameters

### Batch Processing Multiple Models

**Windows batch script** (`eval_all_models.bat`):
```cmd
@echo off
for %%M in (models\*.gguf) do (
    echo Evaluating %%M
    gsm8k-eval.exe -m "%%M" -f gsm8k_test.jsonl > results_%%~nM.txt
)
```

**Linux/Mac shell script** (`eval_all_models.sh`):
```bash
#!/bin/bash
for model in models/*.gguf; do
    echo "Evaluating $model"
    ./gsm8k-eval -m "$model" -f gsm8k_test.jsonl > "results_$(basename $model .gguf).txt"
done
```

## Performance Tips

1. **GPU Acceleration**: Use `-ngl 33` or higher for significant speedup
2. **Batch Size**: Adjust `-c` based on your available memory
3. **Subset Testing**: Always test with `--max-problems 10` first
4. **Verbose Mode**: Use `-v` to debug issues with small samples

## Next Steps

After running evaluation:

1. **Compare Models**: Run on multiple models to see which performs best
2. **Analyze Errors**: Use verbose mode to see where models fail
3. **Fine-tune**: Use the results to guide fine-tuning efforts (see `examples/zeroth-order-opt/`)
4. **Share Results**: Report your findings to help the community

## Questions?

See the full [README.md](README.md) for more details, or check the llama.cpp documentation.

