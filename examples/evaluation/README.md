# GSM8K Evaluation Tool

This tool evaluates GGUF models on the **Grade School Math 8K (GSM8K)** test set, a dataset of grade school level math word problems that require multi-step reasoning.

## Overview

GSM8K is a dataset of 8.5K high quality grade school math problems created by human problem writers. Each problem requires 2-8 steps to solve and tests mathematical reasoning abilities.

The test set contains 1,319 problems. Each problem has:
- A natural language question
- A ground truth answer with step-by-step solution
- A final numerical answer (integer) after `#### `

## Building

The evaluation tool is built as part of the llama.cpp project:

```bash
# From the llama.cpp root directory
mkdir build
cd build
cmake ..
cmake --build . --config Release

# The executable will be in build/bin/
```

On Windows:
```cmd
mkdir build
cd build
cmake ..
cmake --build . --config Release

REM The executable will be in build\bin\Release\
```

## Usage

### Basic Usage

```bash
./gsm8k-eval -m path/to/model.gguf -f path/to/gsm8k_test.jsonl
```

### Windows

```cmd
gsm8k-eval.exe -m D:\Github\llama.cpp\llama3_2_1b_f32.gguf -f gsm8k_test.jsonl
```

### All Options

```
Options:
  -m, --model PATH         Path to GGUF model file (required)
  -f, --file PATH          Path to gsm8k_test.jsonl (default: ./gsm8k_test.jsonl)
  -n, --n-predict N        Number of tokens to predict (default: 512)
  -c, --ctx-size N         Context size (default: 2048)
  -ngl, --n-gpu-layers N   Number of GPU layers (default: 99)
  --max-problems N         Max problems to evaluate (default: all)
  -v, --verbose            Print each problem and answer
  -h, --help               Show help message
```

### Examples

Evaluate on first 100 problems only:
```bash
./gsm8k-eval -m model.gguf -f gsm8k_test.jsonl --max-problems 100
```

Verbose mode (print each question and answer):
```bash
./gsm8k-eval -m model.gguf -f gsm8k_test.jsonl -v --max-problems 10
```

With specific GPU configuration:
```bash
./gsm8k-eval -m model.gguf -f gsm8k_test.jsonl -ngl 33 -c 4096
```

## Dataset Format

The `gsm8k_test.jsonl` file contains one JSON object per line:

```json
{
  "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
  "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
}
```

The final answer is always an integer after `#### ` in the answer field.

## How It Works

1. **Load Dataset**: Reads the JSONL file and parses each problem
2. **Extract Ground Truth**: Extracts the integer answer after `#### ` 
3. **Generate Prompts**: For each question, creates a simple prompt:
   ```
   Q: <question>
   A: Let's solve this step by step.
   ```
4. **Model Inference**: Generates model response (up to `n_predict` tokens)
5. **Extract Model Answer**: Uses multiple strategies to extract the numerical answer:
   - Look for `#### NUMBER` pattern
   - Look for "answer is NUMBER" or "answer: NUMBER"
   - Extract last number in the output
6. **Compare**: Checks if model's answer matches ground truth
7. **Report**: Displays accuracy and timing statistics

## Output

The tool provides:
- Real-time progress with accuracy updates
- Total accuracy percentage
- Number of correct/incorrect answers
- Total evaluation time
- Average time per problem

Example output:
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

## Tips for Better Results

1. **Use appropriate models**: Larger models generally perform better on math reasoning
2. **Adjust n_predict**: Math problems may need longer responses (512-1024 tokens)
3. **Temperature=0**: The default greedy decoding ensures consistent results
4. **Test first**: Use `--max-problems 10` to quickly test your setup
5. **Verbose mode**: Use `-v` on small samples to see how the model responds

## Benchmarks

GSM8K accuracy for reference (from literature):
- GPT-3 175B: ~20%
- GPT-3.5 (text-davinci-003): ~57%
- GPT-4: ~92%
- Llama-2 7B: ~14%
- Llama-2 13B: ~29%
- Llama-2 70B: ~57%
- Llama-3 8B: ~79%
- Llama-3 70B: ~93%

Small models (1-3B) typically achieve 10-30% accuracy without specialized training.

## Troubleshooting

**Problem**: "failed to load dataset"
- Solution: Check that the `gsm8k_test.jsonl` file path is correct

**Problem**: Very low accuracy (< 5%)
- Solution: The model might not be following the prompt format. Try `-v` to see outputs

**Problem**: "failed to load model"
- Solution: Ensure the GGUF model path is correct and the model is compatible

**Problem**: Out of memory
- Solution: Reduce `-c` (context size) or `-ngl` (use fewer GPU layers)

## Citation

If you use GSM8K in your research, please cite:

```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

## License

This evaluation tool follows the llama.cpp license. The GSM8K dataset is licensed under MIT License.

