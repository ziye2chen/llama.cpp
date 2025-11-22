# GSM8K Evaluation - Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GSM8K Evaluation System                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌──────────────────┐
│  gsm8k_test.jsonl│         │  GGUF Model      │
│  (1319 problems) │         │  (e.g., llama3)  │
└────────┬────────┘         └────────┬─────────┘
         │                           │
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      gsm8k-eval.cpp                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  1. Load Dataset (load_gsm8k_dataset)                      │ │
│  │     - Parse JSONL                                          │ │
│  │     - Extract questions & answers                          │ │
│  │     - Parse ground truth (#### NUMBER)                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  2. Initialize Model                                       │ │
│  │     - llama_model_load_from_file()                         │ │
│  │     - llama_init_from_model()                              │ │
│  │     - Setup sampler (greedy decoding)                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  3. Evaluation Loop (run_evaluation)                      │ │
│  │     For each problem:                                      │ │
│  │       ┌──────────────────────────────────────────────────┐ │ │
│  │       │ a) Create prompt: "Q: ... A: Let's solve..."     │ │ │
│  │       └──────────────────────────────────────────────────┘ │ │
│  │       ┌──────────────────────────────────────────────────┐ │ │
│  │       │ b) Generate text (generate_text)                 │ │ │
│  │       │    - Tokenize prompt                             │ │ │
│  │       │    - Run inference (llama_decode)                │ │ │
│  │       │    - Sample tokens (greedy)                      │ │ │
│  │       │    - Decode to text                              │ │ │
│  │       └──────────────────────────────────────────────────┘ │ │
│  │       ┌──────────────────────────────────────────────────┐ │ │
│  │       │ c) Extract answer (extract_model_answer)         │ │ │
│  │       │    - Try pattern 1: #### NUMBER                  │ │ │
│  │       │    - Try pattern 2: answer is NUMBER             │ │ │
│  │       │    - Try pattern 3: last number in output        │ │ │
│  │       └──────────────────────────────────────────────────┘ │ │
│  │       ┌──────────────────────────────────────────────────┐ │ │
│  │       │ d) Compare with ground truth                     │ │ │
│  │       │    - Increment correct/incorrect counters        │ │ │
│  │       └──────────────────────────────────────────────────┘ │ │
│  │       ┌──────────────────────────────────────────────────┐ │ │
│  │       │ e) Update progress                               │ │ │
│  │       │    - Print [N/1319] Accuracy: X% | Time | ETA   │ │ │
│  │       └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  4. Report Results                                         │ │
│  │     - Total problems                                       │ │
│  │     - Correct/Incorrect counts                             │ │
│  │     - Accuracy percentage                                  │ │
│  │     - Timing statistics                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Output Results                             │
│  - Console: Progress updates and final statistics               │
│  - Verbose: Individual problem details                           │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
Input Data:
┌─────────────────────────┐
│ {"question": "...",     │
│  "answer": "...\n#### 3"│
│ }                       │
└───────────┬─────────────┘
            │
            ▼
┌───────────────────────────┐
│ gsm8k_problem struct:     │
│  - question: string       │
│  - answer: string         │
│  - ground_truth: int      │
└───────────┬───────────────┘
            │
            ▼
┌────────────────────────────┐
│ Prompt Generation:         │
│ "Q: " + question           │
│ + "\nA: Let's solve..."    │
└───────────┬────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ Model Inference:             │
│ GGUF Model → Generated Text  │
└───────────┬──────────────────┘
            │
            ▼
┌────────────────────────────────────┐
│ Answer Extraction:                 │
│ "...therefore the answer is 18"    │
│          ↓                         │
│ extract_model_answer()             │
│          ↓                         │
│      answer = 18                   │
└───────────┬────────────────────────┘
            │
            ▼
┌────────────────────────────────┐
│ Comparison:                    │
│ model_answer == ground_truth?  │
│   18 == 18? → YES ✓            │
└───────────┬────────────────────┘
            │
            ▼
┌────────────────────────────────┐
│ Statistics:                    │
│ correct_count++                │
│ accuracy = correct/total       │
└────────────────────────────────┘
```

## Module Breakdown

### 1. Dataset Module

**File**: `gsm8k-eval.cpp` (lines 51-108)

**Functions**:
- `extract_ground_truth(const std::string & answer)` → int
- `load_gsm8k_dataset(const std::string & path, int max_problems)` → vector<gsm8k_problem>

**Responsibilities**:
- Read JSONL file
- Parse JSON using nlohmann/json
- Extract numerical answer from "#### NUMBER" format
- Handle malformed entries
- Support limiting number of problems

**Data Structures**:
```cpp
struct gsm8k_problem {
    std::string question;      // The math word problem
    std::string answer;        // Full answer with reasoning
    int ground_truth;          // The final numerical answer
};
```

### 2. Answer Extraction Module

**File**: `gsm8k-eval.cpp` (lines 70-108)

**Functions**:
- `extract_ground_truth(const std::string & answer)` → int
  - Extracts answer from dataset format
- `extract_model_answer(const std::string & output)` → int
  - Extracts answer from model output using 3 strategies

**Extraction Strategies**:
1. **Pattern Matching**: `#### NUMBER` (if model mimics format)
2. **Keyword Search**: "answer is NUMBER" or "answer: NUMBER"
3. **Last Number**: Extract final integer in output (fallback)

**Regular Expressions Used**:
```cpp
std::regex pattern1(R"(####\s*(-?\d+))");                           // #### 18
std::regex pattern2(R"(answer\s*(?:is|:)\s*(-?\d+))", ...icase);    // answer is 18
std::regex pattern3(R"((-?\d+))");                                  // any number
```

### 3. Model Interface Module

**File**: `gsm8k-eval.cpp` (lines 110-195)

**Functions**:
- `generate_text(llama_context*, llama_sampler*, prompt, n_predict)` → string

**Responsibilities**:
- Tokenize input prompt
- Clear KV cache between problems
- Handle encoder-decoder models
- Run inference loop
- Sample tokens (greedy decoding)
- Decode tokens to text
- Handle end-of-generation tokens

**Key llama.cpp APIs used**:
```cpp
llama_tokenize()          // Prompt → tokens
llama_kv_cache_clear()    // Clear KV cache
llama_batch_get_one()     // Create batch
llama_encode()            // Encoder models
llama_decode()            // Run inference
llama_sampler_sample()    // Sample next token
llama_token_to_piece()    // Token → text
llama_vocab_is_eog()      // Check if end of generation
```

### 4. Evaluation Module

**File**: `gsm8k-eval.cpp` (lines 197-287)

**Functions**:
- `run_evaluation(llama_context*, problems, config)` → void

**Responsibilities**:
- Initialize sampler (greedy decoding)
- Loop through all problems
- Generate answers
- Compare with ground truth
- Track statistics (correct, total, time)
- Display progress updates
- Print final report

**Progress Tracking**:
```cpp
[100/1319] Accuracy: 47.5% (48/100) | Time: 123.4s | ETA: 1234.5s
              ↑         ↑      ↑         ↑            ↑
           Current   Percent  Count   Elapsed    Remaining
```

### 5. Main Program Module

**File**: `gsm8k-eval.cpp` (lines 289-end)

**Functions**:
- `print_usage()` → void
- `main(argc, argv)` → int

**Responsibilities**:
- Parse command-line arguments
- Validate inputs
- Load dataset
- Initialize llama backend
- Load GGUF model
- Create context
- Run evaluation
- Cleanup resources

**Command-Line Parsing**:
```cpp
-m, --model          → config.model_path
-f, --file           → config.dataset_path
-n, --n-predict      → config.n_predict
-c, --ctx-size       → config.n_ctx
-ngl, --n-gpu-layers → config.n_gpu_layers
--max-problems       → config.max_problems
-v, --verbose        → config.verbose
```

## Configuration Structure

```cpp
struct eval_config {
    std::string model_path;     // Path to GGUF model
    std::string dataset_path;   // Path to gsm8k_test.jsonl
    int n_predict = 512;        // Max tokens to generate
    int n_ctx = 2048;           // Context window size
    int n_batch = 512;          // Batch size
    int n_gpu_layers = 99;      // GPU layers to offload
    float temperature = 0.0f;   // Greedy decoding
    int max_problems = -1;      // -1 = all problems
    bool verbose = false;       // Print each problem
};
```

## Dependencies

```
gsm8k-eval
    │
    ├─ llama (llama.cpp core)
    │   ├─ llama_model_load_from_file()
    │   ├─ llama_init_from_model()
    │   ├─ llama_decode()
    │   ├─ llama_sampler_*()
    │   └─ llama_tokenize()
    │
    ├─ ggml (backend)
    │   ├─ ggml_time_us()
    │   └─ ggml_backend_load_all()
    │
    ├─ common (utilities)
    │   └─ (included for consistency)
    │
    └─ nlohmann/json (JSON parsing)
        └─ json::parse()
```

## Build System

```
examples/CMakeLists.txt
    │
    ├─ add_subdirectory(evaluation)
    │
    └─ evaluation/CMakeLists.txt
           │
           ├─ add_executable(gsm8k-eval gsm8k-eval.cpp)
           │
           └─ target_link_libraries(gsm8k-eval
                  PRIVATE ggml llama common)
```

## Error Handling

| Error Type | Handling Strategy |
|------------|-------------------|
| Model loading fails | Exit with error message |
| Dataset loading fails | Exit with error message |
| JSON parsing fails | Skip line, log error, continue |
| Tokenization fails | Skip problem, log error, continue |
| Inference fails | Return empty string, log error |
| Answer extraction fails | Return error value (-999999) |

## Performance Characteristics

### Time Complexity
- **Per Problem**: O(n) where n = generation length
- **Total**: O(P × n) where P = number of problems (1319)

### Space Complexity
- **Dataset**: O(P) for storing all problems
- **Model**: O(M) for model weights
- **Context**: O(C) for KV cache
- **Total**: O(P + M + C)

### Typical Memory Usage
- **Model (1B params, F32)**: ~4 GB
- **Model (1B params, Q4)**: ~1 GB
- **Context (2048)**: ~500 MB
- **Dataset**: ~10 MB
- **Total**: 1-5 GB depending on quantization

### Timing
- **Model loading**: 2-10 seconds
- **Per problem**: 0.5-2 seconds
- **Full evaluation**: 10-45 minutes
- **With GPU**: 2-3× faster

## Extensibility Points

### 1. Custom Prompting
Modify prompt generation in `run_evaluation()`:
```cpp
std::string prompt = custom_prompt_template(problem.question);
```

### 2. Different Sampling
Change sampler configuration:
```cpp
llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
```

### 3. Additional Metrics
Extend statistics tracking:
```cpp
struct eval_stats {
    int correct, incorrect, total;
    float avg_confidence;
    std::vector<int> per_category_accuracy;
};
```

### 4. Result Export
Add CSV/JSON export:
```cpp
void export_results(const eval_stats& stats, const std::string& format);
```

### 5. Parallel Processing
Process multiple problems in parallel:
```cpp
#pragma omp parallel for
for (int i = 0; i < problems.size(); ++i) {
    // Process problem i
}
```

## Testing Strategy

1. **Unit Tests**: Test each module independently
2. **Integration Tests**: Test end-to-end with small dataset
3. **Regression Tests**: Compare with known baseline results
4. **Performance Tests**: Measure speed on different hardware
5. **Stress Tests**: Run with large models and full dataset

## Future Architecture Enhancements

1. **Modular Design**: Separate into multiple files
2. **Plugin System**: Support custom extractors
3. **Caching**: Cache model outputs
4. **Distributed**: Run on multiple machines
5. **Web Interface**: Add HTTP API and web UI

---

This architecture is designed to be:
- **Simple**: Easy to understand and modify
- **Efficient**: Minimal overhead, fast execution
- **Robust**: Handles errors gracefully
- **Extensible**: Easy to add new features
- **Portable**: Works on Windows, Linux, Mac






