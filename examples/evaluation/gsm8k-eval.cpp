// GSM8K Evaluation for GGUF Models
// Evaluates mathematical reasoning on Grade School Math 8K test set

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <sstream>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

using json = nlohmann::ordered_json;

// GSM8K problem structure
struct gsm8k_problem {
    std::string question;
    std::string answer;
    int ground_truth;
};

// Evaluation configuration
struct eval_config {
    std::string model_path;
    std::string dataset_path;
    int n_predict = 512;           // Max tokens to generate
    int n_ctx = 2048;              // Context size
    int n_batch = 512;             // Batch size
    int n_gpu_layers = 99;         // GPU layers
    float temperature = 0.0f;      // Use greedy decoding for consistency
    int max_problems = -1;         // -1 = evaluate all problems
    bool verbose = false;          // Print each problem
};

// Extract the final answer (integer) from the GSM8K answer string
// The answer is always after "#### " in the format
static int extract_ground_truth(const std::string & answer) {
    size_t pos = answer.find("#### ");
    if (pos != std::string::npos) {
        std::string num_str = answer.substr(pos + 5);
        // Remove any trailing whitespace or newlines
        num_str.erase(num_str.find_last_not_of(" \n\r\t") + 1);
        try {
            return std::stoi(num_str);
        } catch (...) {
            return -999999; // Error value
        }
    }
    return -999999; // Error value
}

// Extract numeric answer from model output
// Try multiple patterns to find the final answer
static int extract_model_answer(const std::string & output) {
    int answer = -999999;
    
    // Pattern 1: Look for "#### NUMBER" (if model mimics the format)
    std::regex pattern1(R"(####\s*(-?\d+))");
    std::smatch match;
    if (std::regex_search(output, match, pattern1)) {
        try {
            answer = std::stoi(match[1].str());
            return answer;
        } catch (...) {}
    }
    
    // Pattern 2: Look for "answer is NUMBER" or "answer: NUMBER"
    std::regex pattern2(R"(answer\s*(?:is|:)\s*(-?\d+))", std::regex_constants::icase);
    if (std::regex_search(output, match, pattern2)) {
        try {
            answer = std::stoi(match[1].str());
            return answer;
        } catch (...) {}
    }
    
    // Pattern 3: Look for final number at the end (last integer in output)
    std::regex pattern3(R"((-?\d+))");
    auto words_begin = std::sregex_iterator(output.begin(), output.end(), pattern3);
    auto words_end = std::sregex_iterator();
    
    // Get the last match
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        try {
            answer = std::stoi((*i)[1].str());
        } catch (...) {}
    }
    
    return answer;
}

// Load GSM8K dataset from JSONL file
static std::vector<gsm8k_problem> load_gsm8k_dataset(const std::string & path, int max_problems) {
    std::vector<gsm8k_problem> problems;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        LOG_ERR("%s: failed to open dataset file: %s\n", __func__, path.c_str());
        return problems;
    }
    
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (max_problems > 0 && count >= max_problems) {
            break;
        }
        
        try {
            json j = json::parse(line);
            gsm8k_problem problem;
            problem.question = j["question"].get<std::string>();
            problem.answer = j["answer"].get<std::string>();
            problem.ground_truth = extract_ground_truth(problem.answer);
            
            if (problem.ground_truth != -999999) {
                problems.push_back(problem);
                count++;
            }
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to parse line: %s\n", __func__, e.what());
            continue;
        }
    }
    
    file.close();
    return problems;
}

// Generate text from model given a prompt
static std::string generate_text(
        llama_context * ctx,
        llama_sampler * sampler,
        const std::string & prompt,
        int n_predict) {
    
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    
    // Tokenize prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        LOG_ERR("%s: failed to tokenize prompt\n", __func__);
        return "";
    }
    
    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);
    
    // Prepare batch with prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    
    // Handle encoder-decoder models
    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            LOG_ERR("%s: failed to encode\n", __func__);
            return "";
        }
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }
    
    // Generate tokens
    std::string generated_text;
    int n_pos = 0;
    llama_token new_token_id;
    
    for (int i = 0; i < n_predict && n_pos + batch.n_tokens < llama_n_ctx(ctx); ) {
        // Evaluate
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: failed to decode\n", __func__);
            return generated_text;
        }
        
        n_pos += batch.n_tokens;
        
        // Sample next token
        new_token_id = llama_sampler_sample(sampler, ctx, -1);
        
        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }
        
        // Convert token to text
        char buf[128];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            generated_text.append(buf, n);
        }
        
        // Prepare next batch
        batch = llama_batch_get_one(&new_token_id, 1);
        i++;
    }
    
    return generated_text;
}

// Run evaluation on GSM8K dataset
static void run_evaluation(
        llama_context * ctx,
        const std::vector<gsm8k_problem> & problems,
        const eval_config & config) {
    
    LOG_INF("\n%s: Starting GSM8K evaluation...\n", __func__);
    LOG_INF("%s: Total problems: %zu\n\n", __func__, problems.size());
    
    // Initialize sampler
    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    
    // Use greedy decoding for consistency
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    
    int correct = 0;
    int total = 0;
    const int64_t t_start = ggml_time_us();
    
    for (size_t i = 0; i < problems.size(); ++i) {
        const auto & problem = problems[i];
        
        // Create prompt - simple instruction format
        std::string prompt = "Q: " + problem.question + "\nA: Let's solve this step by step.\n";
        
        // Generate answer
        std::string model_output = generate_text(ctx, sampler, prompt, config.n_predict);
        
        // Extract model's answer
        int model_answer = extract_model_answer(model_output);
        
        // Check if correct
        bool is_correct = (model_answer == problem.ground_truth);
        if (is_correct) {
            correct++;
        }
        total++;
        
        // Print progress
        if (config.verbose || (i + 1) % 10 == 0) {
            const int64_t t_now = ggml_time_us();
            const float elapsed = (t_now - t_start) / 1.0e6f;
            const float avg_time = elapsed / (i + 1);
            const float eta = avg_time * (problems.size() - i - 1);
            
            fprintf(stderr, "\r[%zu/%zu] Accuracy: %.2f%% (%d/%d) | Time: %.1fs | ETA: %.1fs",
                    i + 1, problems.size(),
                    100.0f * correct / total, correct, total,
                    elapsed, eta);
            fflush(stderr);
        }
        
        // Verbose output
        if (config.verbose) {
            fprintf(stderr, "\n");
            LOG_INF("Problem %zu:\n", i + 1);
            LOG_INF("  Question: %s\n", problem.question.c_str());
            LOG_INF("  Ground Truth: %d\n", problem.ground_truth);
            LOG_INF("  Model Answer: %d\n", model_answer);
            LOG_INF("  Model Output: %s\n", model_output.c_str());
            LOG_INF("  Result: %s\n\n", is_correct ? "CORRECT ✓" : "INCORRECT ✗");
        }
    }
    
    fprintf(stderr, "\n\n");
    
    const int64_t t_end = ggml_time_us();
    const float total_time = (t_end - t_start) / 1.0e6f;
    
    // Final results
    LOG_INF("=====================================\n");
    LOG_INF("GSM8K Evaluation Results\n");
    LOG_INF("=====================================\n");
    LOG_INF("Model: %s\n", config.model_path.c_str());
    LOG_INF("Dataset: %s\n", config.dataset_path.c_str());
    LOG_INF("Total problems: %d\n", total);
    LOG_INF("Correct: %d\n", correct);
    LOG_INF("Incorrect: %d\n", total - correct);
    LOG_INF("Accuracy: %.2f%%\n", 100.0f * correct / total);
    LOG_INF("Total time: %.2f seconds\n", total_time);
    LOG_INF("Average time per problem: %.2f seconds\n", total_time / total);
    LOG_INF("=====================================\n");
    
    llama_sampler_free(sampler);
}

static void print_usage(const char * argv0) {
    printf("\nGSM8K Evaluation Tool\n");
    printf("Evaluates GGUF models on Grade School Math 8K test set\n\n");
    printf("Usage: %s [options]\n", argv0);
    printf("\nOptions:\n");
    printf("  -m, --model PATH         Path to GGUF model file (required)\n");
    printf("  -f, --file PATH          Path to gsm8k_test.jsonl (default: ./gsm8k_test.jsonl)\n");
    printf("  -n, --n-predict N        Number of tokens to predict (default: 512)\n");
    printf("  -c, --ctx-size N         Context size (default: 2048)\n");
    printf("  -ngl, --n-gpu-layers N   Number of GPU layers (default: 99)\n");
    printf("  --max-problems N         Max problems to evaluate (default: all)\n");
    printf("  -v, --verbose            Print each problem and answer\n");
    printf("  -h, --help               Show this help message\n");
    printf("\nExample:\n");
    printf("  %s -m llama3_2_1b_f32.gguf -f gsm8k_test.jsonl --max-problems 100\n", argv0);
    printf("\n");
}

int main(int argc, char ** argv) {
    eval_config config;
    config.dataset_path = "gsm8k_test.jsonl";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                config.model_path = argv[++i];
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) {
                config.dataset_path = argv[++i];
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-n" || arg == "--n-predict") {
            if (i + 1 < argc) {
                config.n_predict = std::stoi(argv[++i]);
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-c" || arg == "--ctx-size") {
            if (i + 1 < argc) {
                config.n_ctx = std::stoi(argv[++i]);
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (i + 1 < argc) {
                config.n_gpu_layers = std::stoi(argv[++i]);
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--max-problems") {
            if (i + 1 < argc) {
                config.max_problems = std::stoi(argv[++i]);
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else {
            LOG_ERR("Error: unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (config.model_path.empty()) {
        LOG_ERR("Error: model path is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    // Load dataset
    LOG_INF("%s: Loading GSM8K dataset from %s\n", __func__, config.dataset_path.c_str());
    std::vector<gsm8k_problem> problems = load_gsm8k_dataset(config.dataset_path, config.max_problems);
    
    if (problems.empty()) {
        LOG_ERR("%s: failed to load dataset or dataset is empty\n", __func__);
        return 1;
    }
    
    LOG_INF("%s: Loaded %zu problems\n", __func__, problems.size());
    
    // Initialize llama backend
    llama_backend_init();
    ggml_backend_load_all();
    
    // Load model
    LOG_INF("%s: Loading model from %s\n", __func__, config.model_path.c_str());
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config.n_gpu_layers;
    
    llama_model * model = llama_model_load_from_file(config.model_path.c_str(), model_params);
    
    if (model == NULL) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }
    
    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    ctx_params.n_batch = config.n_batch;
    ctx_params.no_perf = false;
    
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    
    if (ctx == NULL) {
        LOG_ERR("%s: failed to create context\n", __func__);
        llama_model_free(model);
        return 1;
    }
    
    LOG_INF("%s: Model loaded successfully\n", __func__);
    
    // Run evaluation
    run_evaluation(ctx, problems, config);
    
    // Cleanup
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    
    return 0;
}

