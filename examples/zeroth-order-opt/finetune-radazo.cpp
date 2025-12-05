// R-AdaZO Fine-tuning Demo for GGUF Models with GSM8K Dataset
// Refining Adaptive Zeroth-Order Optimization
// Paper: "Refining Adaptive Zeroth-Order Optimization at Ease" (arXiv:2502.01014)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "radazo-optimizer.h"
#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>

// Platform-specific headers for CPU and memory monitoring
#if defined(_WIN32)
    #include <windows.h>
    #include <psapi.h>
    #pragma comment(lib, "psapi.lib")
#elif defined(__linux__)
    #include <unistd.h>
    #include <sys/resource.h>
    #include <fstream>
#elif defined(__APPLE__)
    #include <unistd.h>
    #include <sys/resource.h>
    #include <mach/mach.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

using json = nlohmann::json;

// ========================================
// System Resource Monitoring
// ========================================

// Get current process memory usage in MB
static float get_memory_usage_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0f * 1024.0f);  // Convert bytes to MB
    }
    return 0.0f;
#elif defined(__linux__)
    // Read from /proc/self/status
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            float mem_kb;
            iss >> mem_kb;
            return mem_kb / 1024.0f;  // Convert KB to MB
        }
    }
    return 0.0f;
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t info_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &info_count) == KERN_SUCCESS) {
        return info.resident_size / (1024.0f * 1024.0f);  // Convert bytes to MB
    }
    return 0.0f;
#else
    return 0.0f;
#endif
}

// Get CPU usage percentage (approximate, based on time slice)
struct cpu_usage_tracker {
    int64_t last_time_us;
    int64_t last_cpu_time_us;
    
    cpu_usage_tracker() : last_time_us(0), last_cpu_time_us(0) {}
    
    float get_cpu_usage() {
#if defined(_WIN32)
        FILETIME create_time, exit_time, kernel_time, user_time;
        if (GetProcessTimes(GetCurrentProcess(), &create_time, &exit_time, &kernel_time, &user_time)) {
            int64_t now_us = ggml_time_us();
            
            // Convert FILETIME to microseconds (100-nanosecond intervals)
            int64_t cpu_time_us = ((int64_t)user_time.dwHighDateTime << 32 | user_time.dwLowDateTime) / 10;
            cpu_time_us += ((int64_t)kernel_time.dwHighDateTime << 32 | kernel_time.dwLowDateTime) / 10;
            
            if (last_time_us > 0) {
                int64_t elapsed_us = now_us - last_time_us;
                int64_t cpu_elapsed_us = cpu_time_us - last_cpu_time_us;
                
                if (elapsed_us > 0) {
                    float cpu_percent = (100.0f * cpu_elapsed_us) / elapsed_us;
                    last_time_us = now_us;
                    last_cpu_time_us = cpu_time_us;
                    return cpu_percent;
                }
            }
            
            last_time_us = now_us;
            last_cpu_time_us = cpu_time_us;
        }
        return 0.0f;
#elif defined(__linux__) || defined(__APPLE__)
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            int64_t now_us = ggml_time_us();
            int64_t cpu_time_us = usage.ru_utime.tv_sec * 1000000LL + usage.ru_utime.tv_usec;
            cpu_time_us += usage.ru_stime.tv_sec * 1000000LL + usage.ru_stime.tv_usec;
            
            if (last_time_us > 0) {
                int64_t elapsed_us = now_us - last_time_us;
                int64_t cpu_elapsed_us = cpu_time_us - last_cpu_time_us;
                
                if (elapsed_us > 0) {
                    float cpu_percent = (100.0f * cpu_elapsed_us) / elapsed_us;
                    last_time_us = now_us;
                    last_cpu_time_us = cpu_time_us;
                    return cpu_percent;
                }
            }
            
            last_time_us = now_us;
            last_cpu_time_us = cpu_time_us;
        }
        return 0.0f;
#else
        return 0.0f;
#endif
    }
};

// Training configuration
struct training_config {
    int32_t max_train_samples = 10;   // Maximum number of GSM8K samples to use
    int32_t max_tokens_per_sample = 512; // Maximum tokens per training sample
};

// GSM8K data structure
struct gsm8k_example {
    std::string question;
    std::string answer;
};

// Load GSM8K dataset from JSONL file
static std::vector<gsm8k_example> load_gsm8k(const std::string & filepath, int max_samples = -1) {
    std::vector<gsm8k_example> examples;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        LOG_ERR("%s: failed to open file: %s\n", __func__, filepath.c_str());
        return examples;
    }
    
    std::string line;
    int count = 0;
    
    while (std::getline(file, line) && (max_samples <= 0 || count < max_samples)) {
        try {
            json j = json::parse(line);
            gsm8k_example ex;
            ex.question = j["question"].get<std::string>();
            ex.answer = j["answer"].get<std::string>();
            examples.push_back(ex);
            count++;
        } catch (const std::exception & e) {
            LOG_ERR("%s: error parsing JSON line %d: %s\n", __func__, count + 1, e.what());
            continue;
        }
    }
    
    file.close();
    LOG_INF("%s: loaded %zu GSM8K examples\n", __func__, examples.size());
    return examples;
}

// Format GSM8K example for training
static std::string format_gsm8k_prompt(const gsm8k_example & ex) {
    // Format: Question: <question>\nAnswer: <answer>
    std::stringstream ss;
    ss << "Question: " << ex.question << "\n";
    ss << "Answer: " << ex.answer;
    return ss.str();
}

// Convert GSM8K examples to token sequences
static std::vector<llama_token> gsm8k_to_tokens(
        struct llama_context * ctx,
        const std::vector<gsm8k_example> & examples,
        int max_tokens_per_sample = -1) {
    
    std::vector<llama_token> all_tokens;
    
    LOG_INF("%s: tokenizing %zu GSM8K examples...\n", __func__, examples.size());
    
    for (size_t i = 0; i < examples.size(); ++i) {
        std::string prompt = format_gsm8k_prompt(examples[i]);
        std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true);
        
        // Optionally limit tokens per sample
        if (max_tokens_per_sample > 0 && tokens.size() > (size_t)max_tokens_per_sample) {
            tokens.resize(max_tokens_per_sample);
        }
        
        all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
        
        if ((i + 1) % 10 == 0) {
            LOG_INF("%s: tokenized %zu/%zu examples (%zu tokens total)\n", 
                    __func__, i + 1, examples.size(), all_tokens.size());
        }
    }
    
    LOG_INF("%s: total tokens from %zu examples: %zu\n", 
            __func__, examples.size(), all_tokens.size());
    
    return all_tokens;
}

// Progress callback (with CPU and memory monitoring)
static void progress_callback_radazo(
        bool train,
        int64_t iter,
        int64_t iter_max,
        float loss,
        int64_t t_start_us,
        cpu_usage_tracker & cpu_tracker) {
    const int64_t t_now_us = ggml_time_us();
    const float elapsed = (t_now_us - t_start_us) / 1.0e6f;
    
    // Get system resource usage
    float memory_mb = get_memory_usage_mb();
    float cpu_percent = cpu_tracker.get_cpu_usage();
    
    fprintf(stderr, "\r[%s] Iter %6lld/%6lld | Loss: %.6f | Time: %6.2fs | %.1f it/s | CPU: %5.1f%% | Mem: %7.1f MB",
            train ? "TRAIN" : "EVAL ",
            iter, iter_max, loss, elapsed,
            iter > 0 ? iter / elapsed : 0.0f,
            cpu_percent,
            memory_mb);
    fflush(stderr);
}

// R-AdaZO fine-tuning loop
static void finetune_radazo(
        struct llama_context * ctx,
        const std::vector<llama_token> & train_tokens,
        const std::vector<llama_token> & eval_tokens,
        RAdaZOOptimizer & optimizer,
        int n_epochs) {
    
    LOG_INF("\n%s: starting R-AdaZO fine-tuning...\n", __func__);
    
    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);
    const struct llama_model * model_ptr = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ptr));
    
    LOG_INF("%s: n_ctx=%d, n_batch=%d, n_train_tokens=%zu, n_epochs=%d\n",
            __func__, n_ctx, n_batch, train_tokens.size(), n_epochs);
    
    // Initialize CPU usage tracker for monitoring
    cpu_usage_tracker cpu_tracker;
    
    // Training loop
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        LOG_INF("\n%s: ===== Epoch %d/%d =====\n", __func__, epoch + 1, n_epochs);
        
        const int64_t t_epoch_start = ggml_time_us();
        int64_t n_batches = 0;
        float epoch_loss = 0.0f;
        
        // Process training data in batches
        for (size_t i = 0; i + n_batch < train_tokens.size(); i += n_batch) {
            // Clear KV cache at start of each batch
            llama_memory_clear(llama_get_memory(ctx), true);
            
            // Prepare batch
            llama_batch batch = llama_batch_init(n_batch, 0, 1);
            for (int j = 0; j < n_batch && i + j < train_tokens.size(); ++j) {
                batch.token[j] = train_tokens[i + j];
                batch.pos[j] = j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (j == n_batch - 1);
            }
            batch.n_tokens = n_batch;
            
            // Forward pass to get baseline loss
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: failed to decode batch\n", __func__);
                llama_batch_free(batch);
                continue;
            }
            
            // *** USE R-AdaZO OPTIMIZER ***
            float batch_loss = optimizer.step(ctx, batch, n_vocab);
            
            epoch_loss += batch_loss;
            n_batches++;
            
            // Progress reporting (print every iteration)
            progress_callback_radazo(true, n_batches, 
                train_tokens.size() / n_batch, epoch_loss / n_batches, t_epoch_start, cpu_tracker);
            
            llama_batch_free(batch);
        }
        
        fprintf(stderr, "\n");
        LOG_INF("%s: Epoch %d complete - Avg Loss: %.6f\n", 
                __func__, epoch + 1, epoch_loss / n_batches);
    }
    
    // Print optimizer statistics
    LOG_INF("\n%s: R-AdaZO Optimizer Statistics:\n", __func__);
    LOG_INF("%s:   Total parameter updates: %lld\n", __func__, optimizer.get_total_updates());
    LOG_INF("%s:   Total forward passes: %lld\n", __func__, optimizer.get_forward_passes());
    LOG_INF("%s:   Average forward passes per update: %.2f\n", __func__, 
            (float)optimizer.get_forward_passes() / optimizer.get_total_updates());
    
    // Print final resource usage
    LOG_INF("\n%s: Final Resource Usage:\n", __func__);
    LOG_INF("%s:   Memory: %.1f MB\n", __func__, get_memory_usage_mb());
    LOG_INF("%s:   CPU: %.1f%%\n", __func__, cpu_tracker.get_cpu_usage());
    
    LOG_INF("\n%s: fine-tuning complete!\n", __func__);
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;
    
    // Set defaults suitable for R-AdaZO
    params.n_batch = 4;
    params.n_ctx = 512;
    
    // Default GSM8K dataset path (try multiple locations)
    std::string gsm8k_file;
    std::vector<std::string> possible_paths = {
        "examples/zeroth-order-opt/gsm8k_test.jsonl",           // From repo root
        "../examples/zeroth-order-opt/gsm8k_test.jsonl",        // From build directory
        "../../examples/zeroth-order-opt/gsm8k_test.jsonl",     // From build/bin
        "../../../examples/zeroth-order-opt/gsm8k_test.jsonl"   // From build/bin/Debug
    };
    
    // Check which path exists
    for (const auto & path : possible_paths) {
        std::ifstream test_file(path);
        if (test_file.good()) {
            gsm8k_file = path;
            break;
        }
    }
    
    // If no default path found, use the first one anyway (will error later with helpful message)
    if (gsm8k_file.empty()) {
        gsm8k_file = possible_paths[0];
    }
    
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }
    
    // Force settings for optimization
    if (params.use_mmap) {
        LOG_INF("%s: disabling memory mapping for weight updates\n", __func__);
        params.use_mmap = false;
    }
    
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    
    // Load model and context
    common_init_result llama_init = common_init_from_params(params);
    llama_model_ptr & model = llama_init.model;
    llama_context_ptr & ctx = llama_init.context;
    
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }
    
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n%s: ===== R-AdaZO FINE-TUNING DEMO (GSM8K Dataset) =====\n", __func__);
    LOG_INF("%s: Model: %s\n", __func__, params.model.path.c_str());
    
    // Training configuration
    training_config train_config;
    train_config.max_train_samples = 10;     // Start with 10 samples for testing
    train_config.max_tokens_per_sample = 512; // Max tokens per example
    
    // Load GSM8K dataset (file path was parsed earlier from --gsm8k argument)
    LOG_INF("%s: Loading GSM8K dataset from: %s\n", __func__, gsm8k_file.c_str());
    auto gsm8k_examples = load_gsm8k(gsm8k_file, train_config.max_train_samples);
    
    if (gsm8k_examples.empty()) {
        LOG_ERR("%s: failed to load GSM8K examples\n", __func__);
        return 1;
    }
    
    LOG_INF("%s: Loaded %zu GSM8K examples\n", __func__, gsm8k_examples.size());
    LOG_INF("%s: Example format - Question: <question>\\nAnswer: <answer>\n\n", __func__);
    
    // Show first example
    if (!gsm8k_examples.empty()) {
        LOG_INF("%s: First training example:\n", __func__);
        LOG_INF("%s: ----------------------------------------\n", __func__);
        LOG_INF("%s: %s\n", __func__, format_gsm8k_prompt(gsm8k_examples[0]).c_str());
        LOG_INF("%s: ----------------------------------------\n\n", __func__);
    }
    
    // Tokenize GSM8K examples
    std::vector<llama_token> train_tokens = gsm8k_to_tokens(
        ctx.get(), gsm8k_examples, train_config.max_tokens_per_sample);
    
    if (train_tokens.size() < 10) {
        LOG_ERR("%s: training data too short (need at least 10 tokens)\n", __func__);
        return 1;
    }
    
    LOG_INF("%s: Total training tokens: %zu\n", __func__, train_tokens.size());
    
    // Split into train/eval (90/10)
    size_t split_idx = train_tokens.size() * 0.9;
    std::vector<llama_token> train_set(train_tokens.begin(), train_tokens.begin() + split_idx);
    std::vector<llama_token> eval_set(train_tokens.begin() + split_idx, train_tokens.end());
    
    LOG_INF("%s: train_tokens=%zu, eval_tokens=%zu\n\n", __func__, train_set.size(), eval_set.size());
    
    // *** SETUP R-AdaZO OPTIMIZER ***
    
    // Step 1: Collect trainable parameters
    auto trainable_params = collect_trainable_parameters_radazo(ctx.get(), true);
    LOG_INF("%s: found %zu trainable parameter tensors\n\n", __func__, trainable_params.size());
    
    // Step 2: Configure R-AdaZO parameters
    radazo_params radazo_config;
    radazo_config.lr = 1e-3f;                    // Learning rate (R-AdaZO can use higher than basic ZO)
    radazo_config.beta1 = 0.9f;                  // First moment decay (momentum)
    radazo_config.beta2 = 0.999f;                // Second moment decay
    radazo_config.eps = 1e-8f;                   // Numerical stability
    radazo_config.mu = 5e-3f;                    // Perturbation magnitude
    radazo_config.n_samples = 4;                 // Multiple random samples (key to R-AdaZO!)
    radazo_config.n_params_per_iter = 10;        // Parameters per batch
    radazo_config.full_tensor_gradient = true;   // Update entire tensor each step
    radazo_config.log_gradients = true;         // Set to true for debugging
    
    LOG_INF("%s: R-AdaZO Configuration:\n", __func__);
    LOG_INF("%s:   lr = %.2e (learning rate)\n", __func__, radazo_config.lr);
    LOG_INF("%s:   beta1 = %.3f (first moment decay)\n", __func__, radazo_config.beta1);
    LOG_INF("%s:   beta2 = %.3f (second moment decay)\n", __func__, radazo_config.beta2);
    LOG_INF("%s:   mu = %.2e (perturbation magnitude)\n", __func__, radazo_config.mu);
    LOG_INF("%s:   n_samples = %d (multiple perturbations per gradient!)\n", __func__, radazo_config.n_samples);
    LOG_INF("%s:   n_params_per_iter = %d\n", __func__, radazo_config.n_params_per_iter);
    LOG_INF("%s:   full_tensor_gradient = %s (entire tensor updates)\n", __func__,
            radazo_config.full_tensor_gradient ? "true" : "false");
    LOG_INF("%s:   Forward passes per parameter = %d (n_samples)\n\n", __func__, radazo_config.n_samples);
    
    LOG_INF("%s: Key R-AdaZO Features:\n", __func__);
    LOG_INF("%s:   - Adam-style adaptive learning rate\n", __func__);
    LOG_INF("%s:   - Multiple random samples per gradient (reduces variance)\n", __func__);
    LOG_INF("%s:   - Uses momentum for second moment estimation (key innovation!)\n", __func__);
    LOG_INF("%s:   - Better convergence than basic zeroth-order methods\n\n", __func__);
    
    // Step 3: Create the R-AdaZO optimizer
    RAdaZOOptimizer optimizer(radazo_config, trainable_params);
    
    // Step 4: Run fine-tuning
    int n_epochs = 1;
    finetune_radazo(ctx.get(), train_set, eval_set, optimizer, n_epochs);
    
    // Save the fine-tuned model
    std::string output_file = params.out_file.empty() ? 
        "llama3_2_1b_f32_radazo_finetuned.gguf" : params.out_file;
    
    LOG_INF("\n%s: saving fine-tuned model to %s\n", __func__, output_file.c_str());
    
    llama_model_save_to_file(model.get(), output_file.c_str());
    
    LOG_INF("%s: model saved successfully!\n", __func__);
    LOG_INF("\n%s: ===== SUMMARY =====\n", __func__);
    LOG_INF("%s: Method: R-AdaZO (Refining Adaptive Zeroth-Order)\n", __func__);
    LOG_INF("%s: Dataset: GSM8K (Grade School Math)\n", __func__);
    LOG_INF("%s: Training samples: %zu\n", __func__, gsm8k_examples.size());
    LOG_INF("%s: Epochs: %d\n", __func__, n_epochs);
    LOG_INF("%s: Training tokens: %zu\n", __func__, train_set.size());
    LOG_INF("%s: Output model: %s\n", __func__, output_file.c_str());
    LOG_INF("\n%s: Fine-tuning completed on math reasoning tasks\n", __func__);
    LOG_INF("%s: Model should now be better at:\n", __func__);
    LOG_INF("%s:   - Mathematical problem solving\n", __func__);
    LOG_INF("%s:   - Step-by-step reasoning\n", __func__);
    LOG_INF("%s:   - Arithmetic calculations\n", __func__);
    LOG_INF("\n%s: R-AdaZO Advantages:\n", __func__);
    LOG_INF("%s:   - Adaptive learning rates (like Adam)\n", __func__);
    LOG_INF("%s:   - Lower variance gradient estimates\n", __func__);
    LOG_INF("%s:   - Faster convergence than basic ZO\n", __func__);
    LOG_INF("%s:   - Better handling of noisy gradients\n", __func__);
    
    llama_backend_free();
    
    return 0;
}

