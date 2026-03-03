// R-AdaZO Fine-tuning for Quantized GGUF Models (Q4_K_M, Q8_0, etc.)
// Direct quantized tensor modification: dequantize -> perturb -> requantize -> forward
// Same approach as FP32 fine-tuning (finetune-radazo.cpp) but the optimizer
// transparently handles quantized types via dequant/requant wrappers.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "radazo-optimizer.h"
#include "lora-adapter.h"
#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>

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
static constexpr const char * k_memory_log_path = "memory.txt";

// ========================================
// System Resource Monitoring
// ========================================

// Get current process memory usage in MB
static float get_memory_usage_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize / (1024.0f * 1024.0f);
    }
    return 0.0f;
#elif defined(__linux__)
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            float mem_kb;
            iss >> mem_kb;
            return mem_kb / 1024.0f;
        }
    }
    return 0.0f;
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t info_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &info_count) == KERN_SUCCESS) {
        return info.resident_size / (1024.0f * 1024.0f);
    }
    return 0.0f;
#else
    return 0.0f;
#endif
}

static bool get_gpu_memory_usage_mb(float & gpu_used_mb, float & gpu_free_mb, float & gpu_total_mb) {
    gpu_used_mb = 0.0f;
    gpu_free_mb = 0.0f;
    gpu_total_mb = 0.0f;

    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
        if (total_bytes == 0) {
            continue;
        }

        gpu_free_mb += (float) free_bytes / (1024.0f * 1024.0f);
        gpu_total_mb += (float) total_bytes / (1024.0f * 1024.0f);
    }

    if (gpu_total_mb > 0.0f) {
        gpu_used_mb = gpu_total_mb - gpu_free_mb;
        return true;
    }

    return false;
}

static void append_memory_log_train(
        const char * stage,
        int epoch,
        int64_t iter,
        int32_t batch_tokens,
        const char * extra = nullptr) {
    const float cpu_mb = get_memory_usage_mb();
    float gpu_used_mb = 0.0f, gpu_free_mb = 0.0f, gpu_total_mb = 0.0f;
    const bool has_gpu = get_gpu_memory_usage_mb(gpu_used_mb, gpu_free_mb, gpu_total_mb);

    std::ofstream fout(k_memory_log_path, std::ios::app);
    if (!fout.is_open()) {
        return;
    }

    fout << std::fixed << std::setprecision(2)
         << "scope=train"
         << " epoch=" << epoch
         << " iter=" << iter
         << " stage=" << stage
         << " batch_tokens=" << batch_tokens
         << " cpu_mb=" << cpu_mb;

    if (has_gpu) {
        fout << " gpu_used_mb=" << gpu_used_mb
             << " gpu_free_mb=" << gpu_free_mb
             << " gpu_total_mb=" << gpu_total_mb;
    } else {
        fout << " gpu=unavailable";
    }

    if (extra != nullptr && extra[0] != '\0') {
        fout << " extra=" << extra;
    }

    fout << "\n";
}

// CPU usage tracker
struct cpu_usage_tracker {
    int64_t last_time_us;
    int64_t last_cpu_time_us;

    cpu_usage_tracker() : last_time_us(0), last_cpu_time_us(0) {}

    float get_cpu_usage() {
#if defined(_WIN32)
        FILETIME create_time, exit_time, kernel_time, user_time;
        if (GetProcessTimes(GetCurrentProcess(), &create_time, &exit_time, &kernel_time, &user_time)) {
            int64_t now_us = ggml_time_us();
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

// ========================================
// Training configuration
// ========================================

struct training_config {
    int32_t max_train_samples = 10;
    int32_t max_tokens_per_sample = 512;
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

// Progress callback with CPU and memory monitoring
static void progress_callback(
        bool train,
        int64_t iter,
        int64_t iter_max,
        float loss,
        int64_t t_start_us,
        cpu_usage_tracker & cpu_tracker) {
    const int64_t t_now_us = ggml_time_us();
    const float elapsed = (t_now_us - t_start_us) / 1.0e6f;

    float memory_mb = get_memory_usage_mb();
    float cpu_percent = cpu_tracker.get_cpu_usage();

    fprintf(stderr, "\r[%s] Iter %6ld/%6ld | Loss: %.6f | Time: %6.2fs | %.1f it/s | CPU: %5.1f%% | Mem: %7.1f MB",
            train ? "TRAIN" : "EVAL ",
            iter, iter_max, loss, elapsed,
            iter > 0 ? iter / elapsed : 0.0f,
            cpu_percent,
            memory_mb);
    fflush(stderr);
}

// ========================================
// R-AdaZO fine-tuning loop
// ========================================

static void finetune_radazo_quant(
        struct llama_context * ctx,
        const std::vector<llama_token> & train_tokens,
        const std::vector<llama_token> & /* eval_tokens */,
        RAdaZOOptimizer & optimizer,
        int n_epochs) {

    LOG_INF("\n%s: starting R-AdaZO fine-tuning (quantized model, direct tensor modification)...\n", __func__);

    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);
    const struct llama_model * model_ptr = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ptr));

    LOG_INF("%s: n_ctx=%d, n_batch=%d, n_train_tokens=%zu, n_epochs=%d\n",
            __func__, n_ctx, n_batch, train_tokens.size(), n_epochs);

    cpu_usage_tracker cpu_tracker;

    // Training loop
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        LOG_INF("\n%s: ===== Epoch %d/%d =====\n", __func__, epoch + 1, n_epochs);
        append_memory_log_train("epoch_begin", epoch + 1, 0, 0);

        const int64_t t_epoch_start = ggml_time_us();
        int64_t n_batches = 0;
        float epoch_loss = 0.0f;

        // Process training data in batches
        for (size_t i = 0; i + n_batch < train_tokens.size(); i += n_batch) {
            const llama_token target_token = train_tokens[i + n_batch];
            append_memory_log_train("batch_begin", epoch + 1, n_batches + 1, n_batch);

            // Clear KV cache at start of each batch
            llama_memory_clear(llama_get_memory(ctx), true);
            append_memory_log_train("after_kv_clear", epoch + 1, n_batches + 1, n_batch);

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
            append_memory_log_train("after_baseline_decode", epoch + 1, n_batches + 1, n_batch);

            // *** R-AdaZO optimizer step ***
            // The optimizer handles quantized tensors transparently:
            // dequantize -> perturb in FP32 -> requantize -> forward -> restore -> Adam update
            float batch_loss = optimizer.step(ctx, batch, n_vocab, target_token);
            append_memory_log_train("after_optimizer_step", epoch + 1, n_batches + 1, n_batch);

            epoch_loss += batch_loss;
            n_batches++;

            // Progress reporting (print every iteration)
            progress_callback(true, n_batches,
                train_tokens.size() / n_batch, epoch_loss / n_batches, t_epoch_start, cpu_tracker);

            llama_batch_free(batch);
            append_memory_log_train("after_batch_free", epoch + 1, n_batches, n_batch);
        }

        fprintf(stderr, "\n");
        LOG_INF("%s: Epoch %d complete - Avg Loss: %.6f\n",
                __func__, epoch + 1, epoch_loss / n_batches);
        append_memory_log_train("epoch_end", epoch + 1, n_batches, n_batch);
    }

    // Print optimizer statistics
    LOG_INF("\n%s: R-AdaZO Optimizer Statistics:\n", __func__);
    LOG_INF("%s:   Total parameter updates: %ld\n", __func__, optimizer.get_total_updates());
    LOG_INF("%s:   Total forward passes: %ld\n", __func__, optimizer.get_forward_passes());
    LOG_INF("%s:   Average forward passes per update: %.2f\n", __func__,
            optimizer.get_total_updates() > 0 ?
            (float)optimizer.get_forward_passes() / optimizer.get_total_updates() : 0.0f);

    // Print final resource usage
    LOG_INF("\n%s: Final Resource Usage:\n", __func__);
    LOG_INF("%s:   Memory: %.1f MB\n", __func__, get_memory_usage_mb());
    LOG_INF("%s:   CPU: %.1f%%\n", __func__, cpu_tracker.get_cpu_usage());

    LOG_INF("\n%s: fine-tuning complete!\n", __func__);
}

// ========================================
// Main
// ========================================

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    // Set defaults suitable for R-AdaZO
    params.n_batch = 8;
    params.n_ctx = 512;

    // Default GSM8K dataset path (try multiple locations)
    std::string gsm8k_file;
    std::vector<std::string> possible_paths = {
        "examples/zeroth-order-opt/gsm8k_test.jsonl",
        "../examples/zeroth-order-opt/gsm8k_test.jsonl",
        "../../examples/zeroth-order-opt/gsm8k_test.jsonl",
        "../../../examples/zeroth-order-opt/gsm8k_test.jsonl"
    };

    for (const auto & path : possible_paths) {
        std::ifstream test_file(path);
        if (test_file.good()) {
            gsm8k_file = path;
            break;
        }
    }

    if (gsm8k_file.empty()) {
        gsm8k_file = possible_paths[0];
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Reset memory profiling log file as early as possible so startup baselines are kept.
    {
        std::ofstream fout(k_memory_log_path, std::ios::trunc);
        if (fout.is_open()) {
            fout << "# memory profile log\n";
            fout << "# fields: scope epoch iter step stage param_idx sample_idx batch_tokens cpu_mb gpu_used_mb gpu_free_mb gpu_total_mb extra\n";
        } else {
            LOG_ERR("%s: failed to open %s for memory logging\n", __func__, k_memory_log_path);
        }
    }

    // Baseline before backend/model initialization.
    append_memory_log_train("baseline_before_backend_init", 0, 0, 0);

    // Force settings for optimization (required for in-place tensor updates)
    if (params.use_mmap) {
        LOG_INF("%s: disabling memory mapping for weight updates\n", __func__);
        params.use_mmap = false;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    append_memory_log_train("baseline_after_backend_init", 0, 0, 0);

    // Load model and context
    append_memory_log_train("baseline_before_model_load", 0, 0, 0);
    common_init_result llama_init = common_init_from_params(params);
    llama_model_ptr & model = llama_init.model;
    llama_context_ptr & ctx = llama_init.context;
    append_memory_log_train("baseline_after_model_load", 0, 0, 0);

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n%s: ===== R-AdaZO FINE-TUNING (QUANTIZED MODEL) =====\n", __func__);
    LOG_INF("%s: Model: %s\n", __func__, params.model.path.c_str());

    // Detect model tensor type
    {
        const size_t n_tensors = llama_model_n_tensors(model.get());
        char tname[256];
        int n_fp32 = 0, n_quant = 0;
        for (size_t i = 0; i < n_tensors && i < 20; ++i) {
            struct ggml_tensor * t = llama_model_get_tensor_by_index(model.get(), i, tname, sizeof(tname));
            if (t == nullptr) continue;
            if (ggml_is_quantized(t->type)) n_quant++;
            else n_fp32++;
        }
        LOG_INF("%s: Tensor types detected (sample): %d quantized, %d FP32\n",
                __func__, n_quant, n_fp32);
        if (n_quant > 0) {
            LOG_INF("%s: Quantized model detected - using dequant/requant for parameter updates\n", __func__);
        }
    }

    // Training configuration
    training_config train_config;
    train_config.max_train_samples = 10;
    train_config.max_tokens_per_sample = 512;

    // Load GSM8K dataset
    LOG_INF("%s: Loading GSM8K dataset from: %s\n", __func__, gsm8k_file.c_str());
    auto gsm8k_examples = load_gsm8k(gsm8k_file, train_config.max_train_samples);

    if (gsm8k_examples.empty()) {
        LOG_ERR("%s: failed to load GSM8K examples\n", __func__);
        return 1;
    }

    LOG_INF("%s: Loaded %zu GSM8K examples\n", __func__, gsm8k_examples.size());

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

    // *** SETUP LoRA + R-AdaZO OPTIMIZER ***
    lora_config lora_cfg;
    LoRAAdapter lora_adapter(ctx.get(), lora_cfg);
    if (!lora_adapter.initialize()) {
        LOG_ERR("%s: failed to initialize LoRA adapters\n", __func__);
        return 1;
    }

    auto trainable_params = lora_adapter.get_trainable_params();
    LOG_INF("%s: LoRA layers: %zu, trainable LoRA tensors (A/B): %zu\n",
            __func__, lora_adapter.get_num_layers(), trainable_params.size());
    LOG_INF("%s: total LoRA parameters: %ld\n\n", __func__, lora_adapter.get_total_params());

    if (trainable_params.empty()) {
        LOG_ERR("%s: no trainable parameters found!\n", __func__);
        return 1;
    }

    // Configure R-AdaZO parameters
    radazo_params radazo_config;
    radazo_config.lr = 1e-4f;
    radazo_config.beta1 = 0.9f;
    radazo_config.beta2 = 0.999f;
    radazo_config.eps = 1e-8f;
    radazo_config.mu = 5e-3f;
    // Faster default profile: still performs R-AdaZO, but with fewer perturbation passes.
    // This is critical for large quantized tensors where dequant/requant dominates runtime.
    radazo_config.n_samples = 1;
    radazo_config.n_params_per_iter = 2;
    radazo_config.full_tensor_gradient = true;
    radazo_config.log_gradients = false;

    LOG_INF("%s: R-AdaZO Configuration:\n", __func__);
    LOG_INF("%s:   lr = %.2e (learning rate)\n", __func__, radazo_config.lr);
    LOG_INF("%s:   beta1 = %.3f (first moment decay)\n", __func__, radazo_config.beta1);
    LOG_INF("%s:   beta2 = %.3f (second moment decay)\n", __func__, radazo_config.beta2);
    LOG_INF("%s:   mu = %.2e (perturbation magnitude)\n", __func__, radazo_config.mu);
    LOG_INF("%s:   n_samples = %d (multiple perturbations per gradient)\n", __func__, radazo_config.n_samples);
    LOG_INF("%s:   n_params_per_iter = %d\n", __func__, radazo_config.n_params_per_iter);
    LOG_INF("%s:   full_tensor_gradient = %s\n", __func__,
            radazo_config.full_tensor_gradient ? "true" : "false");
    LOG_INF("%s:   Approx forward passes per batch = 1 + %d * %d = %d\n\n",
            __func__,
            radazo_config.n_params_per_iter,
            radazo_config.n_samples,
            1 + radazo_config.n_params_per_iter * radazo_config.n_samples);

    // Create the R-AdaZO optimizer with LoRA A/B tensors only.
    RAdaZOOptimizer optimizer(radazo_config, trainable_params);
    optimizer.set_logits_postprocessor([&lora_adapter](struct llama_context * ctx, float * logits, int n_vocab) {
        lora_adapter.apply_lora_to_logits(ctx, logits, n_vocab);
    });

    // Run fine-tuning
    int n_epochs = 1;
    finetune_radazo_quant(ctx.get(), train_set, eval_set, optimizer, n_epochs);

    // Save by merging LoRA adapters into base model.
    std::string output_file = params.out_file.empty() ?
        "model_radazo_finetuned.gguf" : params.out_file;

    LOG_INF("\n%s: saving fine-tuned model to %s\n", __func__, output_file.c_str());

    // Ensure all backend updates are visible before merge.
    llama_synchronize(ctx.get());
    if (!lora_adapter.merge_and_save(model.get(), output_file)) {
        LOG_ERR("%s: failed to merge LoRA adapters and save model\n", __func__);
        return 1;
    }

    LOG_INF("%s: model saved successfully!\n", __func__);
    LOG_INF("\n%s: ===== SUMMARY =====\n", __func__);
    LOG_INF("%s: Method: R-AdaZO (direct quantized tensor modification)\n", __func__);
    LOG_INF("%s: Dataset: GSM8K (Grade School Math)\n", __func__);
    LOG_INF("%s: Training samples: %zu\n", __func__, gsm8k_examples.size());
    LOG_INF("%s: Epochs: %d\n", __func__, n_epochs);
    LOG_INF("%s: Training tokens: %zu\n", __func__, train_set.size());
    LOG_INF("%s: Output model: %s\n", __func__, output_file.c_str());
    LOG_INF("\n%s: How it works for quantized models:\n", __func__);
    LOG_INF("%s:   1. Dequantize tensor to FP32\n", __func__);
    LOG_INF("%s:   2. Perturb in FP32 space\n", __func__);
    LOG_INF("%s:   3. Requantize and set for forward pass\n", __func__);
    LOG_INF("%s:   4. Estimate gradient from loss difference\n", __func__);
    LOG_INF("%s:   5. Apply Adam update in FP32, requantize, write back\n", __func__);
    LOG_INF("%s:   -> Model saved in original quantized format with updated weights\n", __func__);

    llama_backend_free();

    return 0;
}
