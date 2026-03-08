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
#include <algorithm>
#include <utility>
#include <limits>
#include <cstdlib>

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
static constexpr bool k_enable_memory_logging = false;

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
    if (!k_enable_memory_logging) {
        return;
    }
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
    int32_t max_eval_samples = 10;
    int32_t max_tokens_per_sample = 512;
};

// DROP data structure
struct drop_example {
    std::string section_id;
    std::string query_id;
    std::string passage;
    std::string question;
    std::string answer_span;
};

static std::string json_value_to_string(const json & v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    if (v.is_number_unsigned()) {
        return std::to_string(v.get<unsigned long long>());
    }
    if (v.is_number_float()) {
        std::ostringstream oss;
        oss << v.get<double>();
        return oss.str();
    }
    return v.dump();
}

// Load DROP dataset from a JSON array file without parsing the full file into memory.
// This keeps startup memory lower for very large train splits.
static std::vector<drop_example> load_drop_json_array_stream(const std::string & filepath, int max_samples = -1) {
    std::vector<drop_example> examples;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        LOG_ERR("%s: failed to open file: %s\n", __func__, filepath.c_str());
        return examples;
    }

    std::string obj_buf;
    obj_buf.reserve(4096);
    bool in_string = false;
    bool escape = false;
    bool capturing = false;
    int depth = 0;
    int count = 0;
    char ch = 0;
    while (file.get(ch) && (max_samples <= 0 || count < max_samples)) {
        if (!capturing) {
            if (ch == '{') {
                capturing = true;
                depth = 1;
                in_string = false;
                escape = false;
                obj_buf.clear();
                obj_buf.push_back(ch);
            }
            continue;
        }

        obj_buf.push_back(ch);

        if (escape) {
            escape = false;
            continue;
        }
        if (ch == '\\') {
            escape = true;
            continue;
        }
        if (ch == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) {
            continue;
        }
        if (ch == '{') {
            depth++;
            continue;
        }
        if (ch == '}') {
            depth--;
            if (depth > 0) {
                continue;
            }

            capturing = false;
            depth = 0;
            json j = json::parse(obj_buf, nullptr, false);
            if (j.is_discarded()) {
                continue;
            }

            if (!j.contains("passage") || !j.contains("question") || !j.contains("answers_spans")) {
                continue;
            }
            if (!j["answers_spans"].is_object()) {
                continue;
            }

            const json & ans = j["answers_spans"];
            if (!ans.contains("spans")) {
                continue;
            }

            drop_example ex;
            ex.section_id = j.value("section_id", "");
            ex.query_id = j.value("query_id", "");
            ex.passage = j.value("passage", "");
            ex.question = j.value("question", "");

            const json & spans = ans["spans"];
            if (spans.is_array()) {
                if (spans.empty()) {
                    continue;
                }
                ex.answer_span = json_value_to_string(spans[0]);
            } else {
                ex.answer_span = json_value_to_string(spans);
            }

            if (ex.passage.empty() || ex.question.empty() || ex.answer_span.empty()) {
                continue;
            }

            examples.push_back(std::move(ex));
            count++;
        }
    }

    LOG_INF("%s: loaded %zu DROP examples from %s\n", __func__, examples.size(), filepath.c_str());
    return examples;
}

// Format DROP example for training
static std::string format_drop_prompt(const drop_example & ex) {
    std::stringstream ss;
    ss << "Passage: " << ex.passage << "\n";
    ss << "Question: " << ex.question << "\n";
    ss << "Answer: " << ex.answer_span;
    return ss.str();
}

// Convert DROP examples to per-sample token sequences.
// Each training sample remains independent (no sequence packing).
static std::vector<std::vector<llama_token>> drop_to_token_sequences(
        struct llama_context * ctx,
        const std::vector<drop_example> & examples,
        int max_tokens_per_sample = -1) {

    std::vector<std::vector<llama_token>> tokenized_samples;
    tokenized_samples.reserve(examples.size());
    size_t total_tokens = 0;

    LOG_INF("%s: tokenizing %zu DROP examples...\n", __func__, examples.size());

    for (size_t i = 0; i < examples.size(); ++i) {
        try {
            std::string prompt = format_drop_prompt(examples[i]);
            std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true);

            if (max_tokens_per_sample > 0 && tokens.size() > (size_t)max_tokens_per_sample) {
                tokens.resize(max_tokens_per_sample);
            }

            total_tokens += tokens.size();
            tokenized_samples.push_back(std::move(tokens));

            if ((i + 1) % 10 == 0) {
                LOG_INF("%s: tokenized %zu/%zu examples (%zu tokens total)\n",
                        __func__, i + 1, examples.size(), total_tokens);
            }
        } catch (const std::exception & e) {
            LOG_ERR("%s: error tokenizing DROP example %zu: %s\n", __func__, i, e.what());
        }
    }

    LOG_INF("%s: tokenized DROP samples=%zu, total tokens=%zu\n",
            __func__, tokenized_samples.size(), total_tokens);

    return tokenized_samples;
}

static size_t count_total_tokens(const std::vector<std::vector<llama_token>> & samples) {
    size_t total = 0;
    for (const auto & sample : samples) {
        total += sample.size();
    }
    return total;
}

static bool parse_int32_arg(const char * s, int32_t & out) {
    if (s == nullptr || s[0] == '\0') {
        return false;
    }
    char * end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    out = (int32_t) v;
    return true;
}

static int32_t parse_epochs_from_argv(int argc, char ** argv, int32_t fallback) {
    int32_t epochs = fallback;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--epochs" && i + 1 < argc) {
            int32_t parsed = 0;
            if (parse_int32_arg(argv[i + 1], parsed) && parsed > 0) {
                epochs = parsed;
            }
            i++;
        } else if (a.rfind("--epochs=", 0) == 0) {
            int32_t parsed = 0;
            const std::string v = a.substr(std::string("--epochs=").size());
            if (parse_int32_arg(v.c_str(), parsed) && parsed > 0) {
                epochs = parsed;
            }
        }
    }
    return epochs;
}

static std::vector<std::vector<llama_token>> filter_short_samples(
        const std::vector<std::vector<llama_token>> & samples) {
    std::vector<std::vector<llama_token>> out;
    out.reserve(samples.size());
    for (const auto & s : samples) {
        if (s.size() >= 2) {
            out.push_back(s);
        }
    }
    return out;
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

static bool build_sample_batch(
        const std::vector<llama_token> & sample_tokens,
        int n_ctx,
        llama_batch & batch,
        std::vector<radazo_loss_target> & loss_targets) {
    const int seq_len = std::min((int) sample_tokens.size(), n_ctx);
    if (seq_len < 2) {
        return false;
    }

    batch = llama_batch_init(seq_len, 0, 1);
    loss_targets.clear();

    // Use the second-to-last position to predict the last token (the answer).
    // Format: "Passage: ...\nQuestion: ...\nAnswer: 3" → last token = answer.
    const int loss_pos = seq_len - 2;

    for (int pos = 0; pos < seq_len; ++pos) {
        batch.token[pos] = sample_tokens[pos];
        batch.pos[pos] = pos;
        batch.n_seq_id[pos] = 1;
        batch.seq_id[pos][0] = 0;
        batch.logits[pos] = (pos == loss_pos);
    }
    batch.n_tokens = seq_len;

    loss_targets.push_back({loss_pos, sample_tokens[loss_pos + 1], 1.0f});
    return true;
}

// ========================================
// R-AdaZO fine-tuning loop
// ========================================

static void finetune_radazo_quant(
        struct llama_context * ctx,
        const std::vector<std::vector<llama_token>> & train_samples,
        const std::vector<std::vector<llama_token>> & /* eval_samples */,
        RAdaZOOptimizer & optimizer,
        LoRAAdapter & lora_adapter,
        int n_epochs) {
    (void) lora_adapter;

    LOG_INF("\n%s: starting R-AdaZO fine-tuning (QLoRA A/B only)...\n", __func__);

    const int n_ctx = llama_n_ctx(ctx);
    const struct llama_model * model_ptr = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ptr));

    LOG_INF("%s: n_ctx=%d, n_train_samples=%zu, n_epochs=%d\n",
            __func__, n_ctx, train_samples.size(), n_epochs);

    cpu_usage_tracker cpu_tracker;
    const int64_t iters_per_epoch = (int64_t) train_samples.size();

    // Native llama_adapter_lora is already registered in lora_adapter.initialize().
    // llama_decode() automatically applies LoRA via the compute graph.
    // Base tensors are never modified during training.

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        LOG_INF("\n%s: ===== Epoch %d/%d =====\n", __func__, epoch + 1, n_epochs);
        append_memory_log_train("epoch_begin", epoch + 1, 0, 0);

        const int64_t t_epoch_start = ggml_time_us();
        int64_t n_done = 0;
        float epoch_loss_sum = 0.0f;

        for (size_t si = 0; si < train_samples.size(); ++si) {
            llama_batch batch = {};
            std::vector<radazo_loss_target> loss_targets;

            if (!build_sample_batch(train_samples[si], n_ctx, batch, loss_targets)) {
                LOG_ERR("%s: sample %zu too short, skipping\n", __func__, si);
                continue;
            }

            // Per-step independent loss returned by optimizer for this sample.
            float sample_loss = optimizer.step(ctx, batch, n_vocab, loss_targets);

            epoch_loss_sum += sample_loss;
            n_done++;

            // Show current step loss (not running average).
            progress_callback(true, n_done, iters_per_epoch,
                sample_loss, t_epoch_start, cpu_tracker);

            llama_batch_free(batch);
        }

        fprintf(stderr, "\n");
        LOG_INF("%s: Epoch %d complete - Avg Loss: %.6f, samples=%ld\n",
                __func__, epoch + 1, n_done > 0 ? epoch_loss_sum / n_done : 0.0f, n_done);
        append_memory_log_train("epoch_end", epoch + 1, n_done, 0);
    }

    LOG_INF("\n%s: R-AdaZO Optimizer Statistics:\n", __func__);
    LOG_INF("%s:   Total parameter updates: %ld\n", __func__, optimizer.get_total_updates());
    LOG_INF("%s:   Total forward passes: %ld\n", __func__, optimizer.get_forward_passes());
    LOG_INF("%s:   Average forward passes per update: %.2f\n", __func__,
            optimizer.get_total_updates() > 0 ?
            (float)optimizer.get_forward_passes() / optimizer.get_total_updates() : 0.0f);

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
    // n_batch is token-capacity for runtime buffers, NOT sample batch size.
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_ctx = 512;
    // Prefer GPU offload by default when CUDA backend is available.
    // User-provided CLI flags (e.g. --n-gpu-layers) still override this.
    params.n_gpu_layers = 999;

    training_config train_config;

    // Default DROP dataset paths (try multiple locations)
    std::string drop_train_file;
    std::vector<std::string> possible_train_paths = {
        "examples/zeroth-order-opt/drop_train.json",
        "../examples/zeroth-order-opt/drop_train.json",
        "../../examples/zeroth-order-opt/drop_train.json",
        "../../../examples/zeroth-order-opt/drop_train.json"
    };
    for (const auto & path : possible_train_paths) {
        std::ifstream test_file(path);
        if (test_file.good()) {
            drop_train_file = path;
            break;
        }
    }
    if (drop_train_file.empty()) {
        drop_train_file = possible_train_paths[0];
    }

    std::string drop_validation_file;
    std::vector<std::string> possible_validation_paths = {
        "examples/zeroth-order-opt/drop_validation.json",
        "../examples/zeroth-order-opt/drop_validation.json",
        "../../examples/zeroth-order-opt/drop_validation.json",
        "../../../examples/zeroth-order-opt/drop_validation.json"
    };
    for (const auto & path : possible_validation_paths) {
        std::ifstream test_file(path);
        if (test_file.good()) {
            drop_validation_file = path;
            break;
        }
    }
    if (drop_validation_file.empty()) {
        drop_validation_file = possible_validation_paths[0];
    }

    bool save_lora_only = false;
    {
        int n = 0;
        for (int i = 0; i < argc; ++i) {
            if (std::strcmp(argv[i], "--save-lora-only") == 0) {
                save_lora_only = true;
                continue;
            }
            argv[n++] = argv[i];
        }
        argc = n;
        argv[argc] = nullptr;
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Each sample is processed independently as a single-sequence batch,
    // so n_parallel=1 (the default) is correct and avoids the slow multi-seq
    // decode path in llama.cpp.

    // Reset memory profiling log file as early as possible so startup baselines are kept.
    if (k_enable_memory_logging) {
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
    train_config.max_train_samples = 10;
    train_config.max_eval_samples = 10;
    train_config.max_tokens_per_sample = 512;

    // Load DROP train/validation datasets
    LOG_INF("%s: Loading DROP train dataset from: %s\n", __func__, drop_train_file.c_str());
    auto drop_train_examples = load_drop_json_array_stream(drop_train_file, train_config.max_train_samples);

    if (drop_train_examples.empty()) {
        LOG_ERR("%s: failed to load DROP train examples\n", __func__);
        return 1;
    }

    LOG_INF("%s: Loading DROP validation dataset from: %s\n", __func__, drop_validation_file.c_str());
    auto drop_validation_examples = load_drop_json_array_stream(drop_validation_file, train_config.max_eval_samples);
    LOG_INF("%s: Loaded DROP examples - train=%zu, validation=%zu\n",
            __func__, drop_train_examples.size(), drop_validation_examples.size());

    // Show first example
    if (!drop_train_examples.empty()) {
        LOG_INF("%s: First training example:\n", __func__);
        LOG_INF("%s: ----------------------------------------\n", __func__);
        LOG_INF("%s: %s\n", __func__, format_drop_prompt(drop_train_examples[0]).c_str());
        LOG_INF("%s: ----------------------------------------\n\n", __func__);
    }

    // Tokenize DROP train examples as independent sequences (no packing).
    std::vector<std::vector<llama_token>> train_set = drop_to_token_sequences(
        ctx.get(), drop_train_examples, train_config.max_tokens_per_sample);
    train_set = filter_short_samples(train_set);
    if (train_set.empty()) {
        LOG_ERR("%s: tokenized DROP train dataset is empty\n", __func__);
        return 1;
    }

    // Tokenize DROP validation examples for eval bookkeeping.
    std::vector<std::vector<llama_token>> eval_set = drop_to_token_sequences(
        ctx.get(), drop_validation_examples, train_config.max_tokens_per_sample);
    eval_set = filter_short_samples(eval_set);

    LOG_INF("%s: train_samples=%zu (%zu tokens), eval_samples=%zu (%zu tokens)\n\n",
            __func__,
            train_set.size(),
            count_total_tokens(train_set),
            eval_set.size(),
            count_total_tokens(eval_set));

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

    // Configure R-AdaZO / Global-SPSA parameters.
    // Each sample now costs exactly 2 * n_samples forward passes (constant).
    radazo_params radazo_config;
    radazo_config.lr       = 1e-5f;
    radazo_config.beta1    = 0.9f;
    radazo_config.beta2    = 0.999f;
    radazo_config.eps      = 1e-8f;
    radazo_config.mu       = 1e-4f;
    radazo_config.n_samples = 1;   // global SPSA draws per sample (each = 2 forward passes)
    radazo_config.log_gradients = true;

    LOG_INF("%s: R-AdaZO Configuration:\n", __func__);
    LOG_INF("%s:   lr = %.2e\n",     __func__, radazo_config.lr);
    LOG_INF("%s:   beta1 = %.3f\n",  __func__, radazo_config.beta1);
    LOG_INF("%s:   beta2 = %.3f\n",  __func__, radazo_config.beta2);
    LOG_INF("%s:   mu = %.2e\n",     __func__, radazo_config.mu);
    LOG_INF("%s:   n_samples = %d  (global SPSA draws; 2 forward passes each)\n",
            __func__, radazo_config.n_samples);
    LOG_INF("%s:   forward passes per sample = %d\n\n",
            __func__, 2 * radazo_config.n_samples);

    // Sync backend before optimizer reads LoRA tensors (may be on GPU).
    llama_synchronize(ctx.get());

    // Create the R-AdaZO optimizer.
    // No param_change_hook or pair_param_hook needed: native llama_adapter_lora
    // handles LoRA automatically inside llama_decode().
    RAdaZOOptimizer optimizer(radazo_config, trainable_params);

    // Run fine-tuning
    int n_epochs = parse_epochs_from_argv(argc, argv, 1);
    finetune_radazo_quant(ctx.get(), train_set, eval_set, optimizer, lora_adapter, n_epochs);

    // Save: either merge into base model, or export standalone FP32 LoRA (Phase 3 verification).
    std::string output_file = params.out_file.empty() ?
        (save_lora_only ? "lora_fp32.gguf" : "model_radazo_finetuned.gguf") : params.out_file;

    LOG_INF("\n%s: saving %s to %s\n", __func__,
            save_lora_only ? "standalone FP32 LoRA (no merge)" : "fine-tuned model", output_file.c_str());

    // Ensure all backend updates are visible before save.
    llama_synchronize(ctx.get());
    if (save_lora_only) {
        if (!lora_adapter.save_lora_standalone(model.get(), output_file)) {
            LOG_ERR("%s: failed to save standalone LoRA adapter\n", __func__);
            return 1;
        }
        LOG_INF("%s: LoRA saved. Verify with: ./llama-cli -m <baseline.gguf> --lora %s -p \"...\"\n",
                __func__, output_file.c_str());
    } else {
        // Two-stage flow:
        //  1) train with GPU-offloaded model
        //  2) reload a CPU-weight model for stable read/write during merge
        common_params params_merge = params;
        params_merge.n_gpu_layers = 0;
        params_merge.tensor_buft_overrides.clear();
        params_merge.tensor_buft_overrides.push_back({ ".*\\.weight$", ggml_backend_cpu_buffer_type() });
        params_merge.tensor_buft_overrides.push_back({ nullptr, nullptr });

        LOG_INF("%s: reloading model on CPU buffers for merge stage\n", __func__);
        common_init_result merge_init = common_init_from_params(params_merge);
        if (merge_init.model == nullptr) {
            LOG_ERR("%s: failed to load merge-stage model\n", __func__);
            return 1;
        }

        if (!lora_adapter.merge_and_save(merge_init.model.get(), output_file)) {
            LOG_ERR("%s: failed to merge LoRA adapters and save model\n", __func__);
            return 1;
        }
    }

    LOG_INF("%s: save completed successfully!\n", __func__);
    LOG_INF("\n%s: ===== SUMMARY =====\n", __func__);
    LOG_INF("%s: Method: LoRA + R-AdaZO (quantized base model)\n", __func__);
    LOG_INF("%s: Dataset: DROP (passage/question/answer span)\n", __func__);
    LOG_INF("%s: Loaded train examples: %zu\n", __func__, drop_train_examples.size());
    LOG_INF("%s: Loaded validation examples: %zu\n", __func__, drop_validation_examples.size());
    LOG_INF("%s: Epochs: %d\n", __func__, n_epochs);
    LOG_INF("%s: Training samples: %zu\n", __func__, train_set.size());
    LOG_INF("%s: Training tokens: %zu\n", __func__, count_total_tokens(train_set));
    LOG_INF("%s: Output model: %s\n", __func__, output_file.c_str());
    LOG_INF("\n%s: How it works (Global-SPSA QLoRA on quantized models):\n", __func__);
    LOG_INF("%s:   1. Native llama_adapter_lora registered at startup\n", __func__);
    LOG_INF("%s:   2. Base GGUF tensors are read-only throughout training\n", __func__);
    LOG_INF("%s:   3. llama_decode() auto-computes Y = W_base*X + (a/r)*B*A*X\n", __func__);
    LOG_INF("%s:   4. Global SPSA: ALL LoRA A/B perturbed simultaneously\n", __func__);
    LOG_INF("%s:      → 2 forward passes per sample (constant, not per-param)\n", __func__);
    LOG_INF("%s:   5. Adam update applied to all LoRA A/B in one step\n", __func__);
    LOG_INF("%s:   6. Final GGUF saved by merging trained LoRA into base once\n", __func__);

    llama_backend_free();

    return 0;
}
