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
    int32_t train_batch_size = 8;
};

// SST-2 data structure
struct sst2_example {
    std::string sentence;
    std::string label_str;  // "positive" or "negative"
    int label_idx = 0;      // optional numeric label from JSON
};

// SFT sample: prompt tokens + label tokens (+ optional EOS)
struct sft_sample {
    std::vector<llama_token> tokens;
    int prompt_length = 0;        // number of prompt tokens
    llama_token label_token = -1; // first label token ("positive"/"negative")
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

// Load SST-2 dataset from a JSON array file without parsing the full file into memory.
// This keeps startup memory lower for very large train splits.
static std::vector<sst2_example> load_sst2_json_array_stream(const std::string & filepath, int max_samples = -1) {
    std::vector<sst2_example> examples;
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

            if (!j.contains("passage") || !j.contains("answers_spans")) {
                continue;
            }
            if (!j["answers_spans"].is_object()) {
                continue;
            }

            const json & ans = j["answers_spans"];
            if (!ans.contains("spans")) {
                continue;
            }

            sst2_example ex;
            ex.sentence = j.value("passage", "");
            ex.label_idx = j.value("label", 0);

            const json & spans = ans["spans"];
            if (spans.is_array()) {
                if (spans.empty()) {
                    continue;
                }
                ex.label_str = json_value_to_string(spans[0]);
            } else {
                ex.label_str = json_value_to_string(spans);
            }

            // Normalize labels to the expected two classes.
            if (ex.label_str == "1") ex.label_str = "positive";
            if (ex.label_str == "0") ex.label_str = "negative";

            if (ex.sentence.empty() || ex.label_str.empty()) {
                continue;
            }
            if (ex.label_str != "positive" && ex.label_str != "negative") {
                continue;
            }

            examples.push_back(std::move(ex));
            count++;
        }
    }

    LOG_INF("%s: loaded %zu SST-2 examples from %s\n", __func__, examples.size(), filepath.c_str());
    return examples;
}

// Format SST-2 classification prompt.
static std::string format_sst2_prompt(const sst2_example & ex) {
    std::stringstream ss;
    ss << "You are a strict sentiment classification expert. "
          "Output ONLY 'positive' or 'negative'. "
          "Do not output any other text, explanation, or punctuation.\n\n";
    ss << "Review: " << ex.sentence << "\n";
    ss << "Sentiment: ";
    return ss.str();
}

// Convert SST-2 examples to per-sample SFT sequences.
// Prompt and label are tokenized separately to avoid tokenizer boundary quirks.
static std::vector<sft_sample> sst2_to_sft_sequences(
        struct llama_context * ctx,
        const std::vector<sst2_example> & examples,
        int max_tokens_per_sample = -1) {
    std::vector<sft_sample> dataset;
    dataset.reserve(examples.size());
    size_t total_tokens = 0;

    const struct llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    LOG_INF("%s: tokenizing %zu SST-2 examples...\n", __func__, examples.size());

    for (size_t i = 0; i < examples.size(); ++i) {
        try {
            const std::string prompt_str = format_sst2_prompt(examples[i]);
            const std::string answer_str = examples[i].label_str;
            std::vector<llama_token> prompt_tokens = common_tokenize(ctx, prompt_str, true);
            std::vector<llama_token> answer_tokens = common_tokenize(ctx, answer_str, false);
            if (prompt_tokens.empty() || answer_tokens.empty()) {
                continue;
            }

            sft_sample sample;
            sample.prompt_length = (int) prompt_tokens.size();
            sample.label_token = answer_tokens[0];
            sample.tokens = std::move(prompt_tokens);
            sample.tokens.insert(sample.tokens.end(), answer_tokens.begin(), answer_tokens.end());
            sample.tokens.push_back(llama_vocab_eos(vocab));

            if (max_tokens_per_sample > 0 && sample.tokens.size() > (size_t)max_tokens_per_sample) {
                sample.tokens.resize(max_tokens_per_sample);
            }
            if ((int)sample.tokens.size() <= sample.prompt_length) {
                continue;
            }

            total_tokens += sample.tokens.size();
            dataset.push_back(std::move(sample));

            if ((i + 1) % 10 == 0) {
                LOG_INF("%s: tokenized %zu/%zu examples (%zu tokens total)\n",
                        __func__, i + 1, examples.size(), total_tokens);
            }
        } catch (const std::exception & e) {
            LOG_ERR("%s: error tokenizing SST-2 example %zu: %s\n", __func__, i, e.what());
        }
    }

    LOG_INF("%s: tokenized SST-2 samples=%zu, total tokens=%zu\n",
            __func__, dataset.size(), total_tokens);

    return dataset;
}

static size_t count_total_tokens(const std::vector<sft_sample> & samples) {
    size_t total = 0;
    for (const auto & sample : samples) {
        total += sample.tokens.size();
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

static int32_t estimate_packed_batch_count(
        const std::vector<sft_sample> & samples,
        int total_token_budget,
        int per_seq_ctx,
        int max_seqs_per_batch) {
    if (samples.empty()) {
        return 0;
    }

    int32_t count = 0;
    size_t i = 0;
    while (i < samples.size()) {
        int used_tokens = 0;
        int used_seqs = 0;

        while (i < samples.size() && used_seqs < max_seqs_per_batch) {
            const int seq_len = std::min((int) samples[i].tokens.size(), per_seq_ctx);
            if (seq_len < 2 || samples[i].prompt_length >= seq_len) {
                ++i;
                continue;
            }
            if (used_seqs > 0 && used_tokens + seq_len > total_token_budget) {
                break;
            }
            used_tokens += seq_len;
            used_seqs++;
            ++i;
        }

        if (used_seqs > 0) {
            count++;
        }
    }

    return count;
}

static std::vector<sft_sample> filter_short_samples(
        const std::vector<sft_sample> & samples) {
    std::vector<sft_sample> out;
    out.reserve(samples.size());
    for (const auto & s : samples) {
        if (s.prompt_length >= 1 && (int)s.tokens.size() > s.prompt_length && s.label_token >= 0) {
            out.push_back(s);
        }
    }
    return out;
}

static bool build_packed_batch(
        const std::vector<sft_sample> & samples,
        size_t start_idx,
        int total_token_budget,
        int per_seq_ctx,
        int max_seqs_per_batch,
        llama_batch & batch,
        std::vector<radazo_loss_target> & loss_targets,
        size_t & next_idx,
        int & packed_samples) {
    std::vector<std::pair<size_t, int>> selected;
    selected.reserve(max_seqs_per_batch);

    int total_tokens = 0;
    next_idx = start_idx;
    packed_samples = 0;
    loss_targets.clear();

    for (size_t i = start_idx; i < samples.size() && packed_samples < max_seqs_per_batch; ++i) {
        const int seq_len = std::min((int) samples[i].tokens.size(), per_seq_ctx);
        if (seq_len < 2 || samples[i].prompt_length >= seq_len) {
            next_idx = i + 1;
            continue;
        }

        if (packed_samples > 0 && total_tokens + seq_len > total_token_budget) {
            break;
        }

        selected.push_back({ i, seq_len });
        total_tokens += seq_len;
        packed_samples++;
        next_idx = i + 1;
    }

    if (selected.empty()) {
        return false;
    }

    batch = llama_batch_init(total_tokens, 0, packed_samples);
    int batch_pos = 0;

    for (int seq_id = 0; seq_id < packed_samples; ++seq_id) {
        const auto & item = selected[(size_t) seq_id];
        const sft_sample & sample = samples[item.first];
        const int seq_len = item.second;
        const int loss_pos = sample.prompt_length - 1;

        for (int pos = 0; pos < seq_len; ++pos) {
            batch.token[batch_pos] = sample.tokens[(size_t) pos];
            batch.pos[batch_pos] = pos;
            batch.n_seq_id[batch_pos] = 1;
            batch.seq_id[batch_pos][0] = seq_id;
            batch.logits[batch_pos] = (pos == loss_pos);
            if (pos == loss_pos) {
                loss_targets.push_back({ batch_pos, sample.tokens[(size_t) loss_pos + 1], 1.0f });
            }
            batch_pos++;
        }
    }

    batch.n_tokens = batch_pos;
    return true;
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
        const sft_sample & sample,
        int n_ctx,
        llama_batch & batch,
        std::vector<radazo_loss_target> & loss_targets) {
    const int seq_len = std::min((int) sample.tokens.size(), n_ctx);
    if (seq_len < 2) {
        return false;
    }

    batch = llama_batch_init(seq_len, 0, 1);
    loss_targets.clear();

    // Single-point classification target:
    // predict first label token at the position right after prompt.
    const int loss_pos = sample.prompt_length - 1;
    if (loss_pos < 0 || loss_pos + 1 >= seq_len) {
        return false;
    }

    for (int pos = 0; pos < seq_len; ++pos) {
        batch.token[pos] = sample.tokens[pos];
        batch.pos[pos] = pos;
        batch.n_seq_id[pos] = 1;
        batch.seq_id[pos][0] = 0;
        batch.logits[pos] = (pos == loss_pos);
    }
    batch.n_tokens = seq_len;

    loss_targets.push_back({loss_pos, sample.tokens[loss_pos + 1], 1.0f});
    return true;
}

static float evaluate_sst2_accuracy(
        struct llama_context * ctx,
        const std::vector<sft_sample> & eval_samples,
        int n_ctx) {
    if (eval_samples.empty()) {
        return 0.0f;
    }

    std::vector<llama_token> pos_toks = common_tokenize(ctx, "positive", false);
    std::vector<llama_token> neg_toks = common_tokenize(ctx, "negative", false);
    if (pos_toks.empty() || neg_toks.empty()) {
        return 0.0f;
    }
    const llama_token pos_tok = pos_toks[0];
    const llama_token neg_tok = neg_toks[0];

    int64_t correct = 0;
    int64_t total = 0;
    for (const auto & sample : eval_samples) {
        const int prompt_len = std::min(sample.prompt_length, n_ctx);
        if (prompt_len < 1 || sample.label_token < 0) {
            continue;
        }

        llama_batch batch = llama_batch_init(prompt_len, 0, 1);
        for (int pos = 0; pos < prompt_len; ++pos) {
            batch.token[pos] = sample.tokens[pos];
            batch.pos[pos] = pos;
            batch.n_seq_id[pos] = 1;
            batch.seq_id[pos][0] = 0;
            batch.logits[pos] = (pos == prompt_len - 1);
        }
        batch.n_tokens = prompt_len;

        llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);
        llama_synchronize(ctx);
        if (llama_decode(ctx, batch) == 0) {
            float * logits = llama_get_logits_ith(ctx, prompt_len - 1);
            if (logits != nullptr) {
                const bool pred_pos = logits[pos_tok] > logits[neg_tok];
                const bool true_pos = sample.label_token == pos_tok;
                if (pred_pos == true_pos) {
                    correct++;
                }
                total++;
            }
        }
        llama_batch_free(batch);
    }

    return total > 0 ? (float)correct / (float)total : 0.0f;
}

static void run_sst2_train_sanity_check(
        struct llama_context * ctx,
        const std::vector<sft_sample> & train_samples,
        int n_ctx) {
    if (train_samples.empty()) {
        LOG_WRN("%s: skip sanity check (empty train set)\n", __func__);
        return;
    }

    const auto & sample = train_samples[0];
    const int prompt_len = std::min(sample.prompt_length, n_ctx);
    if (prompt_len < 1 || (int)sample.tokens.size() <= prompt_len) {
        LOG_WRN("%s: skip sanity check (invalid sample shape)\n", __func__);
        return;
    }

    std::vector<llama_token> pos_toks = common_tokenize(ctx, "positive", false);
    std::vector<llama_token> neg_toks = common_tokenize(ctx, "negative", false);
    if (pos_toks.empty() || neg_toks.empty()) {
        LOG_WRN("%s: skip sanity check (failed to tokenize labels)\n", __func__);
        return;
    }
    const llama_token pos_tok = pos_toks[0];
    const llama_token neg_tok = neg_toks[0];

    llama_batch batch = llama_batch_init(prompt_len, 0, 1);
    for (int i = 0; i < prompt_len; ++i) {
        batch.token[i] = sample.tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == prompt_len - 1);
    }
    batch.n_tokens = prompt_len;

    llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);
    llama_synchronize(ctx);

    LOG_INF("\n=== training-end sanity check (sample 0) ===\n");
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: sanity decode failed\n", __func__);
        llama_batch_free(batch);
        return;
    }

    float * logits = llama_get_logits_ith(ctx, prompt_len - 1);
    if (logits == nullptr) {
        LOG_ERR("%s: sanity logits are null\n", __func__);
        llama_batch_free(batch);
        return;
    }

    const llama_token target_tok = sample.tokens[prompt_len];
    LOG_INF("%s: target token=%d, positive token=%d, negative token=%d\n",
            __func__, target_tok, pos_tok, neg_tok);
    LOG_INF("%s: logits -> positive=%.6f, negative=%.6f, target=%.6f\n",
            __func__, logits[pos_tok], logits[neg_tok], logits[target_tok]);

    llama_batch_free(batch);
}

// ========================================
// R-AdaZO fine-tuning loop
// ========================================

static void finetune_radazo_quant(
        struct llama_context * ctx,
        const std::vector<sft_sample> & train_samples,
        const std::vector<sft_sample> & eval_samples,
        RAdaZOOptimizer & optimizer,
        LoRAAdapter & lora_adapter,
        int n_epochs,
        int train_batch_size) {
    (void) lora_adapter;

    LOG_INF("\n%s: starting R-AdaZO fine-tuning (QLoRA A/B only)...\n", __func__);

    const int n_ctx = llama_n_ctx(ctx);
    const int n_seq_max = (int) llama_n_seq_max(ctx);
    const int n_ctx_per_seq = std::max(1, n_ctx / std::max(1, n_seq_max));
    const int token_budget = std::min(n_ctx, (int) llama_n_batch(ctx));
    const struct llama_model * model_ptr = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ptr));

    LOG_INF("%s: n_ctx=%d, n_seq_max=%d, n_ctx_per_seq=%d, token_budget=%d, train_batch_size=%d, n_train_samples=%zu, n_epochs=%d\n",
            __func__, n_ctx, n_seq_max, n_ctx_per_seq, token_budget, train_batch_size, train_samples.size(), n_epochs);

    cpu_usage_tracker cpu_tracker;
    const int64_t iters_per_epoch = estimate_packed_batch_count(
            train_samples,
            token_budget,
            n_ctx_per_seq,
            std::min(train_batch_size, n_seq_max));

    // Native llama_adapter_lora is already registered in lora_adapter.initialize().
    // llama_decode() automatically applies LoRA via the compute graph.
    // Base tensors are never modified during training.

    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        LOG_INF("\n%s: ===== Epoch %d/%d =====\n", __func__, epoch + 1, n_epochs);
        append_memory_log_train("epoch_begin", epoch + 1, 0, 0);

        const int64_t t_epoch_start = ggml_time_us();
        int64_t n_done = 0;
        int64_t n_samples_done = 0;
        float epoch_loss_sum = 0.0f;

        for (size_t si = 0; si < train_samples.size();) {
            llama_batch batch = {};
            std::vector<radazo_loss_target> loss_targets;
            size_t next_si = si;
            int packed_samples = 0;

            if (!build_packed_batch(
                    train_samples,
                    si,
                    token_budget,
                    n_ctx_per_seq,
                    std::min(train_batch_size, n_seq_max),
                    batch,
                    loss_targets,
                    next_si,
                    packed_samples)) {
                LOG_ERR("%s: failed to build packed batch at sample %zu\n", __func__, si);
                break;
            }

            // Per-step independent loss returned by optimizer for this packed batch.
            float sample_loss = optimizer.step(ctx, batch, n_vocab, loss_targets);

            epoch_loss_sum += sample_loss;
            n_done++;
            n_samples_done += packed_samples;

            // Show current step loss (not running average).
            progress_callback(true, n_done, iters_per_epoch,
                sample_loss, t_epoch_start, cpu_tracker);

            llama_batch_free(batch);
            si = next_si;
        }

        fprintf(stderr, "\n");
        LOG_INF("%s: Epoch %d complete - Avg Loss: %.6f, batches=%ld, samples=%ld\n",
                __func__, epoch + 1, n_done > 0 ? epoch_loss_sum / n_done : 0.0f, n_done, n_samples_done);
        if (!eval_samples.empty()) {
            const float acc = evaluate_sst2_accuracy(ctx, eval_samples, n_ctx);
            LOG_INF("%s: Epoch %d validation accuracy: %.2f%%\n", __func__, epoch + 1, acc * 100.0f);
        }
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

    // Default SST-2 dataset paths (try multiple locations)
    std::string sst2_train_file;
    std::vector<std::string> possible_train_paths = {
        "examples/zeroth-order-opt/sst2_train.json",
        "../examples/zeroth-order-opt/sst2_train.json",
        "../../examples/zeroth-order-opt/sst2_train.json",
        "../../../examples/zeroth-order-opt/sst2_train.json"
    };
    for (const auto & path : possible_train_paths) {
        std::ifstream test_file(path);
        if (test_file.good()) {
            sst2_train_file = path;
            break;
        }
    }
    if (sst2_train_file.empty()) {
        sst2_train_file = possible_train_paths[0];
    }

    std::string sst2_validation_file;
    std::vector<std::string> possible_validation_paths = {
        "examples/zeroth-order-opt/sst2_test.json",
        "../examples/zeroth-order-opt/sst2_test.json",
        "../../examples/zeroth-order-opt/sst2_test.json",
        "../../../examples/zeroth-order-opt/sst2_test.json"
    };
    for (const auto & path : possible_validation_paths) {
        std::ifstream test_file(path);
        if (test_file.good()) {
            sst2_validation_file = path;
            break;
        }
    }
    if (sst2_validation_file.empty()) {
        sst2_validation_file = possible_validation_paths[0];
    }

    bool save_lora_only = false;
    {
        int n = 0;
        for (int i = 0; i < argc; ++i) {
            if (std::strcmp(argv[i], "--save-lora-only") == 0) {
                save_lora_only = true;
                continue;
            }
            if (std::strcmp(argv[i], "--train-batch-size") == 0 && i + 1 < argc) {
                int32_t parsed = 0;
                if (parse_int32_arg(argv[i + 1], parsed) && parsed > 0) {
                    train_config.train_batch_size = parsed;
                } else {
                    LOG_WRN("%s: invalid --train-batch-size value '%s', keeping %d\n",
                            __func__, argv[i + 1], train_config.train_batch_size);
                }
                ++i;
                continue;
            }
            if (std::strncmp(argv[i], "--train-batch-size=", 19) == 0) {
                int32_t parsed = 0;
                const char * value = argv[i] + 19;
                if (parse_int32_arg(value, parsed) && parsed > 0) {
                    train_config.train_batch_size = parsed;
                } else {
                    LOG_WRN("%s: invalid --train-batch-size value '%s', keeping %d\n",
                            __func__, value, train_config.train_batch_size);
                }
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

    // Packed SST-2 training uses multiple independent seq_id streams inside one
    // llama_batch, so the context must allow at least train_batch_size sequences.
    {
        const int32_t effective_parallel = std::max<int32_t>(1, train_config.train_batch_size);
        const int32_t per_seq_ctx_target = std::min<int32_t>(
                std::max<int32_t>(2, train_config.max_tokens_per_sample),
                std::max<int32_t>(2, params.n_ctx));
        params.n_parallel = std::max<int32_t>(params.n_parallel, effective_parallel);
        const int32_t packed_token_budget = params.n_parallel * per_seq_ctx_target;
        params.n_ctx = std::max<int32_t>(params.n_ctx, packed_token_budget);
        params.n_batch = std::max<int32_t>(params.n_batch, packed_token_budget);
        params.n_ubatch = std::max<int32_t>(params.n_ubatch, packed_token_budget);

        LOG_INF("%s: packed training config -> train_batch_size=%d, n_parallel=%d, n_ctx=%d, n_batch=%d, n_ubatch=%d\n",
                __func__,
                train_config.train_batch_size,
                params.n_parallel,
                params.n_ctx,
                params.n_batch,
                params.n_ubatch);
    }

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
    train_config.max_train_samples = 100;
    train_config.max_eval_samples = 100;
    train_config.max_tokens_per_sample = 512;

    // Load SST-2 train/validation datasets
    LOG_INF("%s: Loading SST-2 train dataset from: %s\n", __func__, sst2_train_file.c_str());
    auto sst2_train_examples = load_sst2_json_array_stream(sst2_train_file, train_config.max_train_samples);

    if (sst2_train_examples.empty()) {
        LOG_ERR("%s: failed to load SST-2 train examples\n", __func__);
        return 1;
    }

    LOG_INF("%s: Loading SST-2 validation dataset from: %s\n", __func__, sst2_validation_file.c_str());
    auto sst2_validation_examples = load_sst2_json_array_stream(sst2_validation_file, train_config.max_eval_samples);
    LOG_INF("%s: Loaded SST-2 examples - train=%zu, validation=%zu\n",
            __func__, sst2_train_examples.size(), sst2_validation_examples.size());

    // Show first example
    if (!sst2_train_examples.empty()) {
        LOG_INF("%s: First training example:\n", __func__);
        LOG_INF("%s: ----------------------------------------\n", __func__);
        LOG_INF("%s: %s\n", __func__, format_sst2_prompt(sst2_train_examples[0]).c_str());
        LOG_INF("%s: Label: %s\n", __func__, sst2_train_examples[0].label_str.c_str());
        LOG_INF("%s: ----------------------------------------\n\n", __func__);
    }

    // Tokenize SST-2 train examples as independent sequences (no packing).
    std::vector<sft_sample> train_set = sst2_to_sft_sequences(
        ctx.get(), sst2_train_examples, train_config.max_tokens_per_sample);
    train_set = filter_short_samples(train_set);
    if (train_set.empty()) {
        LOG_ERR("%s: tokenized SST-2 train dataset is empty\n", __func__);
        return 1;
    }

    // Tokenize SST-2 validation examples for eval bookkeeping.
    std::vector<sft_sample> eval_set = sst2_to_sft_sequences(
        ctx.get(), sst2_validation_examples, train_config.max_tokens_per_sample);
    eval_set = filter_short_samples(eval_set);

    LOG_INF("%s: train_samples=%zu (%zu tokens), eval_samples=%zu (%zu tokens)\n\n",
            __func__,
            train_set.size(),
            count_total_tokens(train_set),
            eval_set.size(),
            count_total_tokens(eval_set));
    LOG_INF("%s: runtime batching -> train_batch_size=%d, ctx_n_seq_max=%u, ctx_n_batch=%u, ctx_n_ctx=%u\n\n",
            __func__,
            train_config.train_batch_size,
            llama_n_seq_max(ctx.get()),
            llama_n_batch(ctx.get()),
            llama_n_ctx(ctx.get()));

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
    radazo_config.mu       = 1e-3f;
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
    finetune_radazo_quant(ctx.get(), train_set, eval_set, optimizer, lora_adapter, n_epochs, train_config.train_batch_size);

    // Post-training check: inspect logits on one memorized train sample
    // before saving, to verify train-time behavior inside this process.
    run_sst2_train_sanity_check(ctx.get(), train_set, llama_n_ctx(ctx.get()));

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
    LOG_INF("%s: Dataset: SST-2 (binary sentiment classification)\n", __func__);
    LOG_INF("%s: Loaded train examples: %zu\n", __func__, sst2_train_examples.size());
    LOG_INF("%s: Loaded validation examples: %zu\n", __func__, sst2_validation_examples.size());
    LOG_INF("%s: Epochs: %d\n", __func__, n_epochs);
    LOG_INF("%s: Train batch size: %d\n", __func__, train_config.train_batch_size);
    LOG_INF("%s: Context parallel slots: %u\n", __func__, llama_n_seq_max(ctx.get()));
    LOG_INF("%s: Training samples: %zu\n", __func__, train_set.size());
    LOG_INF("%s: Training tokens: %zu\n", __func__, count_total_tokens(train_set));
    LOG_INF("%s: Output model: %s\n", __func__, output_file.c_str());
    LOG_INF("\n%s: How it works (Global-SPSA QLoRA on quantized models):\n", __func__);
    LOG_INF("%s:   1. Native llama_adapter_lora registered at startup\n", __func__);
    LOG_INF("%s:   2. Base GGUF tensors are read-only throughout training\n", __func__);
    LOG_INF("%s:   3. llama_decode() auto-computes Y = W_base*X + (a/r)*B*A*X\n", __func__);
    LOG_INF("%s:   4. Samples are packed into one llama_batch with distinct seq_id values\n", __func__);
    LOG_INF("%s:      → sparse logits only at each sample's target position\n", __func__);
    LOG_INF("%s:   5. Global SPSA perturbs ALL LoRA A/B once per packed batch\n", __func__);
    LOG_INF("%s:      → 2 forward passes per update step (constant, not per-param)\n", __func__);
    LOG_INF("%s:   6. Adam update applied to all LoRA A/B in one step\n", __func__);
    LOG_INF("%s:   7. Final GGUF saved by merging trained LoRA into base once\n", __func__);

    llama_backend_free();

    return 0;
}
