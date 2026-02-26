// R-AdaZO Optimizer Implementation
// Based on "Refining Adaptive Zeroth-Order Optimization at Ease" (arXiv:2502.01014)
// Supports both FP32 and quantized (Q4_K_M, Q8_0, etc.) tensor types

#include "radazo-optimizer.h"
#include "log.h"
#include "ggml-backend.h"

#include <cmath>
#include <algorithm>
#include <string>
#include <numeric>
#include <limits>
#include <fstream>
#include <sstream>
#include <iomanip>

static constexpr const char * k_memory_log_path = "memory.txt";

static float get_cpu_memory_mb_optimizer() {
#if defined(__linux__)
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            float mem_kb = 0.0f;
            iss >> mem_kb;
            return mem_kb / 1024.0f;
        }
    }
#endif
    return 0.0f;
}

static bool get_gpu_memory_mb_optimizer(float & free_mb, float & total_mb) {
    free_mb = 0.0f;
    total_mb = 0.0f;

    const size_t n_devs = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devs; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
        if (dev_type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dev_type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        size_t dev_free = 0;
        size_t dev_total = 0;
        ggml_backend_dev_memory(dev, &dev_free, &dev_total);
        if (dev_total == 0) {
            continue;
        }

        free_mb += (float) dev_free / (1024.0f * 1024.0f);
        total_mb += (float) dev_total / (1024.0f * 1024.0f);
    }

    return total_mb > 0.0f;
}

// ========================================
// Quantized tensor helper functions
// ========================================

// Dequantize a full tensor from its raw quantized bytes into FP32
static bool dequantize_tensor_to_fp32(
    struct ggml_tensor * param,
    const std::vector<uint8_t> & quant_bytes,
    std::vector<float> & fp32_out) {

    const enum ggml_type qtype = param->type;
    const struct ggml_type_traits * traits = ggml_get_type_traits(qtype);
    if (traits == nullptr || traits->to_float == nullptr) {
        return false;
    }

    const int64_t nrows     = ggml_nrows(param);
    const int64_t n_per_row = param->ne[0];
    const size_t  row_size  = ggml_row_size(qtype, n_per_row);

    fp32_out.resize(ggml_nelements(param));

    for (int64_t r = 0; r < nrows; ++r) {
        const void * src = quant_bytes.data() + r * row_size;
        float      * dst = fp32_out.data()    + r * n_per_row;
        traits->to_float(src, dst, n_per_row);
    }
    return true;
}

// Requantize FP32 values back into raw quantized bytes
static bool requantize_fp32_to_bytes(
    struct ggml_tensor * param,
    const std::vector<float> & fp32_in,
    std::vector<uint8_t> & quant_out) {

    const enum ggml_type qtype = param->type;
    const struct ggml_type_traits * traits = ggml_get_type_traits(qtype);
    if (traits == nullptr || traits->from_float_ref == nullptr) {
        return false;
    }

    const int64_t nrows     = ggml_nrows(param);
    const int64_t n_per_row = param->ne[0];
    const size_t  row_size  = ggml_row_size(qtype, n_per_row);

    quant_out.resize(ggml_nbytes(param));

    for (int64_t r = 0; r < nrows; ++r) {
        const float * src = fp32_in.data()    + r * n_per_row;
        void        * dst = quant_out.data()  + r * row_size;
        traits->from_float_ref(src, dst, n_per_row);
    }
    return true;
}

// Constructor
RAdaZOOptimizer::RAdaZOOptimizer(
    const radazo_params & params,
    const std::vector<struct ggml_tensor *> & trainable_params)
    : params_(params)
    , trainable_params_(trainable_params)
    , rng_(params.random_seed)
    , total_updates(0)
    , forward_passes(0)
    , global_step(0) {
    
    LOG_INF("%s: Initialized R-AdaZO optimizer\n", __func__);
    LOG_INF("%s:   lr=%.2e, beta1=%.3f, beta2=%.3f, mu=%.2e\n", 
            __func__, params_.lr, params_.beta1, params_.beta2, params_.mu);
    LOG_INF("%s:   n_samples=%d (multiple perturbations per gradient)\n", 
            __func__, params_.n_samples);
}

void RAdaZOOptimizer::log_memory_checkpoint(
    const char * stage,
    int32_t param_idx,
    int32_t sample_idx,
    const char * extra) {
    const float cpu_mb = get_cpu_memory_mb_optimizer();
    float gpu_free_mb = 0.0f;
    float gpu_total_mb = 0.0f;
    const bool gpu_ok = get_gpu_memory_mb_optimizer(gpu_free_mb, gpu_total_mb);
    const float gpu_used_mb = gpu_ok ? (gpu_total_mb - gpu_free_mb) : 0.0f;

    std::ofstream fout(k_memory_log_path, std::ios::app);
    if (!fout.is_open()) {
        return;
    }

    fout << std::fixed << std::setprecision(2)
         << "scope=optimizer"
         << " step=" << global_step
         << " stage=" << stage
         << " param_idx=" << param_idx
         << " sample_idx=" << sample_idx
         << " cpu_mb=" << cpu_mb;

    if (gpu_ok) {
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

// Compute loss from logits.
// If target_token is valid, use SFT-style single-target cross-entropy:
//   loss = -log p(target_token | context)
// Otherwise fallback to legacy L2 norm for compatibility.
float RAdaZOOptimizer::compute_loss(float * logits, int n_vocab, llama_token target_token) {
    if (target_token >= 0 && target_token < n_vocab) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < n_vocab; ++v) {
            max_logit = std::max(max_logit, logits[v]);
        }

        double exp_sum = 0.0;
        for (int v = 0; v < n_vocab; ++v) {
            exp_sum += std::exp((double)logits[v] - (double)max_logit);
        }

        const double logsumexp = (double)max_logit + std::log(exp_sum);
        return (float)(logsumexp - (double)logits[target_token]);
    }

    float loss = 0.0f;
    const int n_logits = std::min(1000, n_vocab);
    for (int v = 0; v < n_logits; ++v) {
        loss += logits[v] * logits[v];
    }
    return std::sqrt(loss);
}

// Generate normalized random perturbation (unit norm)
void RAdaZOOptimizer::fill_perturbation(
    std::vector<float> & out,
    int64_t n_elements,
    uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    out.resize(n_elements);
    float norm = 0.0f;

    // Generate random Gaussian vector
    for (int64_t i = 0; i < n_elements; ++i) {
        out[i] = dist(gen);
        norm += out[i] * out[i];
    }

    // Normalize to unit sphere
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (int64_t i = 0; i < n_elements; ++i) {
            out[i] /= norm;
        }
    }
}

// Get or create state for a parameter
radazo_param_state & RAdaZOOptimizer::get_state(struct ggml_tensor * param) {
    auto it = param_states_.find(param);
    if (it == param_states_.end()) {
        // Initialize state for this parameter
        radazo_param_state state;
        const int64_t n_elem = ggml_nelements(param);
        state.exp_avg.resize(n_elem, 0.0f);
        state.exp_avg_sq.resize(n_elem, 0.0f);
        state.step = 0;
        
        param_states_[param] = state;
        return param_states_[param];
    }
    return it->second;
}

// Estimate gradient for the entire tensor using R-AdaZO method
// Supports both FP32 and quantized tensor types transparently
void RAdaZOOptimizer::estimate_tensor_gradient_radazo(
    struct llama_context * ctx,
    llama_batch & batch,
    struct ggml_tensor * param,
    float loss_base,
    int n_vocab,
    llama_token target_token,
    std::vector<float> & grad_est) {
    
    const int64_t n_elements = ggml_nelements(param);
    log_memory_checkpoint("estimate_begin");
    if (n_elements == 0) {
        grad_est.clear();
        return;
    }
    
    grad_est.resize(n_elements);
    std::fill(grad_est.begin(), grad_est.end(), 0.0f);
    std::vector<float> & param_snapshot = scratch_param_snapshot_;
    param_snapshot.resize(n_elements);
    
    const bool is_quant = ggml_is_quantized(param->type);
    
    // Raw quantized bytes — only used when the tensor is quantized
    std::vector<uint8_t> & quant_snapshot = scratch_quant_snapshot_;
    
    if (is_quant) {
        // ---- Quantized path ----
        // 1. Save original raw quantized bytes (for exact restoration)
        const size_t nbytes = ggml_nbytes(param);
        quant_snapshot.resize(nbytes);
        ggml_backend_tensor_get(param, quant_snapshot.data(), 0, nbytes);
        log_memory_checkpoint("estimate_after_quant_snapshot");

        // 2. Dequantize to FP32 for perturbation arithmetic
        if (!dequantize_tensor_to_fp32(param, quant_snapshot, param_snapshot)) {
            LOG_ERR("%s: failed to dequantize tensor %s (type %s)\n",
                    __func__, ggml_get_name(param), ggml_type_name(param->type));
            grad_est.clear();
            return;
        }
        log_memory_checkpoint("estimate_after_dequant");
    } else {
        // ---- FP32 path ----
        ggml_backend_tensor_get(param, param_snapshot.data(), 0, n_elements * sizeof(float));
        log_memory_checkpoint("estimate_after_fp32_snapshot");
    }

    std::vector<float> & perturbed_values = scratch_perturbed_values_;
    perturbed_values.resize(n_elements);
    std::vector<float> & direction = scratch_direction_;
    std::vector<uint8_t> & quant_perturbed = scratch_quant_perturbed_;

    // Sample multiple perturbations and average their gradient estimates
    for (int32_t s = 0; s < params_.n_samples; ++s) {
        log_memory_checkpoint("sample_begin", -1, s);
        // Generate a normalized random direction for the full tensor
        fill_perturbation(direction, n_elements, rng_());

        // Apply perturbation in FP32 space: x + mu * u
        for (int64_t i = 0; i < n_elements; ++i) {
            perturbed_values[i] = param_snapshot[i] + params_.mu * direction[i];
        }

        // Write perturbed values back into the tensor
        if (is_quant) {
            // Re-quantize perturbed FP32 values and set raw bytes
            if (!requantize_fp32_to_bytes(param, perturbed_values, quant_perturbed)) {
                LOG_ERR("%s: failed to requantize perturbed tensor\n", __func__);
                continue;
            }
            ggml_backend_tensor_set(param, quant_perturbed.data(), 0, quant_perturbed.size());
        } else {
            ggml_backend_tensor_set(param, perturbed_values.data(), 0, n_elements * sizeof(float));
        }
        log_memory_checkpoint("sample_after_perturb_write", -1, s);

        // Clear KV cache and do forward pass with perturbed parameter
        llama_memory_clear(llama_get_memory(ctx), true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed during tensor perturbation, restoring parameter\n", __func__);
            // Restore original
            if (is_quant) {
                ggml_backend_tensor_set(param, quant_snapshot.data(), 0, quant_snapshot.size());
            } else {
                ggml_backend_tensor_set(param, param_snapshot.data(), 0, n_elements * sizeof(float));
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            continue;
        }
        log_memory_checkpoint("sample_after_decode", -1, s);

        forward_passes++;

        // Compute loss with perturbed parameter: f(x + mu*u)
        float * logits_perturbed = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        float loss_plus = compute_loss(logits_perturbed, n_vocab, target_token);

        // Gradient estimate contribution: (f(x + mu*u) - f(x)) * u / mu
        const float scale = (loss_plus - loss_base) / params_.mu;
        for (int64_t i = 0; i < n_elements; ++i) {
            grad_est[i] += scale * direction[i];
        }

        // Restore original parameter values before next sample
        if (is_quant) {
            ggml_backend_tensor_set(param, quant_snapshot.data(), 0, quant_snapshot.size());
        } else {
            ggml_backend_tensor_set(param, param_snapshot.data(), 0, n_elements * sizeof(float));
        }
        log_memory_checkpoint("sample_after_restore", -1, s);
    }

    if (params_.n_samples > 0) {
        const float inv_samples = 1.0f / params_.n_samples;
        for (int64_t i = 0; i < n_elements; ++i) {
            grad_est[i] *= inv_samples;
        }
    }
    log_memory_checkpoint("estimate_end");
}

// Update parameter using Adam-style adaptive learning rate (R-AdaZO variant)
// Supports both FP32 and quantized tensor types transparently
void RAdaZOOptimizer::update_parameter_adam(
    struct llama_context * /* ctx */,
    struct ggml_tensor * param,
    const std::vector<float> & gradient) {
    
    const int64_t n_elements = ggml_nelements(param);
    log_memory_checkpoint("update_begin");
    if ((int64_t)gradient.size() != n_elements) {
        LOG_ERR("%s: gradient/state size mismatch for tensor update\n", __func__);
        return;
    }
    
    // Get state for this parameter
    radazo_param_state & state = get_state(param);
    state.step++;
    
    const float bias_correction1 = 1.0f - std::pow(params_.beta1, state.step);
    const float bias_correction2 = 1.0f - std::pow(params_.beta2, state.step);
    const float inv_bias1 = bias_correction1 > 0.0f ? 1.0f / bias_correction1 : 1.0f;
    const float inv_bias2 = bias_correction2 > 0.0f ? 1.0f / bias_correction2 : 1.0f;
    
    std::vector<float> & new_values = scratch_new_values_;
    new_values.resize(n_elements);

    // Reload current tensor values (already restored to baseline after perturbations).
    const bool is_quant = ggml_is_quantized(param->type);
    if (is_quant) {
        std::vector<uint8_t> & quant_snapshot = scratch_quant_snapshot_;
        const size_t nbytes = ggml_nbytes(param);
        quant_snapshot.resize(nbytes);
        ggml_backend_tensor_get(param, quant_snapshot.data(), 0, nbytes);
        if (!dequantize_tensor_to_fp32(param, quant_snapshot, new_values)) {
            LOG_ERR("%s: failed to dequantize tensor for update %s\n",
                    __func__, ggml_get_name(param));
            return;
        }
    } else {
        ggml_backend_tensor_get(param, new_values.data(), 0, n_elements * sizeof(float));
    }
    log_memory_checkpoint("update_after_reload");
    
    for (int64_t i = 0; i < n_elements; ++i) {
        float g = gradient[i];
        
        float & m = state.exp_avg[i];       // First moment (momentum)
        float & v = state.exp_avg_sq[i];    // Second moment (adaptive lr)
        
        m = params_.beta1 * m + (1.0f - params_.beta1) * g;
        v = params_.beta2 * v + (1.0f - params_.beta2) * m * m; // R-AdaZO tweak
        
        const float m_hat = m * inv_bias1;
        const float v_hat = v * inv_bias2;
        
        const float denom = std::sqrt(v_hat) + params_.eps;
        const float update = params_.lr * m_hat / denom;
        
        new_values[i] -= update;
    }
    
    if (is_quant) {
        // Re-quantize the updated FP32 values back into the quantized format
        std::vector<uint8_t> & quant_updated = scratch_quant_updated_;
        if (!requantize_fp32_to_bytes(param, new_values, quant_updated)) {
            LOG_ERR("%s: failed to requantize updated tensor %s\n",
                    __func__, ggml_get_name(param));
            return;
        }
        ggml_backend_tensor_set(param, quant_updated.data(), 0, quant_updated.size());
    } else {
        ggml_backend_tensor_set(param, new_values.data(), 0, n_elements * sizeof(float));
    }
    log_memory_checkpoint("update_after_writeback");

    total_updates += n_elements;
}

void RAdaZOOptimizer::clear_intermediate_buffers() {
    std::vector<float>().swap(scratch_grad_est_);
    std::vector<float>().swap(scratch_param_snapshot_);
    std::vector<float>().swap(scratch_perturbed_values_);
    std::vector<float>().swap(scratch_direction_);
    std::vector<float>().swap(scratch_new_values_);
    std::vector<uint8_t>().swap(scratch_quant_snapshot_);
    std::vector<uint8_t>().swap(scratch_quant_perturbed_);
    std::vector<uint8_t>().swap(scratch_quant_updated_);
}

// Perform one optimization step
float RAdaZOOptimizer::step(
    struct llama_context * ctx,
    llama_batch & batch,
    int n_vocab,
    llama_token target_token) {
    
    if (trainable_params_.empty()) {
        LOG_ERR("%s: no trainable parameters!\n", __func__);
        return 0.0f;
    }
    
    global_step++;
    log_memory_checkpoint("optimizer_step_begin");
    
    // Compute baseline loss
    float * logits_base = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    float loss_base = compute_loss(logits_base, n_vocab, target_token);
    forward_passes++;
    
    // Random parameter sampling without replacement to avoid duplicated work
    std::vector<size_t> param_indices(trainable_params_.size());
    std::iota(param_indices.begin(), param_indices.end(), 0);
    std::shuffle(param_indices.begin(), param_indices.end(), rng_);

    const int32_t n_pick = std::min<int32_t>(params_.n_params_per_iter, trainable_params_.size());

    // Sample and update parameters
    for (int32_t p = 0; p < n_pick; ++p) {
        const size_t param_idx = param_indices[p];
        struct ggml_tensor * param = trainable_params_[param_idx];
        
        const int64_t n_elements = ggml_nelements(param);
        if (n_elements == 0) continue;
        log_memory_checkpoint("param_begin", p);
        
        // Check buffer validity
        if (param->buffer == nullptr) {
            LOG_ERR("%s: parameter has no buffer, skipping\n", __func__);
            continue;
        }
        
        if (!params_.full_tensor_gradient) {
            LOG_ERR("%s: full_tensor_gradient is disabled, but per-element updates were removed\n", __func__);
            continue;
        }
        
        // Estimate gradient for the entire tensor using R-AdaZO (multiple samples)
        std::vector<float> & grad_est = scratch_grad_est_;
        estimate_tensor_gradient_radazo(
            ctx, batch, param, loss_base, n_vocab, target_token, grad_est);
        
        if (grad_est.empty()) {
            continue;
        }
        
        // Update parameter using Adam-style adaptive learning rate
        update_parameter_adam(ctx, param, grad_est);
        log_memory_checkpoint("param_end", p);
        
        // Optional: Log gradient information (report gradient norm)
        if (params_.log_gradients && p == 0) {
            double grad_norm = 0.0;
            for (float g : grad_est) {
                grad_norm += static_cast<double>(g) * static_cast<double>(g);
            }
            grad_norm = std::sqrt(grad_norm);
            
            LOG_INF("  [R-AdaZO] Step %ld: loss_base=%.6f, ||grad||=%.6e, elems=%ld\n",
                    global_step, loss_base, grad_norm, n_elements);
        }

        // Release intermediate vectors after each parameter update to keep RSS flatter.
        clear_intermediate_buffers();
        log_memory_checkpoint("param_after_clear", p);
    }
    log_memory_checkpoint("optimizer_step_end");
    
    return loss_base;
}

// Helper function: Collect trainable parameters from model
std::vector<struct ggml_tensor *> collect_trainable_parameters_radazo(
    struct llama_context * ctx,
    bool verbose) {
    
    std::vector<struct ggml_tensor *> params;
    
    if (verbose) {
        LOG_INF("%s: collecting trainable parameters...\n", __func__);
    }
    
    const llama_model * model = llama_get_model(ctx);
    const size_t n_tensors = llama_model_n_tensors(model);
    
    if (verbose) {
        LOG_INF("%s: model has %zu tensors total\n", __func__, n_tensors);
    }
    
    // Iterate through all model tensors and collect trainable weights
    char tensor_name[256];
    for (size_t i = 0; i < n_tensors; ++i) {
        struct ggml_tensor * tensor = llama_model_get_tensor_by_index(
            model, i, tensor_name, sizeof(tensor_name));
        
        if (tensor == nullptr) continue;
        
        std::string name(tensor_name);
        
        // Select key weight matrices for fine-tuning
        if (name.find("attn_q.weight") != std::string::npos ||
            name.find("attn_k.weight") != std::string::npos ||
            name.find("attn_v.weight") != std::string::npos ||
            name.find("attn_output.weight") != std::string::npos ||
            name.find("ffn_gate.weight") != std::string::npos ||
            name.find("ffn_up.weight") != std::string::npos ||
            name.find("ffn_down.weight") != std::string::npos ||
            name.find(".wq") != std::string::npos ||
            name.find(".wk") != std::string::npos ||
            name.find(".wv") != std::string::npos ||
            name.find(".wo") != std::string::npos ||
            name.find(".w1") != std::string::npos ||
            name.find(".w2") != std::string::npos ||
            name.find(".w3") != std::string::npos) {
            
            params.push_back(tensor);
            
            // Log first 10 and last 5 parameters
            if (verbose) {
                if (params.size() <= 10 || params.size() > n_tensors - 5) {
                    LOG_INF("%s:   [%3zu] %s (shape: [%ld, %ld], %ld elements, type: %s)\n",
                            __func__, params.size(), tensor_name,
                            tensor->ne[0], tensor->ne[1], ggml_nelements(tensor),
                            ggml_type_name(tensor->type));
                } else if (params.size() == 11) {
                    LOG_INF("%s:   ... (showing first/last only)\n", __func__);
                }
            }
        }
    }
    
    if (verbose) {
        LOG_INF("%s: collected %zu trainable parameter tensors\n", __func__, params.size());
    }
    
    return params;
}

