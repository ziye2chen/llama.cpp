// R-AdaZO Optimizer Implementation
// Based on "Refining Adaptive Zeroth-Order Optimization at Ease" (arXiv:2502.01014)

#include "radazo-optimizer.h"
#include "log.h"
#include "ggml-backend.h"

#include <cmath>
#include <algorithm>
#include <string>
#include <numeric>

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

// Compute loss (L2 norm of logits)
float RAdaZOOptimizer::compute_loss(float * logits, int n_vocab) {
    float loss = 0.0f;
    const int n_logits = std::min(1000, n_vocab);
    
    for (int v = 0; v < n_logits; ++v) {
        loss += logits[v] * logits[v];
    }
    
    return std::sqrt(loss);
}

// Generate normalized random perturbation (unit norm)
std::vector<float> RAdaZOOptimizer::get_perturbation(int64_t n_elements, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> u(n_elements);
    float norm = 0.0f;
    
    // Generate random Gaussian vector
    for (int64_t i = 0; i < n_elements; ++i) {
        u[i] = dist(gen);
        norm += u[i] * u[i];
    }
    
    // Normalize to unit sphere
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (int64_t i = 0; i < n_elements; ++i) {
            u[i] /= norm;
        }
    }
    
    return u;
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

// Estimate gradient using R-AdaZO method with multiple samples
float RAdaZOOptimizer::estimate_gradient_radazo(
    struct llama_context * ctx,
    llama_batch & batch,
    struct ggml_tensor * param,
    int64_t elem_idx,
    float loss_base,
    int n_vocab,
    std::vector<float> & grad_est) {
    
    // We'll average gradient estimates from n_samples perturbations
    grad_est.clear();
    grad_est.resize(1, 0.0f);  // For single element
    
    float current_val;
    ggml_backend_tensor_get(param, &current_val, elem_idx * sizeof(float), sizeof(float));
    
    // Sample multiple perturbations and average their gradient estimates
    for (int32_t s = 0; s < params_.n_samples; ++s) {
        // Generate random seed for this sample
        uint32_t sample_seed = rng_();
        
        // For single element, perturbation is just +1 or -1 (normalized)
        // Generate random direction
        std::uniform_real_distribution<float> unif(-1.0f, 1.0f);
        float direction = unif(rng_) > 0.0f ? 1.0f : -1.0f;
        
        // Perturb parameter: x + mu * u
        float perturbed_val = current_val + params_.mu * direction;
        ggml_backend_tensor_set(param, &perturbed_val, elem_idx * sizeof(float), sizeof(float));
        llama_synchronize(ctx);
        
        // Clear KV cache and do forward pass with perturbed parameter
        llama_memory_clear(llama_get_memory(ctx), true);
        
        if (llama_decode(ctx, batch) != 0) {
            // Restore original value on error
            ggml_backend_tensor_set(param, &current_val, elem_idx * sizeof(float), sizeof(float));
            llama_memory_clear(llama_get_memory(ctx), true);
            continue;
        }
        
        forward_passes++;
        
        // Compute loss with perturbed parameter: f(x + mu*u)
        float * logits_perturbed = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        float loss_plus = compute_loss(logits_perturbed, n_vocab);
        
        // Gradient estimate: (f(x + mu*u) - f(x)) * u / mu
        float sample_gradient = (loss_plus - loss_base) * direction / params_.mu;
        grad_est[0] += sample_gradient;
    }
    
    // Average over samples
    grad_est[0] /= params_.n_samples;
    
    // Restore original value
    ggml_backend_tensor_set(param, &current_val, elem_idx * sizeof(float), sizeof(float));
    llama_synchronize(ctx);
    
    return grad_est[0];
}

// Update parameter using Adam-style adaptive learning rate (R-AdaZO variant)
void RAdaZOOptimizer::update_parameter_adam(
    struct ggml_tensor * param,
    int64_t elem_idx,
    float gradient) {
    
    // Get state for this parameter
    radazo_param_state & state = get_state(param);
    state.step++;
    
    // Adam-style updates with R-AdaZO modification
    float & m = state.exp_avg[elem_idx];       // First moment (momentum)
    float & v = state.exp_avg_sq[elem_idx];    // Second moment (adaptive lr)
    
    // Update first moment estimate: m = beta1 * m + (1 - beta1) * g
    m = params_.beta1 * m + (1.0f - params_.beta1) * gradient;
    
    // *** KEY R-AdaZO DIFFERENCE ***
    // Standard Adam/ZO-Adam: v = beta2 * v + (1 - beta2) * g^2
    // R-AdaZO: v = beta2 * v + (1 - beta2) * m^2
    // This uses the momentum-smoothed gradient, reducing variance!
    v = params_.beta2 * v + (1.0f - params_.beta2) * m * m;
    
    // Compute bias-corrected moments (optional, but recommended)
    float m_hat = m / (1.0f - std::pow(params_.beta1, state.step));
    float v_hat = v / (1.0f - std::pow(params_.beta2, state.step));
    
    // Compute parameter update
    float denom = std::sqrt(v_hat) + params_.eps;
    float update = params_.lr * m_hat / denom;
    
    // Read current parameter value
    float current_val;
    ggml_backend_tensor_get(param, &current_val, elem_idx * sizeof(float), sizeof(float));
    
    // Apply update
    float new_val = current_val - update;
    
    // Write back
    ggml_backend_tensor_set(param, &new_val, elem_idx * sizeof(float), sizeof(float));
    
    total_updates++;
}

// Perform one optimization step
float RAdaZOOptimizer::step(
    struct llama_context * ctx,
    llama_batch & batch,
    int n_vocab) {
    
    if (trainable_params_.empty()) {
        LOG_ERR("%s: no trainable parameters!\n", __func__);
        return 0.0f;
    }
    
    global_step++;
    
    // Compute baseline loss
    float * logits_base = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    float loss_base = compute_loss(logits_base, n_vocab);
    forward_passes++;
    
    // Random parameter sampling
    std::uniform_int_distribution<size_t> param_dist(0, trainable_params_.size() - 1);
    
    // Sample and update parameters
    for (int32_t p = 0; p < params_.n_params_per_iter && p < (int32_t)trainable_params_.size(); ++p) {
        size_t param_idx = param_dist(rng_);
        struct ggml_tensor * param = trainable_params_[param_idx];
        
        const int64_t n_elements = ggml_nelements(param);
        if (n_elements == 0) continue;
        
        // Check buffer validity
        if (param->buffer == nullptr) {
            LOG_ERR("%s: parameter has no buffer, skipping\n", __func__);
            continue;
        }
        
        // Sample random elements within this parameter
        std::uniform_int_distribution<int64_t> elem_dist(0, n_elements - 1);
        
        for (int32_t e = 0; e < params_.n_elements_per_param; ++e) {
            int64_t elem_idx = elem_dist(rng_);
            
            // Estimate gradient using R-AdaZO (multiple samples)
            std::vector<float> grad_est;
            float gradient = estimate_gradient_radazo(
                ctx, batch, param, elem_idx, loss_base, n_vocab, grad_est);
            
            // Update parameter using Adam-style adaptive learning rate
            update_parameter_adam(param, elem_idx, gradient);
            
            // Optional: Log gradient information
            if (params_.log_gradients && e == 0 && p == 0) {
                radazo_param_state & state = get_state(param);
                LOG_INF("  [R-AdaZO] Step %lld: loss_base=%.6f, grad=%.6e, m=%.6e, v=%.6e\n",
                        global_step, loss_base, gradient, 
                        state.exp_avg[elem_idx], state.exp_avg_sq[elem_idx]);
            }
        }
    }
    
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
                    LOG_INF("%s:   [%3zu] %s (shape: [%lld, %lld], %lld elements)\n",
                            __func__, params.size(), tensor_name,
                            tensor->ne[0], tensor->ne[1], ggml_nelements(tensor));
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

