// Zeroth-Order Optimizer Implementation

#include "zeroth-order-optimizer.h"
#include "log.h"
#include "ggml-backend.h"

#include <cmath>
#include <algorithm>
#include <string>

// Constructor
ZerothOrderOptimizer::ZerothOrderOptimizer(
    const zeroth_order_params & params,
    const std::vector<struct ggml_tensor *> & trainable_params)
    : params_(params)
    , trainable_params_(trainable_params)
    , rng_(params.random_seed)
    , total_updates(0)
    , forward_passes(0) {
}

// Compute loss (L2 norm of logits)
float ZerothOrderOptimizer::compute_loss(float * logits, int n_vocab) {
    float loss = 0.0f;
    const int n_logits = std::min(1000, n_vocab);
    
    for (int v = 0; v < n_logits; ++v) {
        loss += logits[v] * logits[v];
    }
    
    return std::sqrt(loss);
}

// Update a single parameter element with gradient estimation
float ZerothOrderOptimizer::update_parameter_element(
    struct llama_context * ctx,
    llama_batch & batch,
    struct ggml_tensor * param,
    int64_t elem_idx,
    float loss_base,
    int n_vocab) {
    
    // Read current value
    float current_val = 0.0f;
    ggml_backend_tensor_get(param, &current_val, elem_idx * sizeof(float), sizeof(float));
    
    // Perturb by +epsilon
    float perturbed_val = current_val + params_.epsilon;
    ggml_backend_tensor_set(param, &perturbed_val, elem_idx * sizeof(float), sizeof(float));
    
    // Synchronize to ensure backend sees the update
    llama_synchronize(ctx);
    
    // Clear KV cache to allow reprocessing the same batch
    llama_memory_clear(llama_get_memory(ctx), true);
    
    // Forward pass with perturbed parameter
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: forward pass failed, restoring parameter\n", __func__);
        // Restore original value
        ggml_backend_tensor_set(param, &current_val, elem_idx * sizeof(float), sizeof(float));
        llama_memory_clear(llama_get_memory(ctx), true);
        return 0.0f;  // No update performed
    }
    
    forward_passes++;
    
    // Compute loss with perturbed parameter
    float * logits_perturbed = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    float loss_perturbed = compute_loss(logits_perturbed, n_vocab);
    
    // Estimate gradient via finite difference
    float estimated_gradient = (loss_perturbed - loss_base) / params_.epsilon;
    
    // Update with weight decay (L2 regularization)
    float update = params_.learning_rate * (estimated_gradient + params_.weight_decay * current_val);
    float new_val = current_val - update;
    
    // Write back updated value
    ggml_backend_tensor_set(param, &new_val, elem_idx * sizeof(float), sizeof(float));
    llama_synchronize(ctx);
    
    total_updates++;
    
    return estimated_gradient;
}

// Perform one optimization step
float ZerothOrderOptimizer::step(
    struct llama_context * ctx,
    llama_batch & batch,
    int n_vocab) {
    
    if (trainable_params_.empty()) {
        LOG_ERR("%s: no trainable parameters!\n", __func__);
        return 0.0f;
    }
    
    // Compute baseline loss (this should already be done by caller, but we do it here for safety)
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
            
            float gradient = update_parameter_element(
                ctx, batch, param, elem_idx, loss_base, n_vocab);
            
            // Optional: Log gradient information
            if (params_.log_gradients && e == 0 && p == 0) {
                float current_val;
                ggml_backend_tensor_get(param, &current_val, elem_idx * sizeof(float), sizeof(float));
                LOG_INF("  [Optimizer] Gradient: loss_base=%.6f, grad=%.6e, val=%.6f\n",
                        loss_base, gradient, current_val);
            }
        }
    }
    
    return loss_base;
}

// Helper function: Collect trainable parameters from model
std::vector<struct ggml_tensor *> collect_trainable_parameters(
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
        // Focus on attention (Q/K/V/O) and FFN (gate/up/down) weight matrices
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

