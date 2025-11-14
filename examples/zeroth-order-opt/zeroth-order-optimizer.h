// Zeroth-Order Optimizer for GGUF Models
// Plug-and-play gradient-free optimization component

#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <random>

// Zeroth-order optimization parameters
struct zeroth_order_params {
    float epsilon = 1e-3f;              // perturbation size for finite differences
    float learning_rate = 1e-4f;        // learning rate
    float weight_decay = 0.01f;         // L2 regularization
    int32_t n_params_per_iter = 3;      // parameters to sample per iteration
    int32_t n_elements_per_param = 2;   // elements to update per parameter
    bool use_random_sampling = true;
    uint32_t random_seed = 42;
    bool log_gradients = false;         // verbose gradient logging
};

// Zeroth-order optimizer class
class ZerothOrderOptimizer {
public:
    // Constructor
    ZerothOrderOptimizer(
        const zeroth_order_params & params,
        const std::vector<struct ggml_tensor *> & trainable_params);
    
    // Perform one optimization step on a batch
    // Returns the baseline loss
    float step(
        struct llama_context * ctx,
        llama_batch & batch,
        int n_vocab);
    
    // Get statistics
    int64_t get_total_updates() const { return total_updates; }
    int64_t get_forward_passes() const { return forward_passes; }
    
private:
    // Compute loss from logits (L2 norm of first 1000 logits)
    float compute_loss(float * logits, int n_vocab);
    
    // Estimate gradient and update a single parameter element
    float update_parameter_element(
        struct llama_context * ctx,
        llama_batch & batch,
        struct ggml_tensor * param,
        int64_t elem_idx,
        float loss_base,
        int n_vocab);
    
    // Parameters
    zeroth_order_params params_;
    std::vector<struct ggml_tensor *> trainable_params_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Statistics
    int64_t total_updates;
    int64_t forward_passes;
};

// Helper function: Collect trainable parameters from model
std::vector<struct ggml_tensor *> collect_trainable_parameters(
    struct llama_context * ctx,
    bool verbose = true);

