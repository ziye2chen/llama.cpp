// R-AdaZO Optimizer for GGUF Models
// Refining Adaptive Zeroth-Order Optimization
// Paper: "Refining Adaptive Zeroth-Order Optimization at Ease" (arXiv:2502.01014)

#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <random>
#include <unordered_map>

// R-AdaZO optimization parameters
struct radazo_params {
    float lr = 1e-3f;                    // learning rate
    float beta1 = 0.9f;                  // exponential decay for first moment
    float beta2 = 0.999f;                // exponential decay for second moment
    float eps = 1e-8f;                   // numerical stability constant
    float mu = 5e-3f;                    // smoothing parameter (perturbation magnitude)
    int32_t n_samples = 2;               // number of random samples per gradient estimation
    int32_t n_params_per_iter = 3;       // parameters to sample per iteration
    bool full_tensor_gradient = true;    // if true, estimate gradient for entire tensor
    uint32_t random_seed = 42;
    bool log_gradients = false;          // verbose gradient logging
};

// Per-parameter state for Adam-style momentum
struct radazo_param_state {
    std::vector<float> exp_avg;       // First moment estimate (momentum)
    std::vector<float> exp_avg_sq;    // Second moment estimate (adaptive lr)
    int64_t step = 0;                 // Number of updates for this parameter
};

// R-AdaZO optimizer class
class RAdaZOOptimizer {
public:
    // Constructor
    RAdaZOOptimizer(
        const radazo_params & params,
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
    // Compute loss from logits
    float compute_loss(float * logits, int n_vocab);
    
    // Generate normalized random perturbation
    std::vector<float> get_perturbation(int64_t n_elements, uint32_t seed);
    
    // Estimate gradient using multiple samples
    void estimate_tensor_gradient_radazo(
        struct llama_context * ctx,
        llama_batch & batch,
        struct ggml_tensor * param,
        float loss_base,
        int n_vocab,
        std::vector<float> & grad_est,
        std::vector<float> & param_snapshot);
    
    // Update parameter using Adam-style adaptive learning rate
    void update_parameter_adam(
        struct llama_context * ctx,
        struct ggml_tensor * param,
        const std::vector<float> & gradient,
        const std::vector<float> & param_snapshot);
    
    // Get or create state for a parameter
    radazo_param_state & get_state(struct ggml_tensor * param);
    
    // Parameters
    radazo_params params_;
    std::vector<struct ggml_tensor *> trainable_params_;
    
    // Per-parameter state (maps tensor pointer to state)
    std::unordered_map<struct ggml_tensor *, radazo_param_state> param_states_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Statistics
    int64_t total_updates;
    int64_t forward_passes;
    int64_t global_step;
};

// Helper function: Collect trainable parameters from model
// (Same as zeroth-order-optimizer.h, included for completeness)
std::vector<struct ggml_tensor *> collect_trainable_parameters_radazo(
    struct llama_context * ctx,
    bool verbose = true);

