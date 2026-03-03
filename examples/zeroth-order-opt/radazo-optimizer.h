// R-AdaZO Optimizer for GGUF Models (FP32 and Quantized)
// Refining Adaptive Zeroth-Order Optimization
// Paper: "Refining Adaptive Zeroth-Order Optimization at Ease" (arXiv:2502.01014)
//
// This optimizer transparently handles both FP32 and quantized (Q4_K_M, Q8_0, etc.)
// tensors by dequantizing before perturbation and requantizing after updates.

#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <random>
#include <unordered_map>
#include <string>
#include <functional>

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
// Works with both FP32 and quantized model tensors
class RAdaZOOptimizer {
public:
    using logits_postprocessor_t = std::function<void(struct llama_context *, float *, int)>;

    // Constructor
    RAdaZOOptimizer(
        const radazo_params & params,
        const std::vector<struct ggml_tensor *> & trainable_params);
    
    // Perform one optimization step on a batch
    // Returns the baseline loss
    float step(
        struct llama_context * ctx,
        llama_batch & batch,
        int n_vocab,
        llama_token target_token = -1);
    
    // Get statistics
    int64_t get_total_updates() const { return total_updates; }
    int64_t get_forward_passes() const { return forward_passes; }

    // Optional hook to modify logits before loss evaluation (e.g. LoRA adapters).
    void set_logits_postprocessor(logits_postprocessor_t hook) { logits_postprocessor_ = std::move(hook); }
    
private:
    // Compute loss from logits
    float compute_loss(float * logits, int n_vocab, llama_token target_token);
    
    // Fill a normalized random perturbation (unit norm) in-place
    void fill_perturbation(
        std::vector<float> & out,
        int64_t n_elements,
        uint32_t seed);
    
    // Estimate gradient using multiple samples (handles FP32 and quantized)
    void estimate_tensor_gradient_radazo(
        struct llama_context * ctx,
        llama_batch & batch,
        struct ggml_tensor * param,
        float loss_base,
        int n_vocab,
        llama_token target_token,
        std::vector<float> & grad_est);
    
    // Update parameter using Adam-style adaptive learning rate (handles FP32 and quantized)
    void update_parameter_adam(
        struct llama_context * ctx,
        struct ggml_tensor * param,
        const std::vector<float> & gradient);
    
    // Get or create state for a parameter
    radazo_param_state & get_state(struct ggml_tensor * param);

    // Compute loss after optional logits post-processing.
    float compute_loss_with_postprocess(
        struct llama_context * ctx,
        float * logits,
        int n_vocab,
        llama_token target_token);

    // Drop large intermediate buffers once a parameter step finishes.
    void clear_intermediate_buffers();

    // Append CPU/GPU memory usage for each optimizer stage.
    void log_memory_checkpoint(
        const char * stage,
        int32_t param_idx = -1,
        int32_t sample_idx = -1,
        const char * extra = nullptr);
    
    // Parameters
    radazo_params params_;
    std::vector<struct ggml_tensor *> trainable_params_;
    
    // Per-parameter state (maps tensor pointer to state)
    std::unordered_map<struct ggml_tensor *, radazo_param_state> param_states_;

    // Reused scratch buffers to reduce allocation churn and peak RSS.
    std::vector<float> scratch_grad_est_;
    std::vector<float> scratch_param_snapshot_;
    std::vector<float> scratch_perturbed_values_;
    std::vector<float> scratch_direction_;
    std::vector<float> scratch_new_values_;
    std::vector<uint8_t> scratch_quant_snapshot_;
    std::vector<uint8_t> scratch_quant_perturbed_;
    std::vector<uint8_t> scratch_quant_updated_;
    std::vector<float> scratch_logits_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Statistics
    int64_t total_updates;
    int64_t forward_passes;
    int64_t global_step;
    logits_postprocessor_t logits_postprocessor_;
};

// Helper function: Collect trainable parameters from model
// Works with both FP32 and quantized models
std::vector<struct ggml_tensor *> collect_trainable_parameters_radazo(
    struct llama_context * ctx,
    bool verbose = true);
