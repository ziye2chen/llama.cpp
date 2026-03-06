// R-AdaZO Optimizer  —  Global SPSA + Adam variant
// Paper: "Refining Adaptive Zeroth-Order Optimization at Ease" (arXiv:2502.01014)
//
// Key properties after refactor:
//  • ALL trainable params (LoRA A/B FP32) are perturbed SIMULTANEOUSLY.
//  • Exactly 2 llama_decode() calls per sample (constant, regardless of param count).
//  • No quantised tensor handling needed: LoRA A/B are always FP32.
//  • No param_change_hook: native llama_adapter_lora handles LoRA automatically.

#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <random>
#include <unordered_map>
#include <functional>

struct radazo_loss_target {
    int32_t     batch_token_index = -1;
    llama_token target_token      = -1;
    float       weight            = 1.0f;
};

struct radazo_params {
    float   lr               = 1e-4f;   // Adam learning rate
    float   beta1            = 0.9f;    // first moment decay
    float   beta2            = 0.999f;  // second moment decay
    float   eps              = 1e-8f;   // numerical stability
    float   mu               = 1e-2f;   // perturbation magnitude
    int32_t n_samples        = 1;       // global SPSA draws per step (≥1)
    uint32_t random_seed     = 42;
    bool    log_gradients    = false;

    // Legacy fields kept for config-site compatibility; no longer used internally.
    int32_t n_params_per_iter    = 2;
    bool    full_tensor_gradient = true;
};

struct radazo_param_state {
    std::vector<float> exp_avg;     // first moment
    std::vector<float> exp_avg_sq;  // second moment
    int64_t step = 0;
};

class RAdaZOOptimizer {
public:
    using logits_postprocessor_t = std::function<void(struct llama_context *, float *, int)>;

    RAdaZOOptimizer(
        const radazo_params              & params,
        const std::vector<struct ggml_tensor *> & trainable_params);

    // One optimisation step over a single batch.
    // Returns approximate baseline loss = (loss_+ + loss_-) / 2.
    float step(
        struct llama_context * ctx,
        llama_batch & batch,
        int n_vocab,
        const std::vector<radazo_loss_target> & loss_targets = {});

    int64_t get_total_updates()  const { return total_updates;  }
    int64_t get_forward_passes() const { return forward_passes; }

    void set_logits_postprocessor(logits_postprocessor_t fn) {
        logits_postprocessor_ = std::move(fn);
    }

private:
    // Cross-entropy loss from a single logit vector.
    float compute_loss(float * logits, int n_vocab, llama_token target_token);

    float compute_loss_with_postprocess(
        struct llama_context * ctx, float * logits, int n_vocab, llama_token target);

    float compute_batch_loss_with_masks(
        struct llama_context * ctx,
        int n_vocab,
        const std::vector<radazo_loss_target> & targets);

    // Fill out[] with N(0,1) samples seeded by `seed`.
    void fill_perturbation(std::vector<float> & out, int64_t n_elements, uint32_t seed);

    // Perturb every trainable param with absolute write:
    //   param = master_weight + scale * u_param(seed, param_idx)
    // Phase semantics:
    //   +mu  : apply forward perturbation
    //   -mu  : apply opposite perturbation
    //    0   : restore baseline
    void perturb_all_params(float scale, uint32_t seed);

    // Adam update for all params using gradient estimate  g_i = grad_scale * u_i(seed, i)
    void adam_update_all_params(float grad_scale, uint32_t seed);

    // Adam update for a single FP32 param tensor.
    void update_parameter_adam(
        struct ggml_tensor * param,
        const std::vector<float> & gradient);

    radazo_param_state & get_state(struct ggml_tensor * param);

    // Parameters & trainable tensors
    radazo_params params_;
    std::vector<struct ggml_tensor *> trainable_params_;

    std::unordered_map<struct ggml_tensor *, radazo_param_state> param_states_;
    // CPU master copy for each trainable tensor.
    // We never pull from backend during perturb/update; only push to backend.
    std::unordered_map<struct ggml_tensor *, std::vector<float>> master_weights_;

    // Reused scratch buffers
    std::vector<float>   scratch_direction_;
    std::vector<float>   scratch_param_buf_;
    std::vector<float>   scratch_grad_;
    std::vector<float>   scratch_new_values_;
    std::vector<float>   scratch_logits_;

    std::mt19937 rng_;
    int64_t total_updates  = 0;
    int64_t forward_passes = 0;
    int64_t global_step    = 0;

    logits_postprocessor_t logits_postprocessor_;
};

// Utility: collect trainable tensors from the model.
std::vector<struct ggml_tensor *> collect_trainable_parameters_radazo(
    struct llama_context * ctx,
    bool verbose = true);
