// R-AdaZO Optimizer — Global SPSA + Adam
//
// Each call to step() does exactly:
//   1. Perturb ALL FP32 LoRA A/B simultaneously with +mu*u
//   2. llama_decode  →  loss_+
//   3. Shift ALL params to -mu*u  (apply -2mu*u)
//   4. llama_decode  →  loss_-
//   5. Restore ALL params to baseline  (apply +mu*u)
//   6. grad_scale = (loss_+ - loss_-) / (2*mu)
//   7. Adam update ALL params:  g_i = grad_scale * u_i
//
// Total forward passes per sample: 2  (independent of param count).
// No quantised tensor handling needed: LoRA A/B are always FP32.

#include "radazo-optimizer.h"
#include "log.h"
#include "ggml-backend.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

RAdaZOOptimizer::RAdaZOOptimizer(
        const radazo_params              & params,
        const std::vector<struct ggml_tensor *> & trainable_params)
    : params_(params)
    , trainable_params_(trainable_params)
    , rng_(params.random_seed) {
    // Build one-time CPU master copies for all trainable tensors.
    for (struct ggml_tensor * p : trainable_params_) {
        const int64_t n = ggml_nelements(p);
        std::vector<float> host((size_t) n, 0.0f);
        ggml_backend_tensor_get(p, host.data(), 0, n * sizeof(float));
        master_weights_[p] = std::move(host);
    }
    LOG_INF("%s: Global-SPSA R-AdaZO initialised\n", __func__);
    LOG_INF("%s:   lr=%.2e  beta1=%.3f  beta2=%.3f  mu=%.2e  n_samples=%d\n",
            __func__, params_.lr, params_.beta1, params_.beta2,
            params_.mu, params_.n_samples);
    LOG_INF("%s:   trainable tensors=%zu  forward passes/sample=%d\n",
            __func__, trainable_params_.size(), 2 * params_.n_samples);
}

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

float RAdaZOOptimizer::compute_loss(float * logits, int n_vocab, llama_token target) {
    if (target >= 0 && target < n_vocab) {
        float max_l = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < n_vocab; ++v) max_l = std::max(max_l, logits[v]);
        double sum = 0.0;
        for (int v = 0; v < n_vocab; ++v) sum += std::exp((double)logits[v] - max_l);
        return (float)((double)max_l + std::log(sum) - (double)logits[target]);
    }
    // Fallback: L2 norm of first 1000 logits
    float loss = 0.0f;
    for (int v = 0; v < std::min(1000, n_vocab); ++v) loss += logits[v] * logits[v];
    return std::sqrt(loss);
}

float RAdaZOOptimizer::compute_loss_with_postprocess(
        struct llama_context * ctx, float * logits, int n_vocab, llama_token target) {
    if (!logits_postprocessor_) return compute_loss(logits, n_vocab, target);
    scratch_logits_.resize(n_vocab);
    std::memcpy(scratch_logits_.data(), logits, n_vocab * sizeof(float));
    logits_postprocessor_(ctx, scratch_logits_.data(), n_vocab);
    return compute_loss(scratch_logits_.data(), n_vocab, target);
}

float RAdaZOOptimizer::compute_batch_loss_with_masks(
        struct llama_context * ctx,
        int n_vocab,
        const std::vector<radazo_loss_target> & targets) {
    if (targets.empty()) return 0.0f;
    double wsum = 0.0, lsum = 0.0;
    for (const auto & t : targets) {
        if (t.batch_token_index < 0 || t.target_token < 0 || t.target_token >= n_vocab) continue;
        float * logits = llama_get_logits_ith(ctx, t.batch_token_index);
        if (!logits) continue;
        const float w = t.weight > 0.0f ? t.weight : 1.0f;
        lsum += w * compute_loss_with_postprocess(ctx, logits, n_vocab, t.target_token);
        wsum += w;
    }
    return wsum > 0.0 ? (float)(lsum / wsum) : 0.0f;
}

// ---------------------------------------------------------------------------
// Perturbation helpers
// ---------------------------------------------------------------------------

void RAdaZOOptimizer::fill_perturbation(
        std::vector<float> & out, int64_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    out.resize(n);
    // Use Rademacher perturbations (+1 / -1) to reduce high-dimensional
    // variance compared with Gaussian noise in global SPSA.
    for (int64_t i = 0; i < n; ++i) {
        out[(size_t) i] = (gen() & 1u) ? 1.0f : -1.0f;
    }
}

// Per-param seed derived from global sample seed + param index.
// Multiplier is an odd constant for good bit mixing.
static inline uint32_t param_seed(uint32_t sample_seed, size_t param_idx) {
    return sample_seed ^ (uint32_t)(param_idx * 2654435761u);
}

void RAdaZOOptimizer::perturb_all_params(float scale, uint32_t seed) {
    for (size_t pi = 0; pi < trainable_params_.size(); ++pi) {
        struct ggml_tensor * p = trainable_params_[pi];
        const int64_t n = ggml_nelements(p);

        fill_perturbation(scratch_direction_, n, param_seed(seed, pi));

        scratch_param_buf_.resize(n);
        const auto it = master_weights_.find(p);
        if (it == master_weights_.end()) {
            continue;
        }
        const std::vector<float> & master = it->second;
        for (int64_t i = 0; i < n; ++i)
            scratch_param_buf_[i] = master[(size_t) i] + scale * scratch_direction_[(size_t) i];
        ggml_backend_tensor_set(p, scratch_param_buf_.data(), 0, n * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// Adam update
// ---------------------------------------------------------------------------

radazo_param_state & RAdaZOOptimizer::get_state(struct ggml_tensor * p) {
    auto it = param_states_.find(p);
    if (it == param_states_.end()) {
        const int64_t n = ggml_nelements(p);
        radazo_param_state s;
        s.exp_avg.assign(n, 0.0f);
        s.exp_avg_sq.assign(n, 0.0f);
        param_states_[p] = std::move(s);
        return param_states_[p];
    }
    return it->second;
}

void RAdaZOOptimizer::update_parameter_adam(
        struct ggml_tensor * p,
        const std::vector<float> & gradient) {
    const int64_t n = ggml_nelements(p);
    if ((int64_t)gradient.size() != n) {
        LOG_ERR("%s: gradient size mismatch\n", __func__);
        return;
    }

    radazo_param_state & st = get_state(p);
    st.step++;

    const float bc1 = 1.0f - std::pow(params_.beta1, (float)st.step);
    const float bc2 = 1.0f - std::pow(params_.beta2, (float)st.step);
    const float ib1 = bc1 > 0.0f ? 1.0f / bc1 : 1.0f;
    const float ib2 = bc2 > 0.0f ? 1.0f / bc2 : 1.0f;

    auto wit = master_weights_.find(p);
    if (wit == master_weights_.end()) {
        LOG_ERR("%s: missing master copy\n", __func__);
        return;
    }
    std::vector<float> & master = wit->second;
    scratch_new_values_.resize(n);

    for (int64_t i = 0; i < n; ++i) {
        const float g = gradient[i];
        float & m = st.exp_avg[i];
        float & v = st.exp_avg_sq[i];
        m = params_.beta1 * m + (1.0f - params_.beta1) * g;
        // R-AdaZO tweak: second moment tracks m^2 instead of g^2
        v = params_.beta2 * v + (1.0f - params_.beta2) * m * m;
        const float m_hat = m * ib1;
        const float v_hat = v * ib2;
        const float new_value = master[(size_t) i] - params_.lr * m_hat / (std::sqrt(v_hat) + params_.eps);
        scratch_new_values_[(size_t) i] = new_value;
        master[(size_t) i] = new_value;
    }

    ggml_backend_tensor_set(p, scratch_new_values_.data(), 0, n * sizeof(float));
}

void RAdaZOOptimizer::adam_update_all_params(float grad_scale, uint32_t seed) {
    for (size_t pi = 0; pi < trainable_params_.size(); ++pi) {
        struct ggml_tensor * p = trainable_params_[pi];
        const int64_t n = ggml_nelements(p);

        // Regenerate the same direction used during perturbation.
        fill_perturbation(scratch_direction_, n, param_seed(seed, pi));

        scratch_grad_.resize(n);
        for (int64_t i = 0; i < n; ++i)
            scratch_grad_[i] = grad_scale * scratch_direction_[i];

        update_parameter_adam(p, scratch_grad_);
    }
}

// ---------------------------------------------------------------------------
// Main optimisation step  —  2 * n_samples forward passes
// ---------------------------------------------------------------------------

float RAdaZOOptimizer::step(
        struct llama_context * ctx,
        llama_batch & batch,
        int n_vocab,
        const std::vector<radazo_loss_target> & loss_targets) {
    if (trainable_params_.empty()) {
        LOG_ERR("%s: no trainable parameters\n", __func__);
        return 0.0f;
    }

    global_step++;
    double loss_accum = 0.0;
    int    valid      = 0;

    for (int s = 0; s < params_.n_samples; ++s) {
        const uint32_t seed = rng_();

        // ── Phase 1: apply  +mu * u  to all params ──────────────────────────
        perturb_all_params(+params_.mu, seed);

        // ── Phase 2: forward pass (+) ───────────────────────────────────────
        llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);
        llama_synchronize(ctx);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed (+), sample %d\n", __func__, s);
            perturb_all_params(0.0f, seed);  // restore baseline
            continue;
        }
        forward_passes++;
        const float loss_plus = loss_targets.empty()
            ? compute_loss_with_postprocess(
                ctx, llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab, -1)
            : compute_batch_loss_with_masks(ctx, n_vocab, loss_targets);

        // ── Phase 3: set  -mu * u  to all params ────────────────────────────
        perturb_all_params(-params_.mu, seed);

        // ── Phase 4: forward pass (-) ───────────────────────────────────────
        llama_memory_seq_rm(llama_get_memory(ctx), -1, -1, -1);
        llama_synchronize(ctx);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed (-), sample %d\n", __func__, s);
            perturb_all_params(0.0f, seed);  // restore baseline
            continue;
        }
        forward_passes++;
        const float loss_minus = loss_targets.empty()
            ? compute_loss_with_postprocess(
                ctx, llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab, -1)
            : compute_batch_loss_with_masks(ctx, n_vocab, loss_targets);

        // ── Phase 5: restore baseline ────────────────────────────────────────
        perturb_all_params(0.0f, seed);

        // ── Phase 6: Adam update for all params ──────────────────────────────
        const float grad_scale = (loss_plus - loss_minus) / (2.0f * params_.mu);
        adam_update_all_params(grad_scale, seed);

        loss_accum += 0.5 * ((double)loss_plus + (double)loss_minus);
        ++valid;

        if (params_.log_gradients) {
            LOG_INF("  [R-AdaZO] step=%ld sample=%d loss+/−=%.4f/%.4f scale=%.4e\n",
                    global_step, s, loss_plus, loss_minus, grad_scale);
        }
    }

    total_updates += (int64_t)trainable_params_.size() * valid;
    return valid > 0 ? (float)(loss_accum / valid) : 0.0f;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

std::vector<struct ggml_tensor *> collect_trainable_parameters_radazo(
        struct llama_context * ctx, bool verbose) {
    std::vector<struct ggml_tensor *> out;
    const llama_model * model = llama_get_model(ctx);
    const size_t n = llama_model_n_tensors(model);
    char tname[256];
    for (size_t i = 0; i < n; ++i) {
        struct ggml_tensor * t = llama_model_get_tensor_by_index(model, i, tname, sizeof(tname));
        if (!t) continue;
        const std::string name(tname);
        if (name.find("attn_q.weight")      != std::string::npos ||
            name.find("attn_k.weight")      != std::string::npos ||
            name.find("attn_v.weight")      != std::string::npos ||
            name.find("attn_output.weight") != std::string::npos ||
            name.find("ffn_gate.weight")    != std::string::npos ||
            name.find("ffn_up.weight")      != std::string::npos ||
            name.find("ffn_down.weight")    != std::string::npos) {
            out.push_back(t);
            if (verbose && out.size() <= 10)
                LOG_INF("  [%zu] %s\n", out.size(), tname);
        }
    }
    if (verbose) LOG_INF("collect_trainable: %zu tensors\n", out.size());
    return out;
}
