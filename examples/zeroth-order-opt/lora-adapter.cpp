// LoRA Adapter Implementation
// Registers an in-memory llama_adapter_lora with the llama_context so that
// llama_decode() automatically inserts the LoRA path into the compute graph.
// Base tensors are NEVER modified during training; ZO only perturbs FP32 A/B.

#include "lora-adapter.h"
#include "llama-adapter.h"   // internal: llama_adapter_lora, llama_adapter_lora_weight
#include "log.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include "quant-gguf-support.h"

#include <cmath>
#include <algorithm>
#include <fstream>
#include <random>
#include <cstring>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

static bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::unordered_set<std::string> normalize_lora_targets(
        const std::vector<std::string> & targets) {
    std::unordered_set<std::string> out;
    for (const auto & t : targets) {
        if      (t == "q_proj"    || t == "attn_q"  || t == "wq") out.insert("q_proj");
        else if (t == "k_proj"    || t == "attn_k"  || t == "wk") out.insert("k_proj");
        else if (t == "v_proj"    || t == "attn_v"  || t == "wv") out.insert("v_proj");
        else if (t == "o_proj"    || t == "attn_output" || t == "wo") out.insert("o_proj");
        else if (t == "gate_proj" || t == "ffn_gate" || t == "w1") out.insert("gate_proj");
        else if (t == "down_proj" || t == "ffn_down" || t == "w2") out.insert("down_proj");
        else if (t == "up_proj"   || t == "ffn_up"  || t == "w3") out.insert("up_proj");
        else out.insert(t);
    }
    return out;
}

static bool tensor_matches_lora_targets(
        const std::string & name,
        const std::unordered_set<std::string> & targets) {
    if (!ends_with(name, ".weight")) return false;
    if (targets.count("q_proj")    && (ends_with(name, ".q_proj.weight")    || ends_with(name, ".attn_q.weight")     || ends_with(name, ".wq.weight")))     return true;
    if (targets.count("k_proj")    && (ends_with(name, ".k_proj.weight")    || ends_with(name, ".attn_k.weight")     || ends_with(name, ".wk.weight")))     return true;
    if (targets.count("v_proj")    && (ends_with(name, ".v_proj.weight")    || ends_with(name, ".attn_v.weight")     || ends_with(name, ".wv.weight")))     return true;
    if (targets.count("o_proj")    && (ends_with(name, ".o_proj.weight")    || ends_with(name, ".attn_output.weight")|| ends_with(name, ".wo.weight")))     return true;
    if (targets.count("gate_proj") && (ends_with(name, ".gate_proj.weight") || ends_with(name, ".ffn_gate.weight")   || ends_with(name, ".w1.weight")))  return true;
    if (targets.count("down_proj") && (ends_with(name, ".down_proj.weight") || ends_with(name, ".ffn_down.weight")   || ends_with(name, ".w2.weight")))  return true;
    if (targets.count("up_proj")   && (ends_with(name, ".up_proj.weight")   || ends_with(name, ".ffn_up.weight")     || ends_with(name, ".w3.weight")))   return true;
    return false;
}

// Compute delta = (alpha/rank) * B * A  in row-major [out_dim x in_dim] layout.
//
// Tensor shapes (native adapter convention, ggml column-major):
//   lora_A : ne[0]=in_dim,  ne[1]=rank   → A_data[j + k*in_dim]  = A[k,j]
//   lora_B : ne[0]=rank,    ne[1]=out_dim → B_data[k + i*rank]    = B[i,k]
//
// delta[i*in_dim + j] = (alpha/rank) * sum_k B[i,k] * A[k,j]
static void build_lora_delta(
        const lora_layer         & layer,
        const std::vector<float> & a_data,
        const std::vector<float> & b_data,
        std::vector<float>       & delta) {
    const float scale = layer.alpha / layer.rank;
    delta.assign((size_t)(layer.out_dim * layer.in_dim), 0.0f);
    for (int64_t i = 0; i < layer.out_dim; ++i) {
        for (int64_t j = 0; j < layer.in_dim; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < layer.rank; ++k) {
                sum += b_data[(size_t)(k + i * layer.rank)] *
                       a_data[(size_t)(j + k * layer.in_dim)];
            }
            delta[(size_t)(i * layer.in_dim + j)] = sum * scale;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// LoRAAdapter
// ---------------------------------------------------------------------------

LoRAAdapter::LoRAAdapter(struct llama_context * ctx, const lora_config & config)
    : ctx_(ctx), config_(config) {
    const size_t mem_size = 256ULL * 1024 * 1024;  // 256 MB metadata
    ggml_init_params p = { mem_size, nullptr, /*no_alloc=*/true };
    lora_ctx_ = ggml_init(p);
    if (!lora_ctx_) {
        LOG_ERR("%s: failed to create LoRA ggml context\n", __func__);
    }
}

LoRAAdapter::~LoRAAdapter() {
    // Deregister from context before freeing tensors.
    if (native_adapter_ && ctx_) {
        llama_rm_adapter_lora(ctx_, native_adapter_);
    }
    // Deleting native_adapter_ is safe: its ctxs/bufs vectors are empty
    // (we manage our own tensors via lora_ctx_).
    delete native_adapter_;
    native_adapter_ = nullptr;

    if (lora_ctx_) {
        ggml_free(lora_ctx_);
        lora_ctx_ = nullptr;
    }
}

bool LoRAAdapter::is_target_module(const std::string & name) const {
    return tensor_matches_lora_targets(name, normalize_lora_targets(config_.target_modules));
}

bool LoRAAdapter::create_lora_for_tensor(
        struct ggml_tensor * base_tensor,
        const std::string  & tensor_name) {
    if (!base_tensor || !lora_ctx_) return false;

    // Native adapter convention (matches build_lora_mm in llama-graph.cpp):
    //   lora_A : [in_dim, rank]    in_dim = base->ne[0]
    //   lora_B : [rank,   out_dim] out_dim = base->ne[1]
    const int64_t in_dim  = base_tensor->ne[0];
    const int64_t out_dim = base_tensor->ne[1];
    if (in_dim <= 0 || out_dim <= 0) return false;

    struct ggml_tensor * lora_A = ggml_new_tensor_2d(lora_ctx_, GGML_TYPE_F32, in_dim,        config_.rank);
    struct ggml_tensor * lora_B = ggml_new_tensor_2d(lora_ctx_, GGML_TYPE_F32, config_.rank,  out_dim);
    if (!lora_A || !lora_B) {
        LOG_ERR("%s: failed to create LoRA tensors for %s\n", __func__, tensor_name.c_str());
        return false;
    }

    // Assign names so the native adapter's ab_map lookup works.
    ggml_set_name(lora_A, (tensor_name + ".lora_a").c_str());
    ggml_set_name(lora_B, (tensor_name + ".lora_b").c_str());

    // Kaiming init for A: std = 1/sqrt(rank)
    std::mt19937 gen(42 + (uint32_t)lora_layers_.size());
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float)config_.rank));
    std::vector<float> init_A(ggml_nelements(lora_A));
    for (float & v : init_A) v = dist(gen);

    lora_layer layer;
    layer.lora_A       = lora_A;
    layer.lora_B       = lora_B;
    layer.base_tensor  = base_tensor;
    layer.name         = tensor_name;
    layer.in_dim       = in_dim;
    layer.out_dim      = out_dim;
    layer.rank         = config_.rank;
    layer.alpha        = config_.alpha;
    layer.init_A_data  = std::move(init_A);
    layer.init_B_data.assign(ggml_nelements(lora_B), 0.0f);

    lora_layers_[tensor_name] = std::move(layer);

    LOG_INF("%s: created LoRA for %s ([%ld,%ld] rank=%d)\n",
            __func__, tensor_name.c_str(), out_dim, in_dim, config_.rank);
    return true;
}

bool LoRAAdapter::initialize() {
    if (!lora_ctx_) {
        LOG_ERR("%s: LoRA context not initialised\n", __func__);
        return false;
    }

    LOG_INF("%s: initialising LoRA adapters (rank=%d alpha=%.1f)\n",
            __func__, config_.rank, config_.alpha);

    const llama_model * model = llama_get_model(ctx_);
    const size_t n_tensors    = llama_model_n_tensors(model);

    size_t created = 0;
    char tname[256];
    for (size_t i = 0; i < n_tensors; ++i) {
        struct ggml_tensor * t = llama_model_get_tensor_by_index(model, i, tname, sizeof(tname));
        if (!t) continue;
        if (is_target_module(std::string(tname))) {
            if (create_lora_for_tensor(t, tname)) ++created;
        }
    }

    if (created == 0) {
        LOG_ERR("%s: no LoRA adapters created\n", __func__);
        return false;
    }
    LOG_INF("%s: created %zu LoRA adapters\n", __func__, created);

    // Allocate buffers for A/B. Use CPU buffer type so ggml_backend_tensor_get/set work
    // (AMX buffer has get_tensor=nullptr and would segfault when R-AdaZO reads master copy).
    {
        auto first_base = lora_layers_.begin()->second.base_tensor;
        if (!first_base->buffer) {
            LOG_ERR("%s: base tensor has no buffer\n", __func__);
            return false;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(lora_ctx_, buft);
        if (!buf) {
            LOG_ERR("%s: failed to allocate backend buffer for LoRA tensors\n", __func__);
            return false;
        }
        LOG_INF("%s: allocated LoRA backend buffer\n", __func__);
    }

    // Upload initialisation data.
    for (auto & kv : lora_layers_) {
        lora_layer & layer = kv.second;
        if (!layer.init_A_data.empty()) {
            ggml_backend_tensor_set(layer.lora_A, layer.init_A_data.data(),
                                    0, layer.init_A_data.size() * sizeof(float));
        }
        if (!layer.init_B_data.empty()) {
            ggml_backend_tensor_set(layer.lora_B, layer.init_B_data.data(),
                                    0, layer.init_B_data.size() * sizeof(float));
        }
        // Free temporary init storage.
        layer.init_A_data.clear();
        layer.init_A_data.shrink_to_fit();
        layer.init_B_data.clear();
        layer.init_B_data.shrink_to_fit();
    }
    llama_synchronize(ctx_);

    // Build native adapter and register with context.
    // ab_map key = base tensor name; A/B pointers point into our lora_ctx_.
    native_adapter_ = new llama_adapter_lora();
    native_adapter_->alpha = config_.alpha;
    for (const auto & kv : lora_layers_) {
        native_adapter_->ab_map[kv.first] =
            llama_adapter_lora_weight(kv.second.lora_A, kv.second.lora_B);
    }

    // scale=1.0: effective scale = alpha * 1.0 / rank = alpha/rank (via get_scale())
    llama_set_adapter_lora(ctx_, native_adapter_, 1.0f);
    LOG_INF("%s: native LoRA adapter registered (alpha=%.1f, scale=1.0 -> effective=%.4f)\n",
            __func__, config_.alpha, config_.alpha / config_.rank);

    return true;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

std::vector<struct ggml_tensor *> LoRAAdapter::get_trainable_params() const {
    std::vector<struct ggml_tensor *> out;
    out.reserve(lora_layers_.size() * 2);
    for (const auto & kv : lora_layers_) {
        out.push_back(kv.second.lora_A);
        out.push_back(kv.second.lora_B);
    }
    return out;
}

lora_layer * LoRAAdapter::get_layer(const std::string & name) {
    auto it = lora_layers_.find(name);
    return it != lora_layers_.end() ? &it->second : nullptr;
}

int64_t LoRAAdapter::get_total_params() const {
    int64_t total = 0;
    for (const auto & kv : lora_layers_) {
        total += ggml_nelements(kv.second.lora_A);
        total += ggml_nelements(kv.second.lora_B);
    }
    return total;
}

bool LoRAAdapter::save(const std::string & filepath) const {
    LOG_INF("%s: save not implemented (%s)\n", __func__, filepath.c_str());
    return false;
}

bool LoRAAdapter::load(const std::string & filepath) {
    LOG_INF("%s: load not implemented (%s)\n", __func__, filepath.c_str());
    return false;
}

// ---------------------------------------------------------------------------
// merge_and_save
// Bakes trained LoRA into base weights and writes a new GGUF file.
// ---------------------------------------------------------------------------

bool LoRAAdapter::merge_and_save(
        struct llama_model * model,
        const std::string  & output_path) const {
    if (!model) { LOG_ERR("%s: null model\n", __func__); return false; }

    LOG_INF("%s: merging %zu LoRA layers into base model...\n",
            __func__, lora_layers_.size());

    // Resolve merge targets from the provided model by tensor name.
    std::unordered_map<std::string, struct ggml_tensor *> base_by_name;
    base_by_name.reserve(llama_model_n_tensors(model));
    char tname[256];
    for (size_t i = 0; i < llama_model_n_tensors(model); ++i) {
        struct ggml_tensor * t = llama_model_get_tensor_by_index(model, i, tname, sizeof(tname));
        if (t) {
            base_by_name[std::string(tname)] = t;
        }
    }

    for (const auto & kv : lora_layers_) {
        const lora_layer & layer = kv.second;
        auto it_base = base_by_name.find(layer.name);
        if (it_base == base_by_name.end()) {
            LOG_WRN("%s: skip %s (tensor not found in merge model)\n", __func__, layer.name.c_str());
            continue;
        }
        struct ggml_tensor * base = it_base->second;
        if (!base || !base->buffer) {
            LOG_WRN("%s: skip %s (no base tensor)\n", __func__, layer.name.c_str());
            continue;
        }

        // Read current A/B from device.
        std::vector<float> a_data(ggml_nelements(layer.lora_A));
        std::vector<float> b_data(ggml_nelements(layer.lora_B));
        ggml_backend_tensor_get(layer.lora_A, a_data.data(), 0, a_data.size() * sizeof(float));
        ggml_backend_tensor_get(layer.lora_B, b_data.data(), 0, b_data.size() * sizeof(float));
        llama_synchronize(ctx_);

        std::vector<float> delta;
        build_lora_delta(layer, a_data, b_data, delta);

        const enum ggml_type qtype = base->type;
        const char * quant_name = radazo_quant_support::display_quant_type_name(qtype);

        if (ggml_is_quantized(qtype)) {
            // ---- Phase 1: Double Quantization Loss diagnostic probes ----
            float delta_max_abs = 0.0f;
            double delta_sum_abs = 0.0;
            for (float v : delta) {
                const float a = std::abs(v);
                delta_max_abs = std::max(delta_max_abs, a);
                delta_sum_abs += (double)a;
            }
            const float delta_mean_abs = (float)(delta_sum_abs / (delta.empty() ? 1 : delta.size()));
            LOG_INF("%s: [Probe A] %s delta for %s: max_abs=%.6e mean_abs=%.6e (requant rounding may suppress small updates)\n",
                    __func__, layer.name.c_str(), quant_name, delta_max_abs, delta_mean_abs);
        }

        if (!ggml_is_quantized(qtype)) {
            // FP32 path
            const int64_t n_elem = layer.out_dim * layer.in_dim;
            std::vector<float> base_f32(n_elem);
            ggml_backend_tensor_get(base, base_f32.data(), 0, n_elem * sizeof(float));
            llama_synchronize(ctx_);
            for (int64_t i = 0; i < n_elem; ++i) base_f32[i] += delta[i];
            ggml_backend_tensor_set(base, base_f32.data(), 0, n_elem * sizeof(float));
            llama_synchronize(ctx_);
            LOG_INF("%s: merged %s (F32)\n", __func__, layer.name.c_str());
            continue;
        }

        // Quantised path: dequantise → add delta → requantise
        const bool verified_quant_type = radazo_quant_support::is_verified_quant_type(qtype);
        const struct ggml_type_traits * traits = ggml_get_type_traits(qtype);
        if (!traits || !traits->to_float || !traits->from_float_ref) {
            LOG_ERR("%s: unsupported quantized tensor %s: type %s has no dequant/requant traits\n",
                    __func__, layer.name.c_str(), quant_name);
            return false;
        }

        if (verified_quant_type) {
            LOG_INF("%s: merging %s using verified quant format %s\n",
                    __func__, layer.name.c_str(), quant_name);
        } else {
            LOG_WRN("%s: quant format %s is not in the verified support set; attempting generic traits-driven merge for %s\n",
                    __func__, quant_name, layer.name.c_str());
        }

        const int64_t nrows     = ggml_nrows(base);
        const int64_t n_per_row = base->ne[0];   // = in_dim
        const size_t  row_size  = ggml_row_size(qtype, n_per_row);
        const size_t  nbytes    = ggml_nbytes(base);

        std::vector<uint8_t> quant_data(nbytes);
        ggml_backend_tensor_get(base, quant_data.data(), 0, nbytes);
        llama_synchronize(ctx_);

        std::vector<float> row_f32((size_t)n_per_row);
        std::vector<uint8_t> row_before_quant(row_size);
        int64_t n_blocks_changed = 0;
        const int64_t probe_b_rows = (nrows > 0) ? 1 : 0;  // first row only for Probe B

        for (int64_t ir = 0; ir < nrows; ++ir) {   // ir = out_dim index
            void * quant_row = quant_data.data() + (size_t)ir * row_size;
            traits->to_float(quant_row, row_f32.data(), n_per_row);

            if (ir < probe_b_rows) {
                LOG_INF("%s: [Probe B] %s row %ld BEFORE+delta (%s): first 10 fp32 = ", __func__, layer.name.c_str(), ir, quant_name);
                for (int64_t ic = 0; ic < std::min(10L, n_per_row); ++ic) {
                    fprintf(stderr, "%.6e ", row_f32[(size_t)ic]);
                }
                fprintf(stderr, "\n");
            }

            for (int64_t ic = 0; ic < n_per_row; ++ic) {  // ic = in_dim index
                row_f32[(size_t)ic] += delta[(size_t)(ir * n_per_row + ic)];
            }

            if (ir < probe_b_rows) {
                LOG_INF("%s: [Probe B] %s row %ld AFTER+delta (%s): first 10 fp32 = ", __func__, layer.name.c_str(), ir, quant_name);
                for (int64_t ic = 0; ic < std::min(10L, n_per_row); ++ic) {
                    fprintf(stderr, "%.6e ", row_f32[(size_t)ic]);
                }
                fprintf(stderr, "\n");
            }

            std::memcpy(row_before_quant.data(), quant_row, row_size);
            traits->from_float_ref(row_f32.data(), quant_row, n_per_row);
            if (std::memcmp(row_before_quant.data(), quant_row, row_size) != 0) {
                n_blocks_changed++;
            }
        }

        LOG_INF("%s: [Probe C] %s (%s): %ld/%ld rows changed after requant (0%% = all updates rounded away)\n",
                __func__, layer.name.c_str(), quant_name, n_blocks_changed, nrows);

        ggml_backend_tensor_set(base, quant_data.data(), 0, nbytes);
        llama_synchronize(ctx_);
        LOG_INF("%s: merged %s ([%ld,%ld] type %s)\n",
                __func__, layer.name.c_str(), layer.out_dim, layer.in_dim,
                quant_name);
    }

    LOG_INF("%s: saving merged model to %s\n", __func__, output_path.c_str());
    llama_model_save_to_file(model, output_path.c_str());
    LOG_INF("%s: model saved successfully with LoRA merged!\n", __func__);
    return true;
}

// ---------------------------------------------------------------------------
// save_lora_standalone
// Export LoRA A/B as standalone FP32 GGUF for inference with --lora (no merge).
// Use to verify ZO training: if FP32 LoRA changes output vs baseline, merge
// requantization was wiping updates.
// ---------------------------------------------------------------------------

bool LoRAAdapter::save_lora_standalone(
        struct llama_model * model,
        const std::string  & output_path) const {
    if (!model) { LOG_ERR("%s: null model\n", __func__); return false; }

    LOG_INF("%s: saving %zu LoRA layers as standalone FP32 adapter to %s\n",
            __func__, lora_layers_.size(), output_path.c_str());

    struct gguf_context * ctx_gguf = gguf_init_empty();
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to create GGUF context\n", __func__);
        return false;
    }

    gguf_set_val_str(ctx_gguf, "general.type", "adapter");
    char arch_buf[64];
    if (llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf)) > 0) {
        gguf_set_val_str(ctx_gguf, "general.architecture", arch_buf);
    } else {
        gguf_set_val_str(ctx_gguf, "general.architecture", "unknown");
    }
    gguf_set_val_str(ctx_gguf, "adapter.type", "lora");
    gguf_set_val_f32(ctx_gguf, "adapter.lora.alpha", config_.alpha);

    ggml_init_params meta_params = { ggml_tensor_overhead() * lora_layers_.size() * 2, nullptr, true };
    struct ggml_context * ctx_meta = ggml_init(meta_params);
    if (!ctx_meta) {
        gguf_free(ctx_gguf);
        LOG_ERR("%s: failed to create meta context\n", __func__);
        return false;
    }

    std::vector<std::vector<float>> data_buffers;
    data_buffers.reserve(lora_layers_.size() * 2);

    for (const auto & kv : lora_layers_) {
        const lora_layer & layer = kv.second;
        const std::string name_a = layer.name + ".lora_a";
        const std::string name_b = layer.name + ".lora_b";

        struct ggml_tensor * t_a = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32,
                layer.lora_A->ne[0], layer.lora_A->ne[1]);
        struct ggml_tensor * t_b = ggml_new_tensor_2d(ctx_meta, GGML_TYPE_F32,
                layer.lora_B->ne[0], layer.lora_B->ne[1]);
        ggml_set_name(t_a, name_a.c_str());
        ggml_set_name(t_b, name_b.c_str());

        gguf_add_tensor(ctx_gguf, t_a);
        gguf_add_tensor(ctx_gguf, t_b);

        data_buffers.emplace_back(ggml_nelements(layer.lora_A));
        data_buffers.emplace_back(ggml_nelements(layer.lora_B));

        ggml_backend_tensor_get(layer.lora_A, data_buffers[data_buffers.size() - 2].data(),
                0, data_buffers[data_buffers.size() - 2].size() * sizeof(float));
        ggml_backend_tensor_get(layer.lora_B, data_buffers[data_buffers.size() - 1].data(),
                0, data_buffers[data_buffers.size() - 1].size() * sizeof(float));
    }
    llama_synchronize(ctx_);

    size_t buf_idx = 0;
    for (const auto & kv : lora_layers_) {
        const lora_layer & layer = kv.second;
        gguf_set_tensor_data(ctx_gguf, (layer.name + ".lora_a").c_str(),
                data_buffers[buf_idx].data());
        buf_idx++;
        gguf_set_tensor_data(ctx_gguf, (layer.name + ".lora_b").c_str(),
                data_buffers[buf_idx].data());
        buf_idx++;
    }

    bool ok = gguf_write_to_file(ctx_gguf, output_path.c_str(), false);
    ggml_free(ctx_meta);
    gguf_free(ctx_gguf);

    if (ok) {
        LOG_INF("%s: LoRA adapter saved. Test with: ./llama-cli -m base.gguf --lora %s -p \"...\"\n",
                __func__, output_path.c_str());
    } else {
        LOG_ERR("%s: failed to write GGUF to %s\n", __func__, output_path.c_str());
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

std::vector<struct ggml_tensor *> collect_lora_target_tensors(
        struct llama_context * ctx,
        const lora_config    & config,
        bool verbose) {
    std::vector<struct ggml_tensor *> out;
    const auto targets = normalize_lora_targets(config.target_modules);
    const llama_model * model = llama_get_model(ctx);
    const size_t n = llama_model_n_tensors(model);
    char tname[256];
    for (size_t i = 0; i < n; ++i) {
        struct ggml_tensor * t = llama_model_get_tensor_by_index(model, i, tname, sizeof(tname));
        if (t && tensor_matches_lora_targets(std::string(tname), targets)) {
            out.push_back(t);
            if (verbose && out.size() <= 10)
                LOG_INF("%s:   [%zu] %s\n", __func__, out.size(), tname);
        }
    }
    if (verbose) LOG_INF("%s: collected %zu target tensors\n", __func__, out.size());
    return out;
}
