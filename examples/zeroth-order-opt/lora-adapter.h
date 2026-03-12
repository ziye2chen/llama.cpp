// LoRA Adapter for GGUF Models
// Uses llama.cpp native llama_adapter_lora infrastructure so that
// llama_decode() automatically computes Y = W_base*X + (alpha/rank)*B*A*X
// without ever writing to base tensors.  ZO training only touches FP32 A/B.

#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>

// Full definition is in llama-adapter.h (included by lora-adapter.cpp).
// Forward-declaration keeps this public header free of internal types.
struct llama_adapter_lora;

// LoRA configuration
struct lora_config {
    int rank = 8;
    float alpha = 16.0f;
    float dropout = 0.0f;  // kept for API compatibility; not used in ZO
    std::vector<std::string> target_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    };
};

// LoRA adapter for a single base-model layer.
// Tensor shapes follow the native llama_adapter_lora convention that
// build_lora_mm() in llama-graph.cpp expects:
//   lora_A : [in_dim, rank]   FP32  (ne[0]=in_dim, ne[1]=rank)
//   lora_B : [rank,   out_dim] FP32  (ne[0]=rank,   ne[1]=out_dim)
// where in_dim = base_tensor->ne[0], out_dim = base_tensor->ne[1].
//
// Forward: Y = W_base*X + scale * lora_B^T * (lora_A^T * X)
//        = W_base*X + (alpha/rank) * B * A * X
struct lora_layer {
    struct ggml_tensor * lora_A     = nullptr;
    struct ggml_tensor * lora_B     = nullptr;
    struct ggml_tensor * base_tensor = nullptr;
    std::string name;
    int64_t in_dim  = 0;   // = base_tensor->ne[0]
    int64_t out_dim = 0;   // = base_tensor->ne[1]
    int     rank    = 0;
    float   alpha   = 0.0f;

    // Temporary init data stored before backend buffer allocation.
    std::vector<float> init_A_data;
    std::vector<float> init_B_data;
};

// LoRA adapter manager.
// After initialize(), the native llama_adapter_lora is registered with the
// context.  All subsequent llama_decode() calls automatically apply LoRA.
// ZO training modifies lora_A / lora_B FP32 tensors directly; no merging
// into base weights is ever needed during training.
class LoRAAdapter {
public:
    LoRAAdapter(struct llama_context * ctx, const lora_config & config);
    ~LoRAAdapter();

    // Create LoRA tensors, allocate them, and register the native adapter.
    bool initialize();

    // No-ops retained for call-site compatibility with finetune-radazo-quant.cpp.
    // Training never touches base tensors; no merge/restore needed.
    bool apply_to_base_tensors() { return true; }
    bool restore_base_tensors()  { return true; }

    // Merge trained LoRA into base model weights and write a new GGUF file.
    bool merge_and_save(struct llama_model * model, const std::string & output_path) const;

    // Save LoRA adapters as standalone FP32 GGUF (no merge). Use with --lora at inference.
    bool save_lora_standalone(struct llama_model * model, const std::string & output_path) const;

    // Return all trainable FP32 tensors (alternating A, B per layer).
    std::vector<struct ggml_tensor *> get_trainable_params() const;

    lora_layer * get_layer(const std::string & name);

    // Persistence stubs (not implemented; ZO only needs in-memory tensors).
    bool save(const std::string & filepath) const;
    bool load(const std::string & filepath);

    size_t  get_num_layers()   const { return lora_layers_.size(); }
    int64_t get_total_params() const;

private:
    bool create_lora_for_tensor(
        struct ggml_tensor * base_tensor,
        const std::string  & tensor_name);

    bool is_target_module(const std::string & name) const;

    struct llama_context * ctx_;
    lora_config            config_;

    // Keyed by base tensor name, e.g. "blk.0.attn_q.weight"
    std::unordered_map<std::string, lora_layer> lora_layers_;

    // ggml context owning all lora_A / lora_B allocations.
    struct ggml_context * lora_ctx_ = nullptr;

    // Native adapter registered with the llama_context.
    // build_lora_mm() in llama-graph.cpp reads A/B from here every decode.
    llama_adapter_lora * native_adapter_ = nullptr;
};

// Helper: collect base tensors that would receive LoRA adapters.
std::vector<struct ggml_tensor *> collect_lora_target_tensors(
    struct llama_context * ctx,
    const lora_config    & config,
    bool verbose = true);
