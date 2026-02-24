// LoRA Adapter for Quantized GGUF Models
// Implements Low-Rank Adaptation (LoRA) for Q4_K_M and other quantized models

#pragma once

#include "llama.h"
#include "ggml.h"

#include <vector>
#include <string>
#include <unordered_map>

// LoRA configuration
struct lora_config {
    int rank = 8;                    // LoRA rank (r)
    float alpha = 16.0f;             // LoRA scaling factor (alpha)
    float dropout = 0.0f;            // Dropout (not used in zeroth-order, kept for compatibility)
    std::vector<std::string> target_modules = {
        "attn_q", "attn_k", "attn_v", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down"
    };                               // Target modules for LoRA
};

// LoRA adapter for a single layer
struct lora_layer {
    struct ggml_tensor * lora_A;     // [rank, in_dim] - initialized with Gaussian
    struct ggml_tensor * lora_B;     // [out_dim, rank] - initialized with zeros
    struct ggml_tensor * base_tensor; // Reference to original GGUF tensor (read-only)
    std::string name;                // Layer name for identification
    int64_t in_dim;                  // Input dimension
    int64_t out_dim;                 // Output dimension
    int rank;                        // LoRA rank
    float alpha;                     // Scaling factor
    std::vector<float> init_A_data;  // Initialization data for A (before buffer allocation)
    std::vector<float> init_B_data;  // Initialization data for B (before buffer allocation)
};

// LoRA adapter manager
class LoRAAdapter {
public:
    // Constructor
    LoRAAdapter(
        struct llama_context * ctx,
        const lora_config & config);
    
    // Destructor
    ~LoRAAdapter();
    
    // Initialize LoRA adapters for target layers
    bool initialize();
    
    // Forward pass: Y = Base(X) + LoRA(X)
    // Returns a new tensor that combines base and LoRA outputs
    struct ggml_tensor * forward(
        struct ggml_context * ctx,
        struct ggml_tensor * input,
        const std::string & layer_name);
    
    // Apply LoRA to logits after llama_decode
    // This is a simplified approach: compute LoRA's effect on final logits
    // by approximating the cumulative effect through all LoRA layers
    void apply_lora_to_logits(
        struct llama_context * ctx,
        float * logits,
        int n_vocab) const;
    
    // Get all LoRA parameters (A and B matrices) for training
    std::vector<struct ggml_tensor *> get_trainable_params() const;
    
    // Get LoRA layer by name
    lora_layer * get_layer(const std::string & name);
    
    // Save LoRA adapters to file
    bool save(const std::string & filepath) const;
    
    // Load LoRA adapters from file
    bool load(const std::string & filepath);
    
    // Merge LoRA adapters into base model and save as new GGUF
    // This will dequantize target layers, apply LoRA deltas, re-quantize, and save
    bool merge_and_save(
        struct llama_model * model,
        const std::string & output_path) const;
    
    // Get statistics
    size_t get_num_layers() const { return lora_layers_.size(); }
    int64_t get_total_params() const;
    
private:
    // Create LoRA adapter for a specific tensor
    bool create_lora_for_tensor(
        struct ggml_context * ctx,
        struct ggml_tensor * base_tensor,
        const std::string & tensor_name);
    
    // Check if a tensor name matches target modules
    bool is_target_module(const std::string & name) const;
    
    // Dequantize Q4_K_M tensor to FP32 (temporary, for forward pass)
    struct ggml_tensor * dequantize_tensor(
        struct ggml_context * ctx,
        struct ggml_tensor * quantized_tensor);
    
    struct llama_context * ctx_;
    lora_config config_;
    std::unordered_map<std::string, lora_layer> lora_layers_;
    struct ggml_context * lora_ctx_;  // Separate context for LoRA tensors
};

// Helper: Collect base tensors that should have LoRA adapters
std::vector<struct ggml_tensor *> collect_lora_target_tensors(
    struct llama_context * ctx,
    const lora_config & config,
    bool verbose = true);
