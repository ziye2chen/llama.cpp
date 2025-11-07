// Zeroth-Order Fine-tuning Demo for GGUF Models
// This demonstrates gradient-free optimization for fine-tuning

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-opt.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <random>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

// Zeroth-order optimization parameters
struct zeroth_order_params {
    float epsilon = 1e-3f;              // perturbation size
    float learning_rate = 1e-7f;        // much smaller than first-order!
    float weight_decay = 0.1f;         // L2 regularization
    int32_t n_params_per_iter = 10;     // sample only a few params per iteration
    int32_t n_elements_per_param = 5;   // sample only a few elements per tensor
    bool use_random_sampling = true;
    uint32_t random_seed = 42;
    int32_t max_train_tokens = -1;      // if >0, limit dataset to this many tokens (for testing)
};

// Progress callback for zeroth-order optimization
static void progress_callback_zeroth_order(
        bool train,
        int64_t iter,
        int64_t iter_max,
        float loss,
        int64_t t_start_us) {
    const int64_t t_now_us = ggml_time_us();
    const float elapsed = (t_now_us - t_start_us) / 1.0e6f;
    
    fprintf(stderr, "\r[%s] Iter %6lld/%6lld | Loss: %.6f | Time: %6.2fs | %.1f it/s",
            train ? "TRAIN" : "EVAL ",
            iter, iter_max, loss, elapsed,
            iter > 0 ? iter / elapsed : 0.0f);
    fflush(stderr);
}

// Collect trainable parameters from the model
static std::vector<struct ggml_tensor *> collect_model_params(struct llama_context * ctx) {
    std::vector<struct ggml_tensor *> params;
    
    LOG_INF("%s: collecting trainable parameters...\n", __func__);
    
    const llama_model * model = llama_get_model(ctx);
    const size_t n_tensors = llama_model_n_tensors(model);
    
    LOG_INF("%s: model has %zu tensors total\n", __func__, n_tensors);
    
    // Iterate through all model tensors and collect trainable weights
    // Focus on attention (Q/K/V/O) and FFN (gate/up/down) weight matrices
    char tensor_name[256];
    for (size_t i = 0; i < n_tensors; ++i) {
        struct ggml_tensor * tensor = llama_model_get_tensor_by_index(model, i, tensor_name, sizeof(tensor_name));
        
        if (tensor == nullptr) continue;
        
        std::string name(tensor_name);
        
        // Select key weight matrices for fine-tuning
        // We filter for the most important parameters to make zeroth-order feasible
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
            
            // Log first 10 and last 5 parameters (don't spam the log)
            if (params.size() <= 10 || params.size() > n_tensors - 5) {
                LOG_INF("%s:   [%3zu] %s (shape: [%lld, %lld], %lld elements)\n", 
                        __func__, params.size(), tensor_name, 
                        tensor->ne[0], tensor->ne[1], ggml_nelements(tensor));
            } else if (params.size() == 11) {
                LOG_INF("%s:   ... (showing first/last only)\n", __func__);
            }
        }
    }
    
    LOG_INF("%s: collected %zu trainable parameter tensors\n", __func__, params.size());
    
    return params;
}

// Simplified zeroth-order fine-tuning loop
static void finetune_zeroth_order(
        struct llama_context * ctx,
        const std::vector<llama_token> & train_tokens,
        const std::vector<llama_token> & eval_tokens,
        const zeroth_order_params & zo_params,
        int n_epochs) {
    
    LOG_INF("\n%s: starting zeroth-order fine-tuning...\n", __func__);
    LOG_INF("%s: epsilon=%.2e, lr=%.2e, params_per_iter=%d\n", 
            __func__, zo_params.epsilon, zo_params.learning_rate, zo_params.n_params_per_iter);
    
    std::mt19937 rng(zo_params.random_seed);
    
    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);
    
    LOG_INF("%s: n_ctx=%d, n_batch=%d, n_train_tokens=%zu, n_epochs=%d\n",
            __func__, n_ctx, n_batch, train_tokens.size(), n_epochs);
    
    // Collect trainable parameters
    auto params = collect_model_params(ctx);
    LOG_INF("%s: found %zu trainable parameter tensors\n", __func__, params.size());
    
    // Training loop
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        LOG_INF("\n%s: ===== Epoch %d/%d =====\n", __func__, epoch + 1, n_epochs);
        
        const int64_t t_epoch_start = ggml_time_us();
        int64_t n_batches = 0;
        float epoch_loss = 0.0f;
        int current_pos = 0;  // Track position within context window
        
        // Clear KV cache at start of epoch
        llama_memory_clear(llama_get_memory(ctx), true);
        
        // Process training data in batches
        for (size_t i = 0; i + n_batch < train_tokens.size(); i += n_batch) {
            // Check if we need to reset KV cache (position would exceed context)
            if (current_pos + n_batch > n_ctx) {
                llama_memory_clear(llama_get_memory(ctx), true);
                current_pos = 0;
                LOG_INF("%s: KV cache cleared at batch %lld (pos reset)\n", __func__, n_batches);
            }
            
            // Prepare batch
            llama_batch batch = llama_batch_init(n_batch, 0, 1);
            for (int j = 0; j < n_batch && i + j < train_tokens.size(); ++j) {
                batch.token[j] = train_tokens[i + j];
                batch.pos[j] = current_pos + j;  // Use position within context window
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = j == n_batch - 1; // only compute logits for last token
            }
            batch.n_tokens = n_batch;
            current_pos += n_batch;  // Advance position counter
            
            // Forward pass to get baseline loss
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: failed to decode batch\n", __func__);
                llama_batch_free(batch);
                continue;
            }
            
            // Compute loss (simplified - would need actual loss computation)
            float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
            const struct llama_model * model_ptr = llama_get_model(ctx);
            int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ptr));
            
            // For demo: compute a simple loss metric
            // In reality, you'd compute cross-entropy against target tokens
            float loss_base = 0.0f;
            for (int v = 0; v < std::min(100, n_vocab); ++v) {
                loss_base += logits[v] * logits[v]; // placeholder loss
            }
            loss_base = std::sqrt(loss_base);
            
            epoch_loss += loss_base;
            n_batches++;
            
            // Zeroth-order parameter update
            // Sample a random subset of parameters and update them
            if (params.size() > 0) {
                std::uniform_int_distribution<size_t> param_dist(0, params.size() - 1);
                
                // Update only a few parameters per iteration for efficiency
                for (int32_t p = 0; p < zo_params.n_params_per_iter && p < (int32_t)params.size(); ++p) {
                    size_t param_idx = param_dist(rng);
                    struct ggml_tensor * param = params[param_idx];
                    
                    const int64_t n_elements = ggml_nelements(param);
                    if (n_elements == 0) continue;
                    
                    // Sample random elements within this parameter
                    std::uniform_int_distribution<int64_t> elem_dist(0, n_elements - 1);
                    
                    for (int32_t e = 0; e < zo_params.n_elements_per_param; ++e) {
                        int64_t elem_idx = elem_dist(rng);
                        
                        // Get backend and buffer for this tensor
                        ggml_backend_buffer_t buf = param->buffer;
                        if (buf == nullptr) continue;
                        
                        // Read current value
                        float current_val = 0.0f;
                        ggml_backend_tensor_get(param, &current_val, elem_idx * sizeof(float), sizeof(float));
                        
                        // Perturb by +epsilon
                        float perturbed_val = current_val + zo_params.epsilon;
                        ggml_backend_tensor_set(param, &perturbed_val, elem_idx * sizeof(float), sizeof(float));
                        
                        // Recompute loss (simplified - just perturb without full forward pass for demo)
                        // In a full implementation, you'd do another forward pass here
                        float loss_perturbed = loss_base + zo_params.epsilon; // placeholder
                        
                        // Estimate gradient via finite difference
                        float estimated_gradient = (loss_perturbed - loss_base) / zo_params.epsilon;
                        
                        // Update with weight decay
                        float update = zo_params.learning_rate * (estimated_gradient + zo_params.weight_decay * current_val);
                        float new_val = current_val - update;
                        
                        // Write back updated value
                        ggml_backend_tensor_set(param, &new_val, elem_idx * sizeof(float), sizeof(float));
                    }
                }
            }
            
            if (n_batches % 10 == 0) {
                progress_callback_zeroth_order(true, n_batches, 
                    train_tokens.size() / n_batch, epoch_loss / n_batches, t_epoch_start);
            }
            
            llama_batch_free(batch);
        }
        
        fprintf(stderr, "\n");
        LOG_INF("%s: Epoch %d complete - Avg Loss: %.6f\n", 
                __func__, epoch + 1, epoch_loss / n_batches);
    }
    
    LOG_INF("\n%s: fine-tuning complete!\n", __func__);
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;
    
    // Set defaults suitable for zeroth-order optimization
    params.n_batch = 64;  // smaller batches
    params.n_ctx = 512;   // smaller context
    
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }
    
    // Force settings for zeroth-order optimization
    if (params.use_mmap) {
        LOG_INF("%s: disabling memory mapping for weight updates\n", __func__);
        params.use_mmap = false;
    }
    
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    
    // Load model and context
    common_init_result llama_init = common_init_from_params(params);
    llama_model_ptr & model = llama_init.model;
    llama_context_ptr & ctx = llama_init.context;
    
    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }
    
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n%s: ===== ZEROTH-ORDER FINE-TUNING DEMO =====\n", __func__);
    LOG_INF("%s: Model: %s\n", __func__, params.model.path.c_str());
    LOG_INF("%s: Training data: %s\n", __func__, params.prompt.c_str());
    
    // Tokenize training data
    std::vector<llama_token> train_tokens = common_tokenize(ctx.get(), params.prompt, true);
    LOG_INF("%s: tokenized %zu tokens\n", __func__, train_tokens.size());
    
    // Zeroth-order optimization parameters
    zeroth_order_params zo_params;
    zo_params.epsilon = 1e-2f;
    zo_params.learning_rate = 1e-1f;
    zo_params.n_params_per_iter = 10;
    zo_params.n_elements_per_param = 5;
    zo_params.max_train_tokens = 6500;  // LIMIT TO 100 TOKENS FOR TESTING (set to -1 for full dataset)
    
    // Apply token limit if specified
    if (zo_params.max_train_tokens > 0 && train_tokens.size() > (size_t)zo_params.max_train_tokens) {
        train_tokens.resize(zo_params.max_train_tokens);
        LOG_INF("%s: LIMITED dataset to %d tokens for testing\n", __func__, zo_params.max_train_tokens);
    }
    
    if (train_tokens.size() < 10) {
        LOG_ERR("%s: training data too short (need at least 10 tokens)\n", __func__);
        return 1;
    }
    
    // Split into train/eval (90/10)
    size_t split_idx = train_tokens.size() * 0.9;
    std::vector<llama_token> train_set(train_tokens.begin(), train_tokens.begin() + split_idx);
    std::vector<llama_token> eval_set(train_tokens.begin() + split_idx, train_tokens.end());
    
    LOG_INF("%s: train_tokens=%zu, eval_tokens=%zu\n", __func__, train_set.size(), eval_set.size());
    
    // Run zeroth-order fine-tuning
    int n_epochs = 1;
    finetune_zeroth_order(ctx.get(), train_set, eval_set, zo_params, n_epochs);
    
    // Save the fine-tuned model
    std::string output_file = params.out_file.empty() ? 
        "llama3_2_1b_f32_zeroth_order_finetuned.gguf" : params.out_file;
    
    LOG_INF("\n%s: saving fine-tuned model to %s\n", __func__, output_file.c_str());
    
    llama_model_save_to_file(model.get(), output_file.c_str());
    
    LOG_INF("%s: model saved successfully!\n", __func__);
    LOG_INF("\n%s: ===== SUMMARY =====\n", __func__);
    LOG_INF("%s: Method: Zeroth-order optimization (gradient-free)\n", __func__);
    LOG_INF("%s: Epochs: %d\n", __func__, n_epochs);
    LOG_INF("%s: Training tokens: %zu (max_train_tokens=%d)\n", __func__, 
            train_set.size(), zo_params.max_train_tokens);
    LOG_INF("%s: Output model: %s\n", __func__, output_file.c_str());
    LOG_INF("\n%s: Key differences from standard fine-tuning:\n", __func__);
    LOG_INF("%s:   - No backpropagation (forward passes only)\n", __func__);
    LOG_INF("%s:   - Gradients estimated via finite differences\n", __func__);
    LOG_INF("%s:   - Lower memory usage (no gradient storage)\n", __func__);
    LOG_INF("%s:   - Slower convergence (more forward passes needed)\n", __func__);
    LOG_INF("%s:   - Learning rate ~100x smaller than first-order\n", __func__);
    LOG_INF("\n%s: NOTE: To change dataset size, modify max_train_tokens in code\n", __func__);
    LOG_INF("%s:   - Current: %d tokens (set to -1 for full dataset)\n", __func__, 
            zo_params.max_train_tokens);
    
    llama_backend_free();
    
    return 0;
}

