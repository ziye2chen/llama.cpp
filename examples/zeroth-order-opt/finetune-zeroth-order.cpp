// Zeroth-Order Fine-tuning Demo for GGUF Models
// This demonstrates gradient-free optimization for fine-tuning

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "zeroth-order-optimizer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

// Progress callback for zeroth-order optimization
static void progress_callback_zeroth_order(
        bool train,
        int64_t iter,
        int64_t iter_max,
        float loss,
        int64_t t_start_us) {
    const int64_t t_now_us = ggml_time_us();
    const float elapsed = (t_now_us - t_start_us) / 1.0e6f;
    
    fprintf(stderr, "\r[%s] Iter %6ld/%6ld | Loss: %.6f | Time: %6.2fs | %.1f it/s",
            train ? "TRAIN" : "EVAL ",
            iter, iter_max, loss, elapsed,
            iter > 0 ? iter / elapsed : 0.0f);
    fflush(stderr);
}

// Training configuration
struct training_config {
    int32_t max_train_tokens = -1;  // if >0, limit dataset to this many tokens
};

// Zeroth-order fine-tuning loop using the optimizer
static void finetune_zeroth_order(
        struct llama_context * ctx,
        const std::vector<llama_token> & train_tokens,
        const std::vector<llama_token> & eval_tokens,
        ZerothOrderOptimizer & optimizer,
        int n_epochs) {
    
    LOG_INF("\n%s: starting zeroth-order fine-tuning (REAL gradient estimation)...\n", __func__);
    
    const int n_ctx = llama_n_ctx(ctx);
    const int n_batch = llama_n_batch(ctx);
    const struct llama_model * model_ptr = llama_get_model(ctx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_ptr));
    
    LOG_INF("%s: n_ctx=%d, n_batch=%d, n_train_tokens=%zu, n_epochs=%d\n",
            __func__, n_ctx, n_batch, train_tokens.size(), n_epochs);
    
    // Training loop
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        LOG_INF("\n%s: ===== Epoch %d/%d =====\n", __func__, epoch + 1, n_epochs);
        
        const int64_t t_epoch_start = ggml_time_us();
        int64_t n_batches = 0;
        float epoch_loss = 0.0f;
        
        // Process training data in batches
        for (size_t i = 0; i + n_batch < train_tokens.size(); i += n_batch) {
            // Clear KV cache at start of each batch
            llama_memory_clear(llama_get_memory(ctx), true);
            
            // Prepare batch
            llama_batch batch = llama_batch_init(n_batch, 0, 1);
            for (int j = 0; j < n_batch && i + j < train_tokens.size(); ++j) {
                batch.token[j] = train_tokens[i + j];
                batch.pos[j] = j;  // Position starts from 0 for each batch
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = (j == n_batch - 1); // only compute logits for last token
            }
            batch.n_tokens = n_batch;
            
            // Forward pass to get baseline loss
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("%s: failed to decode batch\n", __func__);
                llama_batch_free(batch);
                continue;
            }
            
            // *** USE THE OPTIMIZER: Single call performs all gradient estimation and updates ***
            float batch_loss = optimizer.step(ctx, batch, n_vocab);
            
            epoch_loss += batch_loss;
            n_batches++;
            
            // Progress reporting
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
    
    // Print optimizer statistics
    LOG_INF("\n%s: Optimizer Statistics:\n", __func__);
    LOG_INF("%s:   Total parameter updates: %ld\n", __func__, optimizer.get_total_updates());
    LOG_INF("%s:   Total forward passes: %ld\n", __func__, optimizer.get_forward_passes());
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
    
    // Training configuration
    training_config train_config;
    train_config.max_train_tokens = 1000;  // LIMIT TO 100 TOKENS FOR TESTING (set to -1 for full dataset)
    
    // Apply token limit if specified
    if (train_config.max_train_tokens > 0 && train_tokens.size() > (size_t)train_config.max_train_tokens) {
        train_tokens.resize(train_config.max_train_tokens);
        LOG_INF("%s: LIMITED dataset to %d tokens for testing\n", __func__, train_config.max_train_tokens);
    }
    
    if (train_tokens.size() < 10) {
        LOG_ERR("%s: training data too short (need at least 10 tokens)\n", __func__);
        return 1;
    }
    
    // Split into train/eval (90/10)
    size_t split_idx = train_tokens.size() * 0.9;
    std::vector<llama_token> train_set(train_tokens.begin(), train_tokens.begin() + split_idx);
    std::vector<llama_token> eval_set(train_tokens.begin() + split_idx, train_tokens.end());
    
    LOG_INF("%s: train_tokens=%zu, eval_tokens=%zu\n\n", __func__, train_set.size(), eval_set.size());
    
    // *** SETUP THE OPTIMIZER (plug-and-play!) ***
    
    // Step 1: Collect trainable parameters
    auto trainable_params = collect_trainable_parameters(ctx.get(), true);
    LOG_INF("%s: found %zu trainable parameter tensors\n\n", __func__, trainable_params.size());
    
    // Step 2: Configure optimizer parameters
    zeroth_order_params zo_params;
    zo_params.epsilon = 1e-3f;              // Perturbation size for gradient estimation
    zo_params.learning_rate = 1e-4f;        // Learning rate (tune based on loss behavior)
    zo_params.weight_decay = 0.01f;         // L2 regularization to prevent overfitting
    zo_params.n_params_per_iter = 3;        // Sample 3 parameters per batch
    zo_params.n_elements_per_param = 2;     // Sample 2 elements per parameter (6 forward passes/batch)
    zo_params.log_gradients = false;        // Set to true for detailed gradient logging
    
    LOG_INF("%s: Optimizer Configuration:\n", __func__);
    LOG_INF("%s:   epsilon = %.2e (perturbation size)\n", __func__, zo_params.epsilon);
    LOG_INF("%s:   learning_rate = %.2e\n", __func__, zo_params.learning_rate);
    LOG_INF("%s:   weight_decay = %.2e\n", __func__, zo_params.weight_decay);
    LOG_INF("%s:   n_params_per_iter = %d\n", __func__, zo_params.n_params_per_iter);
    LOG_INF("%s:   n_elements_per_param = %d\n", __func__, zo_params.n_elements_per_param);
    LOG_INF("%s:   Forward passes per batch = %d (1 baseline + %d perturbed)\n\n", 
            __func__, 1 + zo_params.n_params_per_iter * zo_params.n_elements_per_param,
            zo_params.n_params_per_iter * zo_params.n_elements_per_param);
    
    // Step 3: Create the optimizer
    ZerothOrderOptimizer optimizer(zo_params, trainable_params);
    
    // Step 4: Run fine-tuning
    int n_epochs = 1;
    finetune_zeroth_order(ctx.get(), train_set, eval_set, optimizer, n_epochs);
    
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
            train_set.size(), train_config.max_train_tokens);
    LOG_INF("%s: Output model: %s\n", __func__, output_file.c_str());
    LOG_INF("\n%s: Key differences from standard fine-tuning:\n", __func__);
    LOG_INF("%s:   - No backpropagation (forward passes only)\n", __func__);
    LOG_INF("%s:   - Gradients estimated via finite differences\n", __func__);
    LOG_INF("%s:   - Lower memory usage (no gradient storage)\n", __func__);
    LOG_INF("%s:   - Slower convergence (more forward passes needed)\n", __func__);
    LOG_INF("%s:   - ~7x slower per batch (multiple forward passes)\n", __func__);
    LOG_INF("\n%s: NOTE: To change dataset size, modify train_config.max_train_tokens in code\n", __func__);
    LOG_INF("%s:   - Current: %d tokens (set to -1 for full dataset)\n", __func__, 
            train_config.max_train_tokens);
    
    llama_backend_free();
    
    return 0;
}

