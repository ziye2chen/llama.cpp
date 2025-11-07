// Zeroth-Order Optimization Header for llama.cpp
// This provides gradient-free optimization using finite differences

#pragma once

#include "llama-impl.h"
#include "ggml-opt.h"
#include <vector>

// Forward declarations
struct llama_context;
struct llama_token;
struct ggml_opt_dataset;
struct ggml_opt_result;
struct llama_batch;

// Zeroth-order optimization parameters
struct llama_zeroth_order_params {
    float epsilon = 1e-4f;           // perturbation size for finite differences
    float learning_rate = 1e-5f;     // step size for parameter updates
    float weight_decay = 0.01f;      // L2 regularization coefficient
    int32_t n_params_per_iter = 100; // number of parameters to perturb per iteration
    bool use_random_sampling = true;  // whether to randomly sample parameters
    uint32_t random_seed = 12345;    // random seed for reproducibility
};

// Get default zeroth-order optimization parameters
static inline llama_zeroth_order_params llama_zeroth_order_default_params() {
    llama_zeroth_order_params params;
    params.epsilon = 1e-4f;
    params.learning_rate = 1e-5f;
    params.weight_decay = 0.01f;
    params.n_params_per_iter = 100;
    params.use_random_sampling = true;
    params.random_seed = 12345;
    return params;
}



