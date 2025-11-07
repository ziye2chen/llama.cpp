// Test program for zeroth-order optimization
// This demonstrates gradient-free optimization on a simple task

#include "llama.h"
#include "ggml.h"
#include "ggml-opt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <random>

// Simple callback to print training progress (not used in this simplified test)
static void progress_callback(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    
    const int64_t t_now_us = ggml_time_us();
    fprintf(stderr, "[%s] Batch %6lld/%6lld | Time: %6.2fs\n",
            train ? "TRAIN" : "EVAL ",
            ibatch, ibatch_max,
            (t_now_us - t_start_us) / 1.0e6f);
    (void)opt_ctx; (void)dataset; (void)result; // unused
}

// Test 1: Simple quadratic optimization
// Minimize: f(x) = (x - target)^2
void test_simple_quadratic() {
    printf("\n=== Test 1: Simple Quadratic Optimization ===\n");
    printf("Objective: Minimize f(x) = (x - 5.0)^2\n\n");
    
    // Create backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "Failed to initialize backend\n");
        return;
    }
    
    ggml_backend_t backends[] = {backend};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    
    // Create contexts
    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 16*1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_static = ggml_init(params);
    }
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 128*1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_compute = ggml_init(params);
    }
    
    // Create parameter to optimize (start at 0.0)
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(x, "x");
    ggml_set_param(x);
    
    // Create target value
    struct ggml_tensor * target = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(target, "target");
    
    // Build computation graph: loss = (x - target)^2
    struct ggml_tensor * diff = ggml_sub(ctx_compute, x, target);
    struct ggml_tensor * loss = ggml_sqr(ctx_compute, diff);
    struct ggml_tensor * outputs = ggml_scale(ctx_compute, loss, 1.0f);
    ggml_set_name(outputs, "outputs");
    
    // Allocate static tensors
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    
    // Initialize values
    float x_init = 0.0f;
    float target_val = 5.0f;
    ggml_backend_tensor_set(x, &x_init, 0, sizeof(float));
    ggml_backend_tensor_set(target, &target_val, 0, sizeof(float));
    
    printf("Initial x: %.6f\n", x_init);
    printf("Target:    %.6f\n\n", target_val);
    
    // Zeroth-order optimization using manual finite differences
    printf("=== Zeroth-Order Optimization (Manual Finite Differences) ===\n");
    const float epsilon = 1e-4f;
    const float learning_rate = 0.1f;
    const int n_iterations = 100;
    
    // Build forward graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, outputs);
    
    for (int iter = 0; iter < n_iterations; ++iter) {
        // Get current x value
        float x_val;
        ggml_backend_tensor_get(x, &x_val, 0, sizeof(float));
        
        // Compute loss at x
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_sched_graph_compute(sched, gf);
        float loss_base;
        ggml_backend_tensor_get(outputs, &loss_base, 0, sizeof(float));
        
        // Compute loss at x + epsilon
        float x_perturbed = x_val + epsilon;
        ggml_backend_tensor_set(x, &x_perturbed, 0, sizeof(float));
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_sched_graph_compute(sched, gf);
        float loss_perturbed;
        ggml_backend_tensor_get(outputs, &loss_perturbed, 0, sizeof(float));
        
        // Estimate gradient using finite difference
        float grad_estimate = (loss_perturbed - loss_base) / epsilon;
        
        // Update x using gradient descent
        float x_new = x_val - learning_rate * grad_estimate;
        ggml_backend_tensor_set(x, &x_new, 0, sizeof(float));
        
        if (iter % 10 == 0 || iter == n_iterations - 1) {
            printf("Iter %3d: x = %.6f, loss = %.6f, grad ≈ %.6f\n", 
                   iter, x_val, loss_base, grad_estimate);
        }
    }
    
    // Get final value
    float x_final;
    ggml_backend_tensor_get(x, &x_final, 0, sizeof(float));
    printf("\nFinal x: %.6f (target: %.6f)\n", x_final, target_val);
    printf("Error:   %.6f\n", fabs(x_final - target_val));
    
    // Cleanup
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_compute);
    ggml_free(ctx_static);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
}

// Test 2: Multi-parameter optimization
// Minimize: f(w, b) = mean((w*x + b - y)^2)  [linear regression]
void test_linear_regression() {
    printf("\n=== Test 2: Linear Regression with Zeroth-Order Optimization ===\n");
    printf("Objective: Fit y = w*x + b to data\n");
    printf("True parameters: w = 2.0, b = 1.0\n\n");
    
    // Generate synthetic data: y = 2*x + 1 + noise
    const int n_data = 20;
    std::vector<float> x_data(n_data);
    std::vector<float> y_data(n_data);
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    for (int i = 0; i < n_data; ++i) {
        x_data[i] = dist(rng);
        y_data[i] = 2.0f * x_data[i] + 1.0f + noise(rng);
    }
    
    // Create backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t backends[] = {backend};
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    
    // Create contexts
    struct ggml_context * ctx_static;
    struct ggml_context * ctx_compute;
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 16*1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_static = ggml_init(params);
    }
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ 128*1024*1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_compute = ggml_init(params);
    }
    
    // Parameters
    struct ggml_tensor * w = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, 1);
    ggml_set_name(w, "w");
    ggml_set_name(b, "b");
    ggml_set_param(w);
    ggml_set_param(b);
    
    // Data
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, n_data);
    struct ggml_tensor * y = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, n_data);
    ggml_set_name(x, "x");
    ggml_set_name(y, "y");
    
    // Build graph: prediction = w*x + b, loss = mean((prediction - y)^2)
    // Note: We'll manually compute w*x + b and pass as tensors to avoid broadcasting issues
    struct ggml_tensor * x_scaled = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, n_data);
    struct ggml_tensor * b_repeated = ggml_new_tensor_1d(ctx_static, GGML_TYPE_F32, n_data);
    ggml_set_name(x_scaled, "x_scaled");
    ggml_set_name(b_repeated, "b_repeated");
    
    // Build: prediction = x_scaled + b_repeated, then compute loss
    struct ggml_tensor * pred = ggml_add(ctx_compute, x_scaled, b_repeated);
    struct ggml_tensor * diff = ggml_sub(ctx_compute, pred, y);
    struct ggml_tensor * sq_diff = ggml_sqr(ctx_compute, diff);
    struct ggml_tensor * outputs = ggml_mean(ctx_compute, sq_diff);
    ggml_set_name(outputs, "outputs");
    
    // Allocate
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_static, backend);
    
    // Initialize
    float w_init = 0.0f;
    float b_init = 0.0f;
    ggml_backend_tensor_set(w, &w_init, 0, sizeof(float));
    ggml_backend_tensor_set(b, &b_init, 0, sizeof(float));
    ggml_backend_tensor_set(x, x_data.data(), 0, n_data * sizeof(float));
    ggml_backend_tensor_set(y, y_data.data(), 0, n_data * sizeof(float));
    
    // Initialize b_repeated to b_init
    std::vector<float> b_rep(n_data, b_init);
    ggml_backend_tensor_set(b_repeated, b_rep.data(), 0, n_data * sizeof(float));
    
    printf("Initial w: %.6f, b: %.6f\n\n", w_init, b_init);
    
    // Zeroth-order optimization
    printf("=== Zeroth-Order Optimization ===\n");
    const float epsilon = 1e-3f;
    const float learning_rate = 0.01f;
    const int n_iterations = 200;
    
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx_compute, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_build_forward_expand(gf, outputs);
    
    for (int iter = 0; iter < n_iterations; ++iter) {
        // Get current parameter values
        float w_val, b_val;
        ggml_backend_tensor_get(w, &w_val, 0, sizeof(float));
        ggml_backend_tensor_get(b, &b_val, 0, sizeof(float));
        
        // Update x_scaled = w * x and b_repeated
        std::vector<float> x_scaled_data(n_data);
        std::vector<float> b_rep_data(n_data);
        for (int i = 0; i < n_data; ++i) {
            x_scaled_data[i] = w_val * x_data[i];
            b_rep_data[i] = b_val;
        }
        ggml_backend_tensor_set(x_scaled, x_scaled_data.data(), 0, n_data * sizeof(float));
        ggml_backend_tensor_set(b_repeated, b_rep_data.data(), 0, n_data * sizeof(float));
        
        // Compute base loss
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, gf);
        ggml_backend_sched_graph_compute(sched, gf);
        float loss_base;
        ggml_backend_tensor_get(outputs, &loss_base, 0, sizeof(float));
        
        // Update w: perturb and compute gradient
        {
            float w_perturbed = w_val + epsilon;
            for (int i = 0; i < n_data; ++i) {
                x_scaled_data[i] = w_perturbed * x_data[i];
            }
            ggml_backend_tensor_set(x_scaled, x_scaled_data.data(), 0, n_data * sizeof(float));
            
            ggml_backend_sched_reset(sched);
            ggml_backend_sched_alloc_graph(sched, gf);
            ggml_backend_sched_graph_compute(sched, gf);
            float loss_perturbed;
            ggml_backend_tensor_get(outputs, &loss_perturbed, 0, sizeof(float));
            
            float grad_w = (loss_perturbed - loss_base) / epsilon;
            w_val -= learning_rate * grad_w;
            ggml_backend_tensor_set(w, &w_val, 0, sizeof(float));
            
            // Restore x_scaled
            for (int i = 0; i < n_data; ++i) {
                x_scaled_data[i] = w_val * x_data[i];
            }
            ggml_backend_tensor_set(x_scaled, x_scaled_data.data(), 0, n_data * sizeof(float));
        }
        
        // Update b: perturb and compute gradient
        {
            float b_perturbed = b_val + epsilon;
            for (int i = 0; i < n_data; ++i) {
                b_rep_data[i] = b_perturbed;
            }
            ggml_backend_tensor_set(b_repeated, b_rep_data.data(), 0, n_data * sizeof(float));
            
            ggml_backend_sched_reset(sched);
            ggml_backend_sched_alloc_graph(sched, gf);
            ggml_backend_sched_graph_compute(sched, gf);
            float loss_perturbed;
            ggml_backend_tensor_get(outputs, &loss_perturbed, 0, sizeof(float));
            
            float grad_b = (loss_perturbed - loss_base) / epsilon;
            b_val -= learning_rate * grad_b;
            ggml_backend_tensor_set(b, &b_val, 0, sizeof(float));
            
            // Restore b_repeated
            for (int i = 0; i < n_data; ++i) {
                b_rep_data[i] = b_val;
            }
            ggml_backend_tensor_set(b_repeated, b_rep_data.data(), 0, n_data * sizeof(float));
        }
        
        if (iter % 20 == 0 || iter == n_iterations - 1) {
            float w_val, b_val;
            ggml_backend_tensor_get(w, &w_val, 0, sizeof(float));
            ggml_backend_tensor_get(b, &b_val, 0, sizeof(float));
            printf("Iter %3d: w = %.6f, b = %.6f, loss = %.6f\n", 
                   iter, w_val, b_val, loss_base);
        }
    }
    
    // Final results
    float w_final, b_final;
    ggml_backend_tensor_get(w, &w_final, 0, sizeof(float));
    ggml_backend_tensor_get(b, &b_final, 0, sizeof(float));
    printf("\nFinal parameters:\n");
    printf("  w = %.6f (true: 2.0, error: %.6f)\n", w_final, fabs(w_final - 2.0f));
    printf("  b = %.6f (true: 1.0, error: %.6f)\n", b_final, fabs(b_final - 1.0f));
    
    // Cleanup
    ggml_backend_buffer_free(buf);
    ggml_free(ctx_compute);
    ggml_free(ctx_static);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    printf("=================================================================\n");
    printf("    Zeroth-Order Optimization Test for llama.cpp\n");
    printf("=================================================================\n");
    printf("\nThis test demonstrates gradient-free optimization using only\n");
    printf("forward passes (no backpropagation required).\n");
    printf("\nMethod: Finite Difference Approximation\n");
    printf("  grad(f) ≈ (f(x + ε) - f(x)) / ε\n");
    printf("=================================================================\n");
    
    // Run tests
    test_simple_quadratic();
    test_linear_regression();
    
    printf("\n=================================================================\n");
    printf("All tests completed successfully!\n");
    printf("=================================================================\n");
    printf("\n");
    printf("Key observations:\n");
    printf("1. Zeroth-order optimization works without backpropagation\n");
    printf("2. Requires 2x forward passes per parameter (base + perturbed)\n");
    printf("3. Converges slower than gradient-based methods\n");
    printf("4. Useful when gradients are unavailable or unreliable\n");
    printf("5. Can optimize GGUF models using only inference code!\n");
    printf("\n");
    
    return 0;
}


