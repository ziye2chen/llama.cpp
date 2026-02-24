// Verify Parameter Update and Save
// Minimal program to verify that modifying model tensors and saving
// produces a different model (verifiable via perplexity).
//
// Usage:
//   ./verify-param-update -m model.gguf -o model_modified.gguf
//   ./llama-perplexity -m model.gguf -f wiki.test.raw
//   ./llama-perplexity -m model_modified.gguf -f wiki.test.raw
// Compare perplexity - modified model should differ.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

// Dequantize tensor to FP32 (for quantized types)
static bool dequantize_to_fp32(struct ggml_tensor * t, const std::vector<uint8_t> & quant,
                               std::vector<float> & fp32) {
    const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) return false;

    const int64_t nrows = ggml_nrows(t);
    const int64_t n_per_row = t->ne[0];
    const size_t row_size = ggml_row_size(t->type, n_per_row);

    fp32.resize(ggml_nelements(t));
    for (int64_t r = 0; r < nrows; ++r) {
        traits->to_float(quant.data() + r * row_size, fp32.data() + r * n_per_row, n_per_row);
    }
    return true;
}

// Requantize FP32 to bytes
static bool requantize_to_bytes(struct ggml_tensor * t, const std::vector<float> & fp32,
                               std::vector<uint8_t> & quant) {
    const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->from_float_ref) return false;

    const int64_t nrows = ggml_nrows(t);
    const int64_t n_per_row = t->ne[0];
    const size_t row_size = ggml_row_size(t->type, n_per_row);

    quant.resize(ggml_nbytes(t));
    for (int64_t r = 0; r < nrows; ++r) {
        traits->from_float_ref(fp32.data() + r * n_per_row, quant.data() + r * row_size, n_per_row);
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;
    params.n_batch = 512;
    params.n_ctx = 512;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // CRITICAL: Disable mmap so tensors are in writable memory
    if (params.use_mmap) {
        LOG_INF("%s: disabling mmap for parameter modification\n", __func__);
        params.use_mmap = false;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result init = common_init_from_params(params);
    if (!init.model) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    LOG_INF("\n%s: ===== VERIFY PARAMETER UPDATE =====\n", __func__);
    LOG_INF("%s: Model: %s\n", __func__, params.model.path.c_str());

    const llama_model * model = init.model.get();
    const size_t n_tensors = llama_model_n_tensors(model);

    // Find first trainable weight tensor
    struct ggml_tensor * target = nullptr;
    char tname[256];
    for (size_t i = 0; i < n_tensors; ++i) {
        struct ggml_tensor * t = llama_model_get_tensor_by_index(model, i, tname, sizeof(tname));
        if (!t || !t->buffer) continue;
        if (strstr(tname, "attn_q") || strstr(tname, "attn_k") || strstr(tname, ".wq") || strstr(tname, ".w1")) {
            target = t;
            break;
        }
    }

    if (!target) {
        LOG_ERR("%s: no suitable tensor found\n", __func__);
        return 1;
    }

    LOG_INF("%s: Modifying tensor: %s (shape [%ld,%ld], type %s)\n", __func__,
           target->name, target->ne[0], target->ne[1], ggml_type_name(target->type));

    const float perturbation = 1e-3f;  // Small but visible change

    if (ggml_is_quantized(target->type)) {
        std::vector<uint8_t> quant(ggml_nbytes(target));
        ggml_backend_tensor_get(target, quant.data(), 0, quant.size());
        llama_synchronize(init.context.get());

        std::vector<float> fp32;
        if (!dequantize_to_fp32(target, quant, fp32)) {
            LOG_ERR("%s: dequantize failed\n", __func__);
            return 1;
        }

        // Add perturbation to first 100 elements
        for (size_t i = 0; i < fp32.size() && i < 100; ++i) {
            fp32[i] += perturbation;
        }

        std::vector<uint8_t> quant_new;
        if (!requantize_to_bytes(target, fp32, quant_new)) {
            LOG_ERR("%s: requantize failed\n", __func__);
            return 1;
        }

        ggml_backend_tensor_set(target, quant_new.data(), 0, quant_new.size());
    } else {
        std::vector<float> data(ggml_nelements(target));
        ggml_backend_tensor_get(target, data.data(), 0, data.size() * sizeof(float));
        llama_synchronize(init.context.get());

        for (size_t i = 0; i < data.size() && i < 100; ++i) {
            data[i] += perturbation;
        }

        ggml_backend_tensor_set(target, data.data(), 0, data.size() * sizeof(float));
    }
    llama_synchronize(init.context.get());

    LOG_INF("%s: Applied perturbation %.2e to first 100 elements\n", __func__, perturbation);

    std::string out_path = params.out_file.empty() ? "model_modified.gguf" : params.out_file;
    LOG_INF("%s: Saving to %s\n", __func__, out_path.c_str());

    // Ensure all backend writes are visible before save (critical for GPU)
    llama_synchronize(init.context.get());
    llama_model_save_to_file(init.model.get(), out_path.c_str());

    LOG_INF("%s: Done. Verify with:\n", __func__);
    LOG_INF("%s:   ./llama-perplexity -m %s -f wiki.test.raw\n", __func__, params.model.path.c_str());
    LOG_INF("%s:   ./llama-perplexity -m %s -f wiki.test.raw\n", __func__, out_path.c_str());
    LOG_INF("%s: Perplexity should differ if save works.\n", __func__);

    llama_backend_free();
    return 0;
}
