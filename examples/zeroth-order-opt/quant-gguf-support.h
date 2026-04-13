#pragma once

#include "ggml.h"

#include <array>

namespace radazo_quant_support {

struct verified_quant_type {
    ggml_type type;
    const char * label;
};

inline constexpr std::array<verified_quant_type, 4> k_verified_quant_types = {{
    { GGML_TYPE_Q4_K, "Q4_K_M" },
    { GGML_TYPE_Q5_0, "Q5_0" },
    { GGML_TYPE_Q5_K, "Q5_K_M" },
    { GGML_TYPE_Q6_K, "Q6_K" },
}};

inline bool is_verified_quant_type(ggml_type type) {
    for (const auto & entry : k_verified_quant_types) {
        if (entry.type == type) {
            return true;
        }
    }
    return false;
}

inline const char * verified_quant_type_name(ggml_type type) {
    for (const auto & entry : k_verified_quant_types) {
        if (entry.type == type) {
            return entry.label;
        }
    }
    return nullptr;
}

inline const char * display_quant_type_name(ggml_type type) {
    const char * verified = verified_quant_type_name(type);
    return verified ? verified : ggml_type_name(type);
}

inline bool has_requant_traits(ggml_type type) {
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits && traits->to_float && traits->from_float_ref;
}

} // namespace radazo_quant_support
