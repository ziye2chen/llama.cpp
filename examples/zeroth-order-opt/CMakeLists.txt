# Zeroth-Order Optimization Examples

# Test program
set(TARGET1 zeroth-order-test)
add_executable(${TARGET1} zeroth-order-test.cpp)
target_link_libraries(${TARGET1} PRIVATE ggml llama common)
target_compile_features(${TARGET1} PRIVATE cxx_std_11)

# Fine-tuning demo
set(TARGET2 finetune-zeroth-order)
add_executable(${TARGET2} 
    finetune-zeroth-order.cpp
    zeroth-order-optimizer.cpp
    zeroth-order-optimizer.h
)
target_link_libraries(${TARGET2} PRIVATE ggml llama common)
target_compile_features(${TARGET2} PRIVATE cxx_std_11)
target_include_directories(${TARGET2} PRIVATE ${CMAKE_SOURCE_DIR}/src)

# R-AdaZO fine-tuning demo
set(TARGET3 finetune-radazo)
add_executable(${TARGET3}
    finetune-radazo.cpp
    radazo-optimizer.cpp
    radazo-optimizer.h
)
target_link_libraries(${TARGET3} PRIVATE ggml llama common)
target_compile_features(${TARGET3} PRIVATE cxx_std_11)
target_include_directories(${TARGET3} PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/nlohmann
)

# R-AdaZO fine-tuning for quantized models (direct dequant/requant)
set(TARGET4 finetune-radazo-quant)
add_executable(${TARGET4}
    finetune-radazo-quant.cpp
    radazo-optimizer.cpp
    radazo-optimizer.h
)
target_link_libraries(${TARGET4} PRIVATE ggml llama common)
target_compile_features(${TARGET4} PRIVATE cxx_std_11)
target_include_directories(${TARGET4} PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/vendor/nlohmann
)

# Verify parameter update and save (test that tensor modifications persist)
set(TARGET5 verify-param-update)
add_executable(${TARGET5} verify-param-update.cpp)
target_link_libraries(${TARGET5} PRIVATE ggml llama common)
target_compile_features(${TARGET5} PRIVATE cxx_std_11)


