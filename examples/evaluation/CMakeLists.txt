# GSM8K Evaluation

set(TARGET gsm8k-eval)
add_executable(${TARGET} 
    gsm8k-eval.cpp
)
target_link_libraries(${TARGET} PRIVATE ggml llama common)
target_compile_features(${TARGET} PRIVATE cxx_std_11)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)

