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



