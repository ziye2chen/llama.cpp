# LoRA Training for Quantized Models - Quick Start

## 概述

本项目实现了对 Q4_K_M 量化 GGUF 模型的 LoRA (Low-Rank Adaptation) 训练框架。基座模型保持冻结，只训练低秩适配器。

## 已实现的功能

✅ **LoRA 适配器结构**
- LoRA A 和 B 矩阵的创建和初始化
- 目标层的自动识别和适配器创建
- 参数收集用于训练

✅ **训练框架**
- 与 R-AdaZO 优化器的集成
- 训练循环结构
- GSM8K 数据集支持

✅ **构建系统**
- CMake 配置完成
- 可编译生成 `finetune-radazo-quant` 可执行文件

## 编译

```bash
cd /home/xplor/llama.cpp/build
cmake --build . --target finetune-radazo-quant
```

## 使用

```bash
./bin/finetune-radazo-quant \
  -m /path/to/model-Q4_K_M.gguf \
  -f examples/zeroth-order-opt/gsm8k_test.jsonl \
  -o lora_adapter.bin \
  --epochs 1
```

## 重要说明

### ⚠️ 当前限制

1. **前向传播集成**：LoRA 输出尚未完全集成到模型前向传播中
   - 需要修改 `llama.cpp` 的前向传播逻辑
   - 或实现自定义前向传播函数

2. **解量化**：当前使用占位符函数
   - 需要实现真正的 Q4_K_M 解量化

3. **保存/加载**：LoRA 适配器的序列化尚未完成
   - 需要实现二进制格式的保存/加载

### 🔧 后续工作

1. **完善前向传播**
   - 在计算图中集成 LoRA 计算
   - 实现 `Y = Base(X) + LoRA(X)`

2. **实现解量化**
   - 使用 `ggml_dequantize_row_q4_k` 等函数
   - 正确处理 Q4_K_M 格式

3. **完成序列化**
   - 实现 LoRA 适配器的保存/加载
   - 包含元数据（rank, alpha, 层名等）

4. **创建合并工具**
   - 将训练好的 LoRA 与基座模型合并
   - 重新量化为 Q4_K_M 格式

## 架构说明

```
基座模型 (GGUF Q4_K_M, 冻结)
    ↓ (运行时解量化)
LoRA 适配器 (FP32, 可训练)
    ↓ (前向传播)
输出 = 基座输出 + LoRA 输出
```

## 文件结构

- `lora-adapter.h/cpp`: LoRA 适配器实现
- `finetune-radazo-quant.cpp`: 主训练程序
- `LORA_QUANT_IMPLEMENTATION.md`: 详细实现文档
- `CMakeLists.txt`: 构建配置（已更新）

## 参考

- LoRA 论文: "LoRA: Low-Rank Adaptation of Large Language Models"
- R-AdaZO 论文: "Refining Adaptive Zeroth-Order Optimization at Ease"
