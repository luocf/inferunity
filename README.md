# InferUnity

高性能深度学习推理引擎，支持 ONNX 模型加载和推理。

## 特性

- ✅ 支持 ONNX 模型加载和推理
- ✅ 26个算子实现（数学运算、激活函数、卷积、归一化等）
- ✅ CPU 后端实现
- ✅ 内存池和生命周期优化
- ✅ MatMul 算子 BLAS 优化（Accelerate/OpenBLAS）
- ✅ 线程池支持
- ✅ 70%+ 测试覆盖率

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake .. -DUSE_BLAS=ON
make -j$(nproc)
```

### 基本使用

```cpp
#include "inferunity/runtime.h"

// 创建推理会话
auto session = std::make_unique<InferenceSession>();

// 加载模型
session->LoadModel("model.onnx");

// 准备输入
auto input = CreateTensor(Shape({1, 3, 224, 224}), DataType::FLOAT32);

// 运行推理
std::vector<Tensor*> inputs = {input.get()};
std::vector<Tensor*> outputs;
session->Run(inputs, outputs);
```

更多示例请参见 [用户指南](docs/USER_GUIDE.md)。

## 文档

- [推理引擎基础知识](docs/INFERENCE_ENGINE_BASICS.md) - 详细的推理引擎核心概念和实现详解（推荐阅读）
- [ONNX模型加载流程详解](docs/ONNX_MODEL_LOADING_EXPLAINED.md) - 从文件加载到Graph构建的完整流程（推荐学习）
- [推理执行流程详解](docs/INFERENCE_EXECUTION_FLOW.md) - 从Run()调用到输出结果的完整执行流程（推荐学习）
- [用户指南](docs/USER_GUIDE.md) - 快速开始和使用示例
- [API参考](docs/API_REFERENCE.md) - 完整的API文档
- [功能总结](docs/FEATURE_SUMMARY.md) - 所有已实现功能的详细总结
- [下一步计划](docs/NEXT_STEPS.md) - 后续开发计划
- [性能基准](docs/PERFORMANCE_BENCHMARKS.md) - 性能测试结果
- [项目状态](docs/PROJECT_STATUS.md) - 项目完成情况
- [完成总结](docs/FINAL_COMPLETION_SUMMARY.md) - 详细完成情况

## 支持的算子

### 数学运算
Add, Sub, Mul, Div, MatMul

### 激活函数
Relu, Sigmoid, GELU, SiLU

### 卷积和池化
Conv, MaxPool, AveragePool

### 归一化
BatchNormalization, LayerNormalization, RMSNorm

### 形状操作
Reshape, Transpose, Slice, Gather

## 性能优化

- **BLAS优化**：MatMul算子支持Accelerate（macOS）和OpenBLAS（Linux）
- **SIMD优化**：自动启用AVX/AVX2/NEON优化
- **内存优化**：内存池和生命周期分析

## 测试

```bash
cd build
make test_performance  # 性能测试
make test_math_operators  # 数学算子测试
make test_activation_operators  # 激活函数测试
make test_normalization_operators  # 归一化测试
make test_shape_operators  # 形状操作测试
```

## 更新日志

- 2026-01-06: 完成MatMul算子BLAS优化
- 2026-01-06: 完成性能基准测试
- 2026-01-06: 完成文档完善

## 许可证

[待定]

