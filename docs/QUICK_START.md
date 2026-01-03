# InferUnity 快速开始指南

## ✅ CPU推理引擎已就绪

根据设计文档，**CPU版本的推理引擎核心功能已全部实现**，可以运行推理任务。

## 当前实现状态

### ✅ 已完成的核心功能

1. **模型加载** ✅
   - ONNX模型加载（从文件和内存）
   - ONNX到内部IR转换
   - 形状推断系统

2. **算子支持** ✅ (约29个算子)
   - 基础算子：Conv, MatMul, Add, Mul, Sub, Div
   - 激活函数：ReLU, Sigmoid, Tanh, **GELU, SiLU** (新增)
   - 归一化：BatchNorm, **LayerNorm, RMSNorm** (新增)
   - 形状操作：Reshape, Concat, Split, Transpose, Gather, Slice
   - 融合算子：FusedConvBNReLU等

3. **图优化** ✅
   - 常量折叠、算子融合、SIMD优化

4. **运行时系统** ✅
   - ExecutionEngine、多线程调度、CPU后端

## 运行推理

### 1. 编译项目

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
make -j$(nproc)
```

### 2. 测试简单模型

```bash
# 验证模型
./bin/inferunity_tool validate model.onnx

# 查看模型信息
./bin/inferunity_tool info model.onnx

# 运行推理示例
./bin/inference_example model.onnx
```

### 3. 使用C++ API

```cpp
#include "inferunity/engine.h"

// 创建会话
auto session = inferunity::InferenceSession::Create();

// 加载模型
session->LoadModel("model.onnx");

// 准备输入
auto input = session->CreateInputTensor(0);
// ... 填充数据 ...

// 执行推理
std::vector<inferunity::Tensor*> inputs = {input.get()};
std::vector<inferunity::Tensor*> outputs;
session->Run(inputs, outputs);

// 获取输出
auto output = session->GetOutputTensor(0);
```

## 运行Qwen2.5 Omni 1.5B

### ✅ 可行性

1. **算子支持**：大部分Transformer算子已实现
   - ✅ MatMul, Add, LayerNorm, Softmax
   - ✅ GELU, SiLU (新增)
   - ✅ Reshape, Transpose, Gather, Slice

2. **模型规模**：1.5B参数在CPU上可以运行

3. **内存管理**：内存复用机制已实现

### ⚠️ 注意事项

1. **算子检查**：先检查模型使用的算子
   ```bash
   ./bin/inferunity_tool info qwen2.5_omni_1.5b.onnx
   ```

2. **性能**：CPU推理可能较慢，适合离线推理

3. **缺失算子**：如果遇到不支持的算子，可以按需添加

## 可能缺失的算子

如果运行模型时遇到不支持的算子，可以按需添加：

1. **Embedding** - 词嵌入（可能由Gather实现）
2. **RoPE** - 旋转位置编码（可能由多个算子组合）
3. **ReduceMean/ReduceSum** - 归约操作
4. **Where/Select** - 条件选择
5. **Pow** - 幂运算

## 测试建议

1. **先测试简单模型**：验证功能完整性
2. **检查算子列表**：确认模型使用的算子
3. **按需添加算子**：实现缺失的算子
4. **性能优化**：使用性能分析器优化热点

## 总结

✅ **CPU推理引擎已就绪**，可以开始测试推理功能！

**建议流程**：
1. 编译项目
2. 测试简单ONNX模型
3. 检查目标模型的算子
4. 按需添加缺失算子
5. 进行端到端测试

---

**当前状态**：核心功能完整，可以运行推理！🚀

