# InferUnity 用户指南

## 快速开始

### 编译项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DUSE_BLAS=ON

# 编译
make -j$(nproc)
```

### 基本使用

#### 1. 加载ONNX模型

```cpp
#include "inferunity/runtime.h"
#include "inferunity/onnx_parser.h"

// 创建推理会话
auto session = std::make_unique<InferenceSession>();

// 加载模型
Status status = session->LoadModel("model.onnx");
if (!status.IsOk()) {
    std::cerr << "Failed to load model" << std::endl;
    return -1;
}
```

#### 2. 准备输入数据

```cpp
#include "inferunity/tensor.h"

// 获取输入形状（从模型获取）
Shape input_shape({1, 3, 224, 224});  // NCHW格式

// 创建输入张量
auto input = CreateTensor(input_shape, DataType::FLOAT32);

// 填充输入数据
float* data = static_cast<float*>(input->GetData());
// ... 填充数据 ...
```

#### 3. 运行推理

```cpp
// 准备输入和输出
std::vector<Tensor*> inputs = {input.get()};
std::vector<Tensor*> outputs;

// 运行推理
status = session->Run(inputs, outputs);
if (!status.IsOk()) {
    std::cerr << "Inference failed" << std::endl;
    return -1;
}

// 处理输出
for (auto* output : outputs) {
    const Shape& shape = output->GetShape();
    float* output_data = static_cast<float*>(output->GetData());
    // ... 处理输出数据 ...
}
```

## 完整示例

### 示例1：简单推理

```cpp
#include "inferunity/runtime.h"
#include "inferunity/tensor.h"
#include <iostream>

int main() {
    // 创建会话
    auto session = std::make_unique<InferenceSession>();
    
    // 加载模型
    if (!session->LoadModel("model.onnx").IsOk()) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }
    
    // 创建输入
    Shape input_shape({1, 3, 224, 224});
    auto input = CreateTensor(input_shape, DataType::FLOAT32);
    
    // 填充输入（示例）
    float* data = static_cast<float*>(input->GetData());
    size_t count = input->GetElementCount();
    for (size_t i = 0; i < count; ++i) {
        data[i] = 1.0f;  // 示例数据
    }
    
    // 运行推理
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs;
    
    Status status = session->Run(inputs, outputs);
    if (!status.IsOk()) {
        std::cerr << "Inference failed: " << status.GetMessage() << std::endl;
        return -1;
    }
    
    // 输出结果
    std::cout << "Inference successful!" << std::endl;
    std::cout << "Number of outputs: " << outputs.size() << std::endl;
    
    return 0;
}
```

### 示例2：使用CPU后端

```cpp
#include "inferunity/backend.h"
#include "inferunity/runtime.h"

int main() {
    // 获取CPU执行提供者
    auto provider = ExecutionProviderRegistry::Instance().Create("CPU");
    if (!provider) {
        std::cerr << "CPU provider not available" << std::endl;
        return -1;
    }
    
    // 检查支持的算子
    std::cout << "Conv supported: " << provider->SupportsOperator("Conv") << std::endl;
    std::cout << "MatMul supported: " << provider->SupportsOperator("MatMul") << std::endl;
    
    // 创建会话并设置提供者
    auto session = std::make_unique<InferenceSession>();
    // ... 使用会话 ...
    
    return 0;
}
```

## 性能优化建议

### 1. 启用BLAS优化

编译时启用BLAS库可以显著提升MatMul算子性能：

```bash
cmake .. -DUSE_BLAS=ON
```

### 2. 内存管理

对于长时间运行的推理服务，定期释放未使用的内存：

```cpp
#include "inferunity/memory.h"

// 定期调用
ReleaseUnusedMemory();
```

### 3. 批量推理

对于多个输入，使用批量处理可以提高吞吐量：

```cpp
// 创建批量输入
std::vector<std::unique_ptr<Tensor>> batch_inputs;
for (int i = 0; i < batch_size; ++i) {
    batch_inputs.push_back(CreateTensor(input_shape, DataType::FLOAT32));
    // 填充数据...
}

// 批量推理
for (auto& input : batch_inputs) {
    std::vector<Tensor*> inputs = {input.get()};
    std::vector<Tensor*> outputs;
    session->Run(inputs, outputs);
}
```

## 常见问题

### Q: 如何检查模型是否加载成功？

A: 检查 `LoadModel` 返回的 `Status`：

```cpp
Status status = session->LoadModel("model.onnx");
if (!status.IsOk()) {
    std::cerr << "Error: " << status.GetMessage() << std::endl;
    return;
}
```

### Q: 如何获取模型的输入输出形状？

A: 在加载模型后，可以通过会话获取：

```cpp
// 获取输入信息
auto input_infos = session->GetInputInfos();
for (const auto& info : input_infos) {
    std::cout << "Input: " << info.name << ", Shape: " << info.shape << std::endl;
}

// 获取输出信息
auto output_infos = session->GetOutputInfos();
for (const auto& info : output_infos) {
    std::cout << "Output: " << info.name << ", Shape: " << info.shape << std::endl;
}
```

### Q: 支持哪些ONNX算子？

A: 当前支持26个算子，包括：
- 数学运算：Add, Sub, Mul, Div, MatMul
- 激活函数：Relu, Sigmoid, GELU, SiLU
- 卷积和池化：Conv, MaxPool, AveragePool
- 归一化：BatchNormalization, LayerNormalization, RMSNorm
- 形状操作：Reshape, Transpose, Slice, Gather
- 更多算子正在开发中

### Q: 如何启用性能分析？

A: 使用性能测试工具：

```bash
./build/bin/test_performance
```

## 更新日志

- 2026-01-06: 创建用户指南

