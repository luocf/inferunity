# InferUnity API 参考文档

## 概述

InferUnity 是一个高性能的深度学习推理引擎，支持 ONNX 模型加载和推理。本文档提供了完整的 API 参考。

## 核心类

### Tensor（张量）

`Tensor` 是数据的基本容器，用于存储多维数组。

```cpp
#include "inferunity/tensor.h"

// 创建张量
Shape shape({1, 3, 224, 224});  // NCHW格式
auto tensor = CreateTensor(shape, DataType::FLOAT32);

// 访问数据
float* data = static_cast<float*>(tensor->GetData());
size_t count = tensor->GetElementCount();

// 获取形状信息
const Shape& tensor_shape = tensor->GetShape();
```

### Graph（计算图）

`Graph` 表示计算图，包含节点和边。

```cpp
#include "inferunity/graph.h"

// 创建图
auto graph = std::make_unique<Graph>();

// 添加输入节点
auto input = graph->AddInput("input", shape, DataType::FLOAT32);

// 添加算子节点
auto conv_node = graph->AddNode("Conv", {input}, {"conv_output"});

// 添加输出节点
graph->AddOutput("output", conv_node);

// 验证图
Status status = graph->Validate();
```

### InferenceSession（推理会话）

`InferenceSession` 是推理的主要接口。

```cpp
#include "inferunity/runtime.h"

// 创建会话
auto session = std::make_unique<InferenceSession>();

// 加载模型
Status status = session->LoadModel("model.onnx");
if (!status.IsOk()) {
    std::cerr << "Failed to load model: " << status.GetMessage() << std::endl;
    return;
}

// 准备输入
std::vector<Tensor*> inputs = {input_tensor.get()};
std::vector<Tensor*> outputs;

// 运行推理
status = session->Run(inputs, outputs);
if (!status.IsOk()) {
    std::cerr << "Inference failed: " << status.GetMessage() << std::endl;
    return;
}

// 获取输出
for (auto* output : outputs) {
    // 处理输出数据
}
```

## ONNX 模型加载

### ONNXParser

```cpp
#include "inferunity/onnx_parser.h"

// 创建解析器
auto parser = std::make_unique<ONNXParser>();

// 从文件加载
Status status = parser->LoadFromFile("model.onnx");
if (!status.IsOk()) {
    std::cerr << "Failed to load ONNX model" << std::endl;
    return;
}

// 转换为内部图表示
auto graph = parser->ConvertToGraph();
if (!graph) {
    std::cerr << "Failed to convert ONNX model" << std::endl;
    return;
}
```

## 执行提供者（Execution Provider）

### CPUExecutionProvider

```cpp
#include "inferunity/backend.h"

// 获取CPU执行提供者
auto provider = ExecutionProviderRegistry::Instance().Create("CPU");
if (!provider) {
    std::cerr << "CPU provider not available" << std::endl;
    return;
}

// 检查算子支持
bool supports_conv = provider->SupportsOperator("Conv");
bool supports_matmul = provider->SupportsOperator("MatMul");
```

## 内存管理

### 内存统计

```cpp
#include "inferunity/memory.h"

// 获取内存统计
MemoryStats stats = GetMemoryStats();
std::cout << "Total allocated: " << stats.total_allocated << " bytes" << std::endl;
std::cout << "Peak usage: " << stats.peak_usage << " bytes" << std::endl;
std::cout << "Allocation count: " << stats.allocation_count << std::endl;
```

### 内存释放

```cpp
// 释放未使用的内存
ReleaseUnusedMemory();
```

## 线程池

```cpp
#include "inferunity/runtime.h"

// 获取线程池
auto& thread_pool = ThreadPool::Instance();

// 提交任务
auto future = thread_pool.Submit([]() {
    // 执行任务
    return 42;
});

// 等待完成
int result = future.get();
```

## 错误处理

所有 API 函数返回 `Status` 对象，用于错误处理。

```cpp
Status status = some_operation();
if (!status.IsOk()) {
    std::cerr << "Error: " << status.GetMessage() << std::endl;
    std::cerr << "Error code: " << static_cast<int>(status.GetCode()) << std::endl;
    return;
}
```

## 数据类型

支持的数据类型：

- `DataType::FLOAT32` - 32位浮点数
- `DataType::FLOAT64` - 64位浮点数
- `DataType::INT32` - 32位整数
- `DataType::INT64` - 64位整数
- `DataType::UINT8` - 8位无符号整数

## 支持的算子

### 数学运算
- `Add` - 加法
- `Sub` - 减法
- `Mul` - 乘法
- `Div` - 除法
- `MatMul` - 矩阵乘法（支持BLAS优化）

### 激活函数
- `Relu` - ReLU激活
- `Sigmoid` - Sigmoid激活
- `GELU` - GELU激活
- `SiLU` - SiLU激活

### 卷积和池化
- `Conv` - 卷积
- `MaxPool` - 最大池化
- `AveragePool` - 平均池化

### 归一化
- `BatchNormalization` - 批归一化
- `LayerNormalization` - 层归一化
- `RMSNorm` - RMS归一化

### 形状操作
- `Reshape` - 重塑
- `Transpose` - 转置
- `Slice` - 切片
- `Gather` -  gather操作

## 性能优化

### BLAS 优化

MatMul 算子支持 BLAS 库优化（macOS 使用 Accelerate 框架，Linux 使用 OpenBLAS）。

编译时启用：
```bash
cmake .. -DUSE_BLAS=ON
```

### SIMD 优化

自动启用 SIMD 优化（AVX/AVX2/NEON），无需额外配置。

## 完整示例

参见 `examples/` 目录下的示例代码：
- `test_cpu_backend.cpp` - CPU后端测试示例
- 更多示例待添加

## 更新日志

- 2026-01-06: 创建API参考文档

