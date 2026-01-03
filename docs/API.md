# InferUnity API 文档

## 快速开始

```cpp
#include "inferunity/engine.h"

// 创建会话
SessionOptions options;
options.execution_providers = {"CPUExecutionProvider"};
auto session = InferenceSession::Create(options);

// 加载模型
session->LoadModel("model.onnx");

// 准备输入
auto input = session->CreateInputTensor(0);
// ... 填充数据 ...

// 执行推理
std::vector<Tensor*> inputs = {input.get()};
std::vector<Tensor*> outputs;
session->Run(inputs, outputs);

// 获取输出
auto output = session->GetOutputTensor(0);
```

## 核心API

### InferenceSession

主要推理接口，参考ONNX Runtime设计。

```cpp
// 创建会话
std::unique_ptr<InferenceSession> Create(const SessionOptions& options);

// 模型加载
Status LoadModel(const std::string& filepath);
Status LoadModelFromMemory(const void* data, size_t size);

// 推理执行
Status Run(const std::vector<Tensor*>& inputs, 
           std::vector<Tensor*>& outputs);
```

### Tensor

张量数据结构。

```cpp
// 创建张量
std::shared_ptr<Tensor> CreateTensor(const Shape& shape, DataType dtype);

// 数据访问
void* GetData();
const Shape& GetShape() const;
```

### ExecutionProvider

执行提供者接口，参考ONNX Runtime。

```cpp
// 注册提供者
ExecutionProviderRegistry::Instance().Register("MyProvider", factory);

// 创建提供者
auto provider = ExecutionProviderRegistry::Instance().Create("CPUExecutionProvider");
```

## 详细文档

详见代码注释和示例代码。
