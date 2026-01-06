# 后端集成指南

## 📖 概述

InferUnity采用**后端集成架构**，将成熟的推理框架（如ONNX Runtime）作为执行提供者，而非重复实现底层算子。

## 🏗️ 架构

```
用户代码
    ↓
InferUnity API (统一接口)
    ↓
ExecutionProvider Interface (后端抽象层)
    ↓
Backend Implementations
    ├─ ONNX Runtime (推荐)
    ├─ CPU (过渡实现)
    └─ 其他后端 (可选)
```

## 🔧 使用方式

### 1. 使用ONNX Runtime后端（推荐）

```cpp
#include "inferunity/engine.h"

using namespace inferunity;

int main() {
    // 创建会话选项
    SessionOptions options;
    options.execution_providers = {"ONNXRuntime"};  // 使用ONNX Runtime后端
    options.graph_optimization_level = SessionOptions::GraphOptimizationLevel::ALL;
    
    // 创建推理会话
    auto session = InferenceSession::Create(options);
    if (!session) {
        std::cerr << "Failed to create session" << std::endl;
        return 1;
    }
    
    // 加载模型
    Status status = session->LoadModel("model.onnx");
    if (!status.IsOk()) {
        std::cerr << "Failed to load model: " << status.GetMessage() << std::endl;
        return 1;
    }
    
    // 准备输入
    std::vector<Tensor*> inputs = {input_tensor};
    std::vector<Tensor*> outputs = {output_tensor};
    
    // 执行推理
    status = session->Run(inputs, outputs);
    if (!status.IsOk()) {
        std::cerr << "Inference failed: " << status.GetMessage() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### 2. 自动选择后端

```cpp
SessionOptions options;
// 不指定execution_providers，系统会自动选择最优后端
options.execution_providers = {};  // 或省略

auto session = InferenceSession::Create(options);
// 系统会按优先级选择：ONNX Runtime > CPU
```

### 3. 查询后端能力

```cpp
auto& registry = ExecutionProviderRegistry::Instance();
auto providers = registry.GetRegisteredProviders();

for (const auto& name : providers) {
    auto provider = registry.Create(name);
    if (provider && provider->IsAvailable()) {
        std::cout << "Backend: " << provider->GetName() << std::endl;
        std::cout << "Version: " << provider->GetVersion() << std::endl;
        std::cout << "Device: " << (int)provider->GetDeviceType() << std::endl;
    }
}
```

## 🛠️ 编译配置

### 启用ONNX Runtime后端

```bash
# 安装ONNX Runtime（macOS）
brew install onnxruntime

# 或从源码编译
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib

# 配置CMake
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_DIR=/path/to/onnxruntime/build

# 编译
make -j$(nproc)
```

### 仅使用CPU后端（过渡实现）

```bash
# 不启用ONNX Runtime
cmake .. -DENABLE_ONNXRUNTIME=OFF
make -j$(nproc)
```

## 📊 后端对比

| 特性 | ONNX Runtime | CPU (内部实现) |
|------|-------------|---------------|
| 性能 | ⭐⭐⭐⭐⭐ 优秀 | ⭐⭐ 基础 |
| 稳定性 | ⭐⭐⭐⭐⭐ 经过验证 | ⭐⭐⭐ 测试中 |
| 算子支持 | ⭐⭐⭐⭐⭐ 完整ONNX标准 | ⭐⭐ 部分算子 |
| SIMD优化 | ⭐⭐⭐⭐⭐ 深度优化 | ❌ 未实现 |
| GPU支持 | ✅ 支持 | ❌ 不支持 |
| 推荐场景 | 生产环境 | 测试/开发 |

## 🎯 最佳实践

1. **生产环境**：使用ONNX Runtime后端
2. **开发测试**：可以使用CPU后端（如果ONNX Runtime不可用）
3. **性能优化**：利用ONNX Runtime的图优化（自动执行）
4. **多后端**：未来可以同时支持多个后端，根据模型自动选择

## 📝 注意事项

1. **ONNX Runtime依赖**：需要单独安装ONNX Runtime库
2. **模型格式**：后端期望ONNX格式模型
3. **内存管理**：Tensor内存由InferUnity管理，后端负责执行
4. **错误处理**：后端错误会通过Status返回

---

**下一步**：实现ONNX Runtime后端的完整集成和测试

