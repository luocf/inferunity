# InferUnity 推理引擎

跨平台、高性能的深度学习推理引擎，支持多种硬件后端和模型格式。

## 核心特性

- **跨平台支持**: Linux (x86_64/ARM64), macOS (Apple Silicon + Intel), 嵌入式平台 (NVIDIA Jetson, 高通骁龙)
- **多硬件后端**: CPU, NVIDIA GPU (CUDA/TensorRT), Vulkan, Metal, SNPE, ARM NN
- **模型格式**: ONNX, TensorFlow, PyTorch, TFLite
- **高性能优化**: 算子融合、量化、内存优化、自动调优
- **轻量级部署**: 最小依赖、模块化编译、静态链接支持

## 文档

- [设计文档](docs/DESIGN.md) - 架构设计和核心组件
- [实现计划](docs/IMPLEMENTATION_PLAN.md) - 开发计划和排期
- [API文档](docs/API.md) - API使用说明
- [任务清单](TODO.md) - 当前开发任务

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 使用示例

```cpp
#include "inferunity/engine.h"

// 加载模型
auto engine = inferunity::Engine::Create();
engine->LoadModel("model.onnx");

// 准备输入
auto input = engine->CreateTensor({1, 3, 224, 224}, inferunity::DataType::FLOAT32);
// ... 填充输入数据 ...

// 执行推理
engine->Run(input, output);

// 获取输出
auto result = engine->GetOutput(0);
```

## 目录结构

```
inferunity/
├── include/          # 公共头文件
├── src/              # 核心实现
│   ├── core/        # 核心组件 (IR, 张量, 内存)
│   ├── hal/         # 硬件抽象层
│   ├── optimizers/  # 优化器
│   ├── runtime/     # 运行时系统
│   └── backends/    # 硬件后端实现
├── tools/            # 工具链
├── tests/            # 测试代码
└── docs/             # 文档
```

## 性能目标

- **延迟**: ResNet-50 (224x224) < 10ms (Jetson Orin)
- **内存**: 运行时内存 < 100MB (基础)
- **启动**: 冷启动时间 < 100ms
- **精度**: FP32精度无损，INT8精度损失<1%

## 许可证

[待定]

