# InferUnity 推理引擎设计文档

## 1. 架构设计

### 1.1 分层架构

```
应用层 (API)
  ↓
运行时层 (调度器、执行引擎)
  ↓
优化层 (图优化器、Pass管理)
  ↓
中间表示层 (计算图IR、节点、张量)
  ↓
硬件抽象层 (ExecutionProvider接口)
  ↓
后端实现层 (CPU、CUDA、TensorRT等)
```

### 1.2 核心组件

- **InferenceSession**: 主要API入口（参考ONNX Runtime）
- **ExecutionProvider**: 执行提供者接口（参考ONNX Runtime）
- **Graph**: 计算图IR
- **Optimizer**: 图优化器（参考TVM Pass机制）
- **OperatorRegistry**: 算子注册表（参考TensorFlow Lite）

### 1.3 参考设计

- **ONNX Runtime**: ExecutionProvider架构、Session设计
- **TVM**: Pass机制、图优化框架
- **TensorFlow Lite**: 算子注册、Interpreter设计
- **NCNN**: 内存池、内存复用策略

## 2. 实现计划

详见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
