# 后端集成架构设计

## 🎯 设计理念

**核心思想**：成为成熟推理框架的"使用者"和"集成者"，而非重复实现底层算子。

**优势**：
- ✅ 性能卓越：利用专家深度优化的SIMD/GPU内核
- ✅ 稳定可靠：经过海量项目验证
- ✅ 生态丰富：支持大量算子、模型和硬件
- ✅ 快速落地：专注引擎架构和应用逻辑

## 🏗️ 架构设计

### 三层架构

```
┌─────────────────────────────────────────┐
│   InferUnity API Layer                  │  ← 用户接口
│   (统一API、资源管理、调度)              │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│   ExecutionProvider Interface          │  ← 后端抽象层
│   (统一接口、后端选择、切换)             │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│   Backend Implementations               │  ← 后端实现
│   - ONNX Runtime                        │
│   - TensorRT (可选)                     │
│   - NCNN (可选)                         │
│   - OpenVINO (可选)                     │
│   - TFLite (可选)                       │
└─────────────────────────────────────────┘
```

### ExecutionProvider接口设计

```cpp
class ExecutionProvider {
public:
    virtual ~ExecutionProvider() = default;
    
    // 后端能力查询
    virtual std::string GetName() const = 0;
    virtual std::vector<std::string> GetSupportedOps() const = 0;
    virtual bool IsOpSupported(const std::string& op_type) const = 0;
    virtual DeviceType GetDeviceType() const = 0;  // CPU, GPU, etc.
    
    // 模型加载和执行
    virtual Status LoadModel(const std::string& model_path) = 0;
    virtual Status LoadModelFromMemory(const void* data, size_t size) = 0;
    virtual Status Run(const std::vector<Tensor*>& inputs,
                      std::vector<Tensor*>& outputs) = 0;
    
    // 优化选项
    virtual Status OptimizeGraph(Graph* graph) = 0;
    virtual Status SetOptimizationLevel(int level) = 0;
    
    // 资源管理
    virtual Status AllocateMemory(size_t size) = 0;
    virtual Status ReleaseMemory() = 0;
};
```

## 📋 实现计划

### Phase 1: 后端抽象层 (优先级: P0)

#### Task 1.1: 扩展ExecutionProvider接口
- 添加后端能力查询方法
- 添加模型加载和执行方法
- 添加优化选项接口

#### Task 1.2: 实现后端注册和选择机制
- 后端自动发现和注册
- 根据模型和硬件自动选择最优后端
- 支持手动指定后端

### Phase 2: ONNX Runtime集成 (优先级: P0)

#### Task 2.1: 集成ONNX Runtime库
- 在CMakeLists.txt中添加ONNX Runtime依赖
- 配置编译选项（CPU/GPU支持）

#### Task 2.2: 实现ONNXRuntimeExecutionProvider
- 封装ONNX Runtime的C++ API
- 实现ExecutionProvider接口
- 处理输入/输出Tensor转换

#### Task 2.3: 测试ONNX Runtime集成
- 加载简单模型测试
- 性能对比测试
- 错误处理测试

### Phase 3: 模型转换与优化 (优先级: P1)

#### Task 3.1: 模型转换工具
- PyTorch → ONNX转换脚本
- TensorFlow → ONNX转换脚本
- 模型验证工具

#### Task 3.2: 图优化
- 利用ONNX Runtime的图优化
- 层融合（Conv+BN+ReLU等）
- 常量折叠
- 死代码消除

#### Task 3.3: 量化支持
- INT8量化（如果后端支持）
- 动态量化
- 静态量化

### Phase 4: 运行时封装 (优先级: P1)

#### Task 4.1: 统一API设计
- InferenceSession封装
- 输入/输出管理
- 错误处理和日志

#### Task 4.2: 资源管理
- 内存池管理
- Tensor生命周期管理
- 跨后端内存拷贝

### Phase 5: 调度与并发 (优先级: P2)

#### Task 5.1: 线程池管理
- 工作线程池
- 任务队列
- 负载均衡

#### Task 5.2: 流水线执行
- 多阶段流水线
- 异步执行
- 批处理优化

#### Task 5.3: 多模型并发
- 模型实例管理
- 请求路由
- 资源隔离

## 🔧 技术选型

### 主要后端：ONNX Runtime

**选择理由**：
- ✅ 跨平台支持（Windows/Linux/macOS）
- ✅ CPU和GPU支持（CUDA、TensorRT、OpenVINO）
- ✅ 丰富的算子支持
- ✅ 活跃的社区和文档
- ✅ 良好的C++ API

**集成方式**：
```cpp
#include <onnxruntime_cxx_api.h>

class ONNXRuntimeExecutionProvider : public ExecutionProvider {
private:
    Ort::Env env_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    
public:
    Status LoadModel(const std::string& model_path) override {
        Ort::SessionOptions session_options;
        session_ = Ort::Session(env_, model_path.c_str(), session_options);
        return Status::Ok();
    }
    
    Status Run(const std::vector<Tensor*>& inputs,
               std::vector<Tensor*>& outputs) override {
        // 转换输入Tensor到ONNX Runtime格式
        std::vector<Ort::Value> ort_inputs;
        for (auto* input : inputs) {
            ort_inputs.push_back(CreateOrtValue(input));
        }
        
        // 执行推理
        auto ort_outputs = session_.Run(
            Ort::RunOptions{nullptr},
            input_names_.data(), ort_inputs.data(), ort_inputs.size(),
            output_names_.data(), output_names_.size()
        );
        
        // 转换输出Tensor
        for (size_t i = 0; i < outputs.size(); ++i) {
            ConvertOrtValue(ort_outputs[i], outputs[i]);
        }
        
        return Status::Ok();
    }
};
```

### 可选后端

- **TensorRT**: NVIDIA GPU加速（需要CUDA）
- **NCNN**: 移动端优化（ARM NEON）
- **OpenVINO**: Intel硬件优化
- **TFLite**: TensorFlow Lite（移动端）

## 📊 实现优先级

1. **P0 (立即实现)**:
   - 后端抽象层
   - ONNX Runtime集成
   - 基本模型加载和执行

2. **P1 (短期)**:
   - 模型转换工具
   - 图优化
   - 资源管理

3. **P2 (中期)**:
   - 调度与并发
   - 多后端支持
   - 性能优化

## 🎯 成功指标

- ✅ 能够加载和执行ONNX模型
- ✅ 性能接近原生ONNX Runtime
- ✅ 支持CPU和GPU（如果可用）
- ✅ 统一的API接口
- ✅ 良好的错误处理和日志

---

**开始实现**: Phase 1 - 后端抽象层

