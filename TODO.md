# InferUnity 开发任务清单

## 当前阶段: Phase 1 - 核心功能

### 🔴 进行中

- [x] **Task 1.1: ONNX模型解析器** (P0, 1周) - 已完成
  - [x] 创建ONNXParser类框架
  - [x] 集成ONNX protobuf库（CMake配置）
  - [x] 实现ONNX到内部IR的转换
  - [x] 支持基础算子映射（框架已就绪）
  - [x] 形状推断（完整实现）
  - [x] 测试和验证

### ⏸️ 待开始

- [x] **Task 1.2: 基础算子实现** (P0, 2周) - 已完成
  - [x] Conv算子（参考NCNN实现）
  - [x] ReLU/Sigmoid/Tanh
  - [x] MatMul/Add/Mul
  - [x] MaxPool/AvgPool
  - [x] BatchNormalization
  - [x] Softmax

- [x] **Task 1.3: 图优化器实现** (P0, 1周) - 已完成
  - [x] 常量折叠算法（基本实现）
  - [x] Conv+BN+ReLU融合（完整实现）
  - [x] MatMul+Add融合（完整实现）
  - [x] Conv+ReLU融合（新增）
  - [x] BN+ReLU融合（新增）
  - [x] 死代码消除（已实现）
  - [x] SIMD优化（AVX/SSE/NEON）
  - [x] 单元测试（融合算子和融合Pass）
  - [x] 内存布局优化（完整实现）

- [x] **Task 1.4: 内存管理优化** (P1, 1周) - 已完成
  - [x] 张量生命周期分析
  - [x] 内存复用算法
  - [x] 内存池优化（参考NCNN）
  - [x] 内存统计（完整实现）
  - [x] 内存碎片整理（框架已实现）

## 当前阶段: Phase 2 - 性能优化

### 🔴 进行中

- [x] **Task 2.1: CPU后端优化** (P0, 2周) - 部分完成
  - [x] SIMD优化（AVX/SSE/NEON）- 已在Task 1.3中完成
  - [x] 多线程并行（已实现ParallelScheduler）
  - [x] 性能基准测试工具（已创建benchmark工具）
  - [ ] 性能测试和调优（进行中）

### ⏸️ 待开始

- [x] **Task 4.1: 模型转换工具** (P1, 1周) - 已完成
  - [x] 命令行工具
  - [x] 模型格式转换
  - [x] 模型验证

- [x] **Task 4.2: 性能分析器** (P1, 1周) - 已完成
  - [x] 逐层性能分析
  - [x] 内存使用分析
  - [x] 报告生成（CSV导出）

- [x] **Task 4.3: 基准测试套件** (P1, 1周) - 已完成
  - [x] 性能基准测试工具（已创建）
  - [x] 标准模型测试（已实现）
  - [x] 性能对比（已实现）
  - [x] 回归测试（已实现）

## 当前阶段: 核心功能完善

### 🔴 进行中 - 核心C++功能

- [x] **完善Graph功能** (P0) - 已完成
  - [x] Graph序列化（已实现）
  - [ ] Graph反序列化（部分实现，建议使用ONNX格式）
  - [x] Graph深拷贝（Clone已实现）
  - [x] 图验证逻辑完善（已实现）

- [x] **完善Tensor功能** (P0) - 已完成
  - [x] Tensor序列化/反序列化（已实现）
  - [x] Tensor切片操作（Slice已实现）
  - [ ] 跨设备拷贝实现（基础框架，待硬件后端）

- [x] **扩展算子支持** (P0) - 已完成
  - [x] Reshape算子（已实现）
  - [x] Concat算子（已实现，属性获取已完善）
  - [x] Split算子（已实现，属性获取已完善）
  - [x] Transpose算子（已实现，属性获取已完善）
  - [x] Gather算子（已实现）
  - [x] Slice算子（已实现）
  - [ ] 其他常用算子（按需添加）

- [x] **CPU后端优化完善** (P0) - 已完成
  - [x] CPU优化策略实现（已实现）
  - [x] 节点编译优化（已实现）
  - [x] 执行准备优化（已实现）

- [x] **日志系统框架** (P0) - 已完成
  - [x] 日志级别定义（VERBOSE/INFO/WARNING/ERROR/FATAL）
  - [x] 单例日志器实现
  - [x] 控制台和文件输出支持
  - [x] 线程安全的日志输出
  - [x] 便捷宏定义（LOG_INFO, LOG_ERROR等）
  - [x] 集成到InferenceSession

- [x] **核心功能完善** (P0) - 已完成
  - [x] LoadModelFromMemory实现（支持从内存加载ONNX模型）
  - [x] RunAsync异步推理（已实现）
  - [x] 跨设备拷贝基础框架（通过CPU中转）
  - [x] Graph反序列化（部分实现，建议使用ONNX格式）
  - [x] 更多算子实现（Gather、Slice已添加）

- [x] **CUDA后端框架** (P1) - 框架已完成
  - [x] CUDA设备管理（CUDADevice实现）
  - [x] CUDA执行提供者框架（CUDAExecutionProvider）
  - [x] CUDA内存管理接口
  - [x] CUDA流管理接口
  - [ ] CUDA算子实现（待CUDA SDK环境）

- [x] **性能优化** (P0) - 已完成
  - [x] FusedConvBNReLU kernel优化（预计算BN参数，优化内存访问模式）
  - [x] Gather算子支持任意axis（从节点属性获取）
  - [x] 算子属性获取完善（所有算子都支持从节点属性获取参数）

### ⏸️ 待开始

- [ ] **Task 3.1: CUDA后端** (P1, 2周) - 可选，需要CUDA SDK
  - [ ] CUDA设备管理
  - [ ] CUDA内存分配器
  - [ ] CUDA算子实现
  - [ ] 流管理

- [ ] **Task 5.2: Python绑定** (P2, 可选) - 框架已创建，默认关闭
  - [x] pybind11集成（默认BUILD_PYTHON=OFF）
  - [ ] 按需完善

## 下一步行动（聚焦核心C++引擎）

1. ✅ **已完成**: 完善Graph验证和Tensor切片操作
2. ✅ **已完成**: 实现更多常用算子（Reshape、Concat、Split、Transpose）
3. ✅ **已完成**: CPU后端优化策略和节点编译
4. ✅ **已完成**: 完整的单元测试和集成测试框架
5. **按需实现**: CUDA后端（如有GPU需求）
6. **持续优化**: 性能测试和调优

## 已完成 ✅

- [x] Phase 0: 基础架构
  - [x] 类型系统
  - [x] 张量系统
  - [x] 内存管理
  - [x] 计算图IR
  - [x] 算子抽象层
  - [x] ExecutionProvider接口
  - [x] 优化器框架
  - [x] 运行时系统
  - [x] CPU ExecutionProvider
  - [x] ONNXParser框架

- [x] 核心功能完善
  - [x] Graph验证逻辑（完整实现）
  - [x] Tensor切片操作（Slice实现）
  - [x] Split算子完整实现
  - [x] Transpose高维实现
  - [x] 算子属性获取完善

- [x] CPU后端优化
  - [x] CPU优化策略实现
  - [x] 节点编译优化
  - [x] 执行准备优化

- [x] 测试框架
  - [x] 核心功能单元测试（Tensor、Graph、Memory）
  - [x] 算子单元测试（所有基础算子）
  - [x] 集成测试（端到端推理）
  - [x] 测试文档和指南

- [x] 日志系统
  - [x] 日志系统框架实现
  - [x] 集成到核心组件

## 设计特色

参考 [DESIGN_PHILOSOPHY.md](docs/DESIGN_PHILOSOPHY.md) 了解InferUnity的差异化设计。
