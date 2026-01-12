# 推理执行流程详解

本文档详细讲解 InferUnity 中推理执行的完整流程，从用户调用 `Run()` 到获取输出结果的整个过程。

## 目录

1. [整体流程概览](#整体流程概览)
2. [第一步：Run 方法入口](#第一步run-方法入口)
3. [第二步：输入绑定](#第二步输入绑定)
4. [第三步：获取执行顺序](#第三步获取执行顺序)
5. [第四步：节点执行](#第四步节点执行)
6. [第五步：输出收集](#第五步输出收集)
7. [调度器详解](#调度器详解)
8. [执行提供者选择](#执行提供者选择)

---

## 整体流程概览

从用户调用 `session->Run(inputs, outputs)` 到获取结果，主要经过以下步骤：

```
用户调用 Run(inputs, outputs)
    ↓
1. 输入验证和绑定
    ↓
2. 获取执行顺序（拓扑排序）
    ↓
3. 遍历执行每个节点
    ├─ 选择执行提供者
    ├─ 准备输入输出张量
    ├─ 执行算子
    └─ 更新输出Value
    ↓
4. 收集输出结果
    ↓
返回输出张量
```

---

## 第一步：Run 方法入口

### 代码位置
`src/core/engine.cpp` 的 `InferenceSession::Run` 方法

### 核心逻辑

```cpp
Status InferenceSession::Run(const std::vector<Tensor*>& inputs, 
                            std::vector<Tensor*>& outputs) {
    if (!graph_) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Model not loaded");
    }
    
    ExecutionOptions options;
    options.enable_profiling = options_.enable_profiling;
    
    // 委托给ExecutionEngine执行
    return execution_engine_->Execute(graph_.get(), inputs, outputs, options);
}
```

### 关键点说明

1. **模型检查**：确保模型已加载
2. **选项配置**：设置执行选项（如性能分析）
3. **委托执行**：将实际执行委托给 `ExecutionEngine`

### 两种Run方法

**方法1：按索引输入输出**
```cpp
Status Run(const std::vector<Tensor*>& inputs, 
          std::vector<Tensor*>& outputs);
```

**方法2：按名称输入输出**
```cpp
Status Run(const std::unordered_map<std::string, Tensor*>& inputs,
           const std::unordered_map<std::string, Tensor*>& outputs);
```

第二种方法内部会转换为第一种方法：

```cpp
// 按名称映射输入
std::vector<Tensor*> input_tensors;
for (Value* input_value : graph_->GetInputs()) {
    std::string name = input_value->GetName();
    auto it = inputs.find(name);
    if (it != inputs.end()) {
        input_tensors.push_back(it->second);
    } else {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Missing input: " + name);
    }
}

// 执行推理
Status status = Run(input_tensors, output_tensors);

// 按名称映射输出
for (size_t i = 0; i < output_tensors.size(); ++i) {
    std::string name = graph_outputs[i]->GetName();
    outputs[name] = output_tensors[i];
}
```

---

## 第二步：输入绑定

### 代码位置
`src/runtime/runtime.cpp` 的 `ExecutionEngine::ExecuteInternal` 方法

### 核心逻辑

```cpp
Status ExecutionEngine::ExecuteInternal(const Graph* graph,
                                       const std::vector<Tensor*>& inputs,
                                       std::vector<Tensor*>& outputs,
                                       const ExecutionOptions& options) {
    // 1. 验证输入数量
    if (inputs.size() != graph->GetInputs().size()) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Input count mismatch");
    }
    
    // 2. 绑定输入Tensor到图的输入Value
    for (size_t i = 0; i < inputs.size(); ++i) {
        graph->GetInputs()[i]->SetTensor(
            std::shared_ptr<Tensor>(inputs[i], [](Tensor*){}));
    }
    
    // ... 后续执行逻辑 ...
}
```

### 关键点说明

1. **输入验证**：检查输入数量是否匹配
2. **Tensor绑定**：将用户提供的Tensor绑定到图的输入Value
3. **共享指针**：使用shared_ptr管理Tensor生命周期（但不拥有，使用空删除器）

**为什么使用空删除器？**
- 用户的Tensor由用户管理生命周期
- 我们只是临时引用，不应该删除它

---

## 第三步：获取执行顺序

### 代码位置
`src/runtime/runtime.cpp` 的 `ExecutionEngine::ExecuteInternal` 方法

### 核心逻辑

```cpp
// 获取执行顺序
std::vector<Node*> execution_order = scheduler_->GetExecutionOrder(graph);
```

### 拓扑排序

**为什么需要拓扑排序？**

计算图可能有多个节点，它们之间存在依赖关系。必须按照依赖顺序执行，确保每个节点的输入都已准备好。

**示例**：
```
图结构：
  Input → Conv → Relu → Output

拓扑排序结果：
  [Conv, Relu]
  
执行顺序：
  1. 先执行Conv（因为Relu依赖它的输出）
  2. 再执行Relu（此时Conv的输出已准备好）
```

### TopologicalScheduler 实现

```cpp
std::vector<Node*> TopologicalScheduler::GetExecutionOrder(const Graph* graph) const {
    return graph->TopologicalSort();
}
```

**拓扑排序算法**（在 `Graph::TopologicalSort` 中实现）：

```cpp
std::vector<Node*> Graph::TopologicalSort() const {
    std::vector<Node*> result;
    std::unordered_map<Node*, int> in_degree;  // 入度计数
    
    // 1. 计算每个节点的入度（依赖的输入数量）
    for (Node* node : nodes_) {
        in_degree[node] = node->GetInputs().size();
    }
    
    // 2. 找到所有入度为0的节点（没有依赖）
    std::queue<Node*> ready_queue;
    for (Node* node : nodes_) {
        if (in_degree[node] == 0) {
            ready_queue.push(node);
        }
    }
    
    // 3. 处理队列中的节点
    while (!ready_queue.empty()) {
        Node* node = ready_queue.front();
        ready_queue.pop();
        result.push_back(node);
        
        // 4. 更新依赖此节点的其他节点
        for (Value* output : node->GetOutputs()) {
            for (Node* consumer : output->GetConsumers()) {
                in_degree[consumer]--;
                if (in_degree[consumer] == 0) {
                    ready_queue.push(consumer);
                }
            }
        }
    }
    
    return result;
}
```

**关键点**：
- **入度**：节点依赖的输入数量
- **队列**：维护可执行的节点（入度为0）
- **依赖更新**：执行节点后，更新依赖它的节点入度

---

## 第四步：节点执行

### 代码位置
`src/runtime/runtime.cpp` 的 `ExecutionEngine::ExecuteInternal` 方法

### 核心逻辑

```cpp
// 执行每个节点
for (Node* node : execution_order) {
    // 1. 选择执行提供者
    ExecutionProvider* provider = ExecutionProviderSelector::SelectProvider(node, backend_ptrs_);
    
    // 2. 准备输入输出张量
    std::vector<Tensor*> node_inputs;
    std::vector<Tensor*> node_outputs;
    
    // 3. 收集输入张量
    for (Value* input : node->GetInputs()) {
        if (input->GetTensor()) {
            node_inputs.push_back(input->GetTensor().get());
        }
    }
    
    // 4. 创建输出张量（推断形状）
    for (Value* output : node->GetOutputs()) {
        // 推断输出形状
        auto op = provider->CreateOperator(node->GetOpType());
        std::vector<Shape> output_shapes;
        op->InferOutputShape(node_inputs, output_shapes);
        
        // 创建输出Tensor
        auto tensor = CreateTensor(output_shapes[0], DataType::FLOAT32);
        output->SetTensor(tensor);
        node_outputs.push_back(tensor.get());
    }
    
    // 5. 执行节点
    Status status = provider->ExecuteNode(node, &ctx);
    if (!status.IsOk()) {
        return status;
    }
}
```

### 4.1 执行提供者选择

**目的**：为节点选择合适的后端（CPU、CUDA等）

```cpp
ExecutionProvider* provider = ExecutionProviderSelector::SelectProvider(node, backend_ptrs_);
```

**选择策略**：
1. 按优先级顺序遍历执行提供者
2. 检查是否支持该算子类型
3. 返回第一个支持的提供者

**示例**：
```
节点: Conv
执行提供者列表: [CUDAExecutionProvider, CPUExecutionProvider]
  - CUDAExecutionProvider: 检查是否支持 "Conv" → 支持 ✓
  → 选择: CUDAExecutionProvider
```

### 4.2 输入张量收集

**目的**：从输入Value中收集Tensor

```cpp
for (Value* input : node->GetInputs()) {
    if (input->GetTensor()) {
        node_inputs.push_back(input->GetTensor().get());
    }
}
```

**关键点**：
- 输入Value的Tensor在之前节点执行时已设置
- 图输入的Tensor在输入绑定时已设置
- 权重（初始值）的Tensor在模型加载时已设置

### 4.3 输出张量创建

**目的**：为节点输出创建Tensor

```cpp
// 1. 创建算子实例
auto op = provider->CreateOperator(node->GetOpType());

// 2. 推断输出形状
std::vector<Shape> output_shapes;
op->InferOutputShape(node_inputs, output_shapes);

// 3. 创建输出Tensor
auto tensor = CreateTensor(output_shapes[0], DataType::FLOAT32);
output->SetTensor(tensor);
node_outputs.push_back(tensor.get());
```

**关键点**：
- **形状推断**：使用算子的 `InferOutputShape` 方法
- **内存分配**：创建Tensor时自动分配内存
- **关联Value**：将Tensor设置到输出Value，供后续节点使用

### 4.4 执行算子

**目的**：实际执行计算

```cpp
Status status = provider->ExecuteNode(node, &ctx);
```

**ExecuteNode 的实现**（以CPU后端为例）：

```cpp
Status CPUExecutionProvider::ExecuteNode(Node* node, ExecutionContext* ctx) {
    // 1. 从注册表创建算子
    auto op = OperatorRegistry::Instance().Create(node->GetOpType());
    if (!op) {
        return Status::Error(StatusCode::ERROR_NOT_FOUND,
                           "Operator not found: " + node->GetOpType());
    }
    
    // 2. 准备输入输出张量
    std::vector<Tensor*> inputs, outputs;
    for (Value* input : node->GetInputs()) {
        if (input->GetTensor()) {
            inputs.push_back(input->GetTensor().get());
        }
    }
    for (Value* output : node->GetOutputs()) {
        if (output->GetTensor()) {
            outputs.push_back(output->GetTensor().get());
        }
    }
    
    // 3. 执行算子
    return op->Execute(inputs, outputs, ctx);
}
```

**算子执行流程**（以Add算子为例）：

```cpp
Status AddOperator::Execute(const std::vector<Tensor*>& inputs,
                           const std::vector<Tensor*>& outputs,
                           ExecutionContext* ctx) {
    Tensor* input0 = inputs[0];
    Tensor* input1 = inputs[1];
    Tensor* output = outputs[0];
    
    // 获取数据指针
    const float* data0 = static_cast<const float*>(input0->GetData());
    const float* data1 = static_cast<const float*>(input1->GetData());
    float* out_data = static_cast<float*>(output->GetData());
    
    // 执行计算
    size_t count = output->GetElementCount();
    for (size_t i = 0; i < count; ++i) {
        out_data[i] = data0[i] + data1[i];
    }
    
    return Status::Ok();
}
```

---

## 第五步：输出收集

### 代码位置
`src/runtime/runtime.cpp` 的 `ExecutionEngine::ExecuteInternal` 方法

### 核心逻辑

```cpp
// 收集输出
outputs.clear();
for (Value* output_value : graph->GetOutputs()) {
    if (output_value->GetTensor()) {
        outputs.push_back(output_value->GetTensor().get());
    }
}
```

### 关键点说明

1. **输出Value**：图的输出Value在最后一个节点执行时已设置Tensor
2. **Tensor提取**：从输出Value中提取Tensor指针
3. **返回给用户**：将Tensor指针添加到outputs列表

**注意**：返回的是Tensor指针，Tensor的实际数据在Value中，由Value管理生命周期。

---

## 调度器详解

### 调度器类型

InferUnity 支持三种调度器：

#### 1. TopologicalScheduler（拓扑排序调度器）

**特点**：按依赖顺序执行，单线程

```cpp
std::vector<Node*> TopologicalScheduler::GetExecutionOrder(const Graph* graph) const {
    return graph->TopologicalSort();
}
```

**适用场景**：
- 简单模型
- 调试和测试
- 单线程环境

#### 2. ParallelScheduler（并行调度器）

**特点**：并行执行独立的节点

```cpp
class ParallelScheduler : public Scheduler {
    // 将图分为多个层级
    // 同一层级的节点可以并行执行
    // 使用线程池执行
};
```

**执行流程**：
```
层级1: [Conv1, Conv2]  → 并行执行
层级2: [Add]           → 等待层级1完成
层级3: [Relu1, Relu2]  → 并行执行
```

**适用场景**：
- 有独立分支的模型
- 多核CPU环境
- 需要提高吞吐量

#### 3. PipelineScheduler（流水线调度器）

**特点**：将图分为多个阶段，流水线执行

```cpp
class PipelineScheduler : public Scheduler {
    void PartitionGraph(const Graph* graph) {
        // 将图分为多个阶段
        // 每个阶段可以并行处理不同的batch
    }
};
```

**执行流程**：
```
阶段1: [Conv]  处理batch1 → 阶段2: [Relu]  处理batch1
阶段1: [Conv]  处理batch2 → 阶段2: [Relu]  处理batch2
阶段1: [Conv]  处理batch3 → 阶段2: [Relu]  处理batch3
```

**适用场景**：
- 批量推理
- 需要高吞吐量
- 有多个处理单元

---

## 执行提供者选择

### 选择策略

```cpp
ExecutionProvider* ExecutionProviderSelector::SelectProvider(
    Node* node, 
    const std::vector<ExecutionProvider*>& providers) {
    
    // 按优先级顺序尝试
    for (ExecutionProvider* provider : providers) {
        if (provider->SupportsOperator(node->GetOpType())) {
            return provider;  // 返回第一个支持的
        }
    }
    
    return nullptr;  // 没有支持的提供者
}
```

### 支持检查

**CPU后端**：

```cpp
bool CPUExecutionProvider::SupportsOperator(const std::string& op_type) const {
    // 检查支持列表
    static const std::unordered_set<std::string> supported_operators = {
        "Conv", "Relu", "MatMul", "Add", ...
    };
    
    if (supported_operators.find(op_type) != supported_operators.end()) {
        return true;
    }
    
    // 也检查算子注册表
    return OperatorRegistry::Instance().IsRegistered(op_type);
}
```

### 节点分配

在模型加载时，可以为每个节点预先分配执行提供者：

```cpp
// 在LoadAndOptimizeGraph中
Status status = ExecutionProviderSelector::AssignProviders(graph_.get(), provider_ptrs_);
```

**分配逻辑**：
- 遍历所有节点
- 为每个节点选择执行提供者
- 将提供者信息存储在节点中

---

## 完整执行流程图

```
用户调用: session->Run(inputs, outputs)
    ↓
┌─────────────────────────────────────┐
│ 1. InferenceSession::Run            │
│    - 验证模型已加载                  │
│    - 设置执行选项                    │
│    - 调用ExecutionEngine::Execute    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. ExecutionEngine::ExecuteInternal │
│    - 验证输入数量                    │
│    - 绑定输入Tensor到输入Value       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. 获取执行顺序                     │
│    - 调用Scheduler::GetExecutionOrder│
│    - 拓扑排序                        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. 遍历执行每个节点                 │
│    ├─ 选择执行提供者                │
│    ├─ 收集输入张量                  │
│    ├─ 创建输出张量（推断形状）      │
│    ├─ 执行算子                      │
│    └─ 更新输出Value                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. 收集输出                         │
│    - 从输出Value提取Tensor          │
│    - 添加到outputs列表               │
└─────────────────────────────────────┘
    ↓
返回输出张量 ✓
```

---

## 执行示例

假设有一个简单的模型：`Input → Conv → Relu → Output`

**执行过程**：

```
1. 输入绑定
   Input Value → 绑定用户提供的Tensor

2. 拓扑排序
   执行顺序: [Conv, Relu]

3. 执行Conv节点
   - 选择提供者: CPUExecutionProvider
   - 输入: Input Tensor
   - 推断输出形状: [1, 64, 224, 224]
   - 创建输出Tensor
   - 执行Conv算子
   - 设置输出Value的Tensor

4. 执行Relu节点
   - 选择提供者: CPUExecutionProvider
   - 输入: Conv的输出Tensor
   - 推断输出形状: [1, 64, 224, 224]
   - 创建输出Tensor
   - 执行Relu算子
   - 设置输出Value的Tensor

5. 收集输出
   - 从Output Value提取Tensor
   - 返回给用户
```

---

## 关键数据结构

### ExecutionContext

执行上下文，存储执行时的状态信息：

```cpp
class ExecutionContext {
    DeviceType device_type_;      // 设备类型
    void* device_context_;        // 设备上下文（如CUDA stream）
    // ... 其他执行状态 ...
};
```

### ExecutionOptions

执行选项，控制执行行为：

```cpp
struct ExecutionOptions {
    ExecutionMode mode;            // 同步/异步
    bool enable_profiling;         // 是否启用性能分析
    int max_parallel_streams;      // 最大并行流数
    std::string backend_preference;// 后端偏好
};
```

---

## 总结

推理执行流程是一个精心设计的多步骤过程：

1. **输入绑定**：将用户输入绑定到图的输入Value
2. **执行顺序**：通过拓扑排序确定节点执行顺序
3. **节点执行**：按顺序执行每个节点
   - 选择执行提供者
   - 准备输入输出张量
   - 执行算子计算
4. **输出收集**：从输出Value提取结果

整个过程确保了：
- **正确性**：按依赖顺序执行
- **灵活性**：支持多种调度策略
- **可扩展性**：支持多种执行提供者
- **性能**：支持并行和流水线执行

---

**最后更新**: 2026-01-11
