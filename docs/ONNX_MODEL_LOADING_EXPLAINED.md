# ONNX 模型加载流程详解

本文档详细讲解 InferUnity 中 ONNX 模型加载的完整流程，包括代码逻辑和实现细节。

## 目录

1. [整体流程概览](#整体流程概览)
2. [第一步：入口函数 LoadModel](#第一步入口函数-loadmodel)
3. [第二步：文件读取和 Protobuf 解析](#第二步文件读取和-protobuf-解析)
4. [第三步：转换为内部 Graph](#第三步转换为内部-graph)
5. [第四步：图验证和形状推断](#第四步图验证和形状推断)
6. [第五步：图优化](#第五步图优化)
7. [第六步：执行提供者分配](#第六步执行提供者分配)

---

## 整体流程概览

从用户调用 `session->LoadModel("model.onnx")` 到模型就绪，主要经过以下步骤：

```
用户调用 LoadModel("model.onnx")
    ↓
1. 文件读取和格式检测
    ↓
2. ONNX Protobuf 解析
    ↓
3. 转换为内部 Graph
    ↓
4. 图验证和形状推断
    ↓
5. 图优化
    ↓
6. 执行提供者分配
    ↓
模型就绪，可以推理
```

---

## 第一步：入口函数 LoadModel

### 代码位置
`src/core/engine.cpp` 的 `InferenceSession::LoadModel` 方法

### 核心逻辑

```cpp
Status InferenceSession::LoadModel(const std::string& filepath) {
    // 1. 根据文件扩展名选择解析器
    std::string ext = filepath.substr(filepath.find_last_of(".") + 1);
    
    if (ext == "onnx") {
        // 2. 创建ONNX解析器
        frontend::ONNXParser parser;
        
        // 3. 从文件加载
        Status status = parser.LoadFromFile(filepath);
        if (!status.IsOk()) {
            return status;
        }
        
        // 4. 转换为内部Graph
        std::unique_ptr<Graph> graph;
        status = parser.ConvertToGraph(graph);
        if (!status.IsOk()) {
            return status;
        }
        
        // 5. 加载并优化图
        return LoadModelFromGraph(std::move(graph));
    }
    
    return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                       "Unsupported model format: " + ext);
}
```

### 关键点说明

1. **格式检测**：通过文件扩展名（`.onnx`）判断模型格式
2. **解析器创建**：创建 `ONNXParser` 实例
3. **文件加载**：调用 `LoadFromFile` 读取并解析文件
4. **图转换**：将 ONNX 格式转换为内部 `Graph` 结构
5. **后续处理**：调用 `LoadModelFromGraph` 完成验证、优化等步骤

### 设计思路

- **可扩展性**：通过扩展名判断，便于后续支持其他格式（如 TensorFlow Lite）
- **职责分离**：解析器负责格式转换，Session 负责后续处理
- **错误处理**：每步都有状态检查，失败立即返回

---

## 第二步：文件读取和 Protobuf 解析

### 代码位置
`src/frontend/onnx_parser_impl.cpp` 的 `ONNXParser::LoadFromFile` 和 `LoadFromMemory` 方法

### 核心逻辑

#### 2.1 文件读取

```cpp
Status ONNXParser::LoadFromFile(const std::string& filepath) {
    // 1. 打开文件（二进制模式）
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return Status::Error(StatusCode::ERROR_NOT_FOUND,
                           "Cannot open ONNX file: " + filepath);
    }
    
    // 2. 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 3. 读取文件内容到内存
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    // 4. 调用内存加载方法
    return LoadFromMemory(data.data(), data.size());
}
```

**要点**：
- 使用二进制模式读取（ONNX 是 Protobuf 格式）
- 先获取文件大小，再一次性读取，避免多次 I/O
- 将文件内容转为内存数据，统一处理

#### 2.2 Protobuf 解析

```cpp
Status ONNXParser::LoadFromMemory(const void* data, size_t size) {
    // 使用 ONNX Protobuf 解析
    onnx::ModelProto model;
    if (!model.ParseFromArray(data, size)) {
        return Status::Error(StatusCode::ERROR_INVALID_MODEL,
                            "Failed to parse ONNX model");
    }
    
    // 转换为简化格式以便处理
    auto* simple_model = new SimpleONNXModel();
    simple_model->model_version = model.model_version();
    
    // 解析输入输出
    const auto& graph = model.graph();
    // ... 解析逻辑 ...
}
```

### 解析的关键数据结构

ONNX 模型的主要组成部分：

1. **ModelProto**：顶层模型结构
   - `model_version`：模型版本
   - `graph`：计算图

2. **GraphProto**：计算图结构
   - `input`：输入信息（名称、形状、类型）
   - `output`：输出信息
   - `node`：计算节点
   - `initializer`：初始值（权重和偏置）

3. **NodeProto**：节点信息
   - `name`：节点名称
   - `op_type`：算子类型（如 "Conv", "Relu"）
   - `input`：输入值名称列表
   - `output`：输出值名称列表
   - `attribute`：节点属性（如 Conv 的 kernel_size, stride）

### 输入解析示例

```cpp
// 解析输入信息
for (const auto& input : graph.input()) {
    SimpleONNXModel::InputInfo input_info;
    input_info.name = input.name();
    
    if (input.type().has_tensor_type()) {
        const auto& tensor_type = input.type().tensor_type();
        input_info.data_type = tensor_type.elem_type();
        
        // 解析形状维度
        if (tensor_type.has_shape()) {
            const auto& shape = tensor_type.shape();
            for (const auto& dim : shape.dim()) {
                if (dim.has_dim_value()) {
                    // 静态维度：具体数值
                    input_info.dims.push_back(dim.dim_value());
                } else if (dim.has_dim_param()) {
                    // 动态维度：使用 -1 表示
                    input_info.dims.push_back(-1);
                }
            }
        }
    }
    simple_model->input_infos.push_back(input_info);
}
```

**关键点**：
- **静态维度**：`dim_value()` 提供具体数值，如 `[1, 3, 224, 224]`
- **动态维度**：`dim_param()` 表示运行时确定，用 `-1` 标记
- **数据类型**：从 `elem_type()` 获取（如 FLOAT32=1）

### 初始值（权重）解析

```cpp
// 解析初始值（权重和偏置）
for (const auto& init : graph.initializer()) {
    SimpleONNXModel::Tensor tensor;
    tensor.name = init.name();
    
    // 解析形状
    for (int64_t dim : init.dims()) {
        tensor.dims.push_back(dim);
    }
    
    // 解析数据类型
    tensor.data_type = init.data_type();
    
    // 复制原始数据（权重值）
    tensor.raw_data = std::vector<uint8_t>(
        init.raw_data().begin(), init.raw_data().end());
    
    simple_model->initializers.push_back(tensor);
}
```

**要点**：
- **初始值**：模型的权重和偏置，在推理时不变
- **数据存储**：以字节数组形式存储，需要根据数据类型解析
- **名称映射**：通过名称与节点输入关联

---

## 第三步：转换为内部 Graph

### 代码位置
`src/frontend/onnx_parser_impl.cpp` 的 `ONNXParser::ConvertToGraph` 方法

### 核心逻辑

这是最复杂的一步，需要将 ONNX 的图结构转换为内部的 Graph 结构。

#### 3.1 整体流程

```cpp
Status ONNXParser::ConvertToGraph(std::unique_ptr<Graph>& graph) {
    // 1. 创建内部Graph
    graph = std::make_unique<Graph>();
    
    // 2. 创建名称到Value的映射表
    std::unordered_map<std::string, Value*> name_to_value;
    
    // 3. 先处理初始值（权重）
    // 4. 再处理图输入
    // 5. 然后处理节点
    // 6. 最后设置图输出
    // 7. 验证图
}
```

### 3.2 处理初始值（权重）

**为什么先处理初始值？**

在 ONNX 中，权重可能同时出现在 `graph.input` 和 `graph.initializer` 中。我们需要先创建初始值的 Value，这样后续节点引用时就能找到。

```cpp
// 先创建初始值（权重）的Value节点
std::unordered_set<std::string> initializer_names;
for (const auto& init : simple_model->initializers) {
    // 1. 创建Value
    Value* value = graph->AddValue();
    value->SetId(value_id++);
    
    // 2. 创建Tensor并填充数据
    Shape shape(init.dims);
    DataType dtype = ConvertDataType(init.data_type);
    auto tensor = CreateTensor(shape, dtype);
    
    // 3. 复制权重数据
    if (!init.raw_data.empty()) {
        std::memcpy(tensor->GetData(), init.raw_data.data(),
                   std::min(init.raw_data.size(), tensor->GetSizeInBytes()));
    }
    
    // 4. 关联Tensor到Value
    value->SetTensor(tensor);
    
    // 5. 记录名称映射
    name_to_value[init.name] = value;
    initializer_names.insert(init.name);
}
```

**关键点**：
- **数据复制**：将 ONNX 的原始字节数据复制到 Tensor
- **类型转换**：使用 `ConvertDataType` 将 ONNX 类型转换为内部类型
- **名称映射**：建立名称到 Value 的映射，供后续节点引用

### 3.3 处理图输入

**区分真正的输入和初始值**：

```cpp
// 创建图输入Value节点（排除已经是初始值的）
for (size_t i = 0; i < simple_model->input_names.size(); ++i) {
    const std::string& input_name = simple_model->input_names[i];
    
    // 如果输入名称已经在初始值中，跳过
    if (initializer_names.find(input_name) != initializer_names.end()) {
        // 这是权重，已经创建了Value
        continue;
    }
    
    // 这是真正的图输入（运行时提供数据）
    Value* value = graph->AddValue();
    
    // 如果输入有明确的形状，创建Tensor
    bool has_concrete_shape = true;
    for (int64_t dim : input_info.dims) {
        if (dim < 0) {  // 动态维度
            has_concrete_shape = false;
            break;
        }
    }
    
    if (has_concrete_shape && !input_info.dims.empty()) {
        Shape input_shape(input_info.dims);
        DataType input_dtype = ConvertDataType(input_info.data_type);
        auto input_tensor = CreateTensor(input_shape, input_dtype);
        value->SetTensor(input_tensor);
    }
    
    name_to_value[input_name] = value;
    graph->AddInput(value);  // 标记为图输入
}
```

**关键点**：
- **区分输入类型**：真正的输入 vs 权重（初始值）
- **形状处理**：有明确形状时创建 Tensor，动态形状时留空
- **输入标记**：调用 `graph->AddInput` 标记为图输入

### 3.4 处理节点

**构建计算图的核心**：

```cpp
// 解析节点，构建计算图
for (const auto& onnx_node : simple_model->nodes) {
    // 1. 创建节点
    Node* node = graph->AddNode(onnx_node.op_type, onnx_node.name);
    
    // 2. 设置属性（如Conv的kernel_size, stride等）
    for (const auto& attr : onnx_node.attributes) {
        node->SetAttribute(attr.first, attr.second);
    }
    
    // 3. 连接输入
    for (const std::string& input_name : onnx_node.inputs) {
        if (input_name.empty()) continue;  // 跳过空输入
        
        auto it = name_to_value.find(input_name);
        if (it != name_to_value.end()) {
            // 找到已存在的Value（输入、权重或之前节点的输出）
            node->AddInput(it->second);
        } else {
            // 输入不存在，创建新的Value（中间结果）
            Value* value = graph->AddValue();
            value->SetId(value_id++);
            name_to_value[input_name] = value;
            node->AddInput(value);
        }
    }
    
    // 4. 创建输出Value
    for (const std::string& output_name : onnx_node.outputs) {
        Value* value = graph->AddValue();
        value->SetId(value_id++);
        name_to_value[output_name] = value;
        node->AddOutput(value);
    }
}
```

**关键点**：
- **节点创建**：使用算子类型（如 "Conv", "Relu"）创建节点
- **属性设置**：将 ONNX 属性转换为节点属性
- **输入连接**：
  - 如果 Value 已存在（输入/权重/之前输出），直接连接
  - 如果不存在，创建新 Value（中间结果）
- **输出创建**：为每个输出创建新的 Value

### 3.5 设置图输出

```cpp
// 设置图输出
for (const std::string& output_name : simple_model->output_names) {
    auto it = name_to_value.find(output_name);
    if (it != name_to_value.end()) {
        graph->AddOutput(it->second);  // 标记为图输出
    }
}
```

### 3.6 图验证

```cpp
// 验证图的完整性
Status status = graph->Validate();
if (!status.IsOk()) {
    return status;
}
```

**验证内容**：
- 所有节点都有输入输出
- 图有输入和输出
- 没有孤立节点
- 数据流连接正确

### 转换示例

假设有一个简单的 ONNX 模型：`Input → Conv → Relu → Output`

转换过程：

```
ONNX格式：
  input: "x" (Shape: [1,3,224,224])
  node1: Conv(input="x", weight="w", output="conv_out")
  node2: Relu(input="conv_out", output="relu_out")
  output: "relu_out"

转换为内部Graph：
  Value: "x" (Input) → Node: Conv → Value: "conv_out"
  Value: "conv_out" → Node: Relu → Value: "relu_out" (Output)
  Value: "w" (Initializer/Weight)
```

---

## 第四步：图验证和形状推断

### 代码位置
`src/core/engine.cpp` 的 `InferenceSession::LoadAndOptimizeGraph` 方法

### 4.1 图验证

```cpp
// 验证图
Status status = graph_->Validate();
if (!status.IsOk()) {
    return status;
}
```

**验证内容**：
- 图有输入和输出
- 所有节点连接正确
- 没有循环依赖（DAG检查）

### 4.2 形状推断

**为什么需要形状推断？**

ONNX 模型可能只包含输入形状，中间节点和输出的形状需要根据算子特性推断。

```cpp
// 形状推断（参考ONNX Runtime的ShapeInference）
status = InferShapes(graph_.get());
if (!status.IsOk()) {
    // 形状推断失败不影响加载，只记录警告
    LOG_WARNING("Shape inference failed: " + status.Message());
}
```

**形状推断流程**（在 `src/core/shape_inference.cpp`）：

```cpp
Status InferShapes(Graph* graph) {
    // 1. 拓扑排序
    std::vector<Node*> nodes = graph->TopologicalSort();
    
    // 2. 按顺序推断每个节点的输出形状
    for (Node* node : nodes) {
        // 获取算子
        auto op = OperatorRegistry::Instance().Create(node->GetOpType());
        
        // 准备输入张量（用于形状推断）
        std::vector<Tensor*> input_tensors;
        for (Value* input : node->GetInputs()) {
            if (input->GetTensor()) {
                input_tensors.push_back(input->GetTensor().get());
            }
        }
        
        // 调用算子的形状推断方法
        std::vector<Shape> output_shapes;
        op->InferOutputShape(input_tensors, output_shapes);
        
        // 设置输出Value的形状
        for (size_t i = 0; i < node->GetOutputs().size(); ++i) {
            Value* output = node->GetOutputs()[i];
            if (!output->GetTensor()) {
                // 创建Tensor并设置形状
                auto tensor = CreateTensor(output_shapes[i], DataType::FLOAT32);
                output->SetTensor(tensor);
            }
        }
    }
}
```

**关键点**：
- **拓扑排序**：按依赖顺序处理节点
- **算子推断**：每个算子实现 `InferOutputShape` 方法
- **形状传播**：从输入到输出逐层推断

**示例**：
```
Input: [1, 3, 224, 224]
  ↓ Conv (kernel=3x3, stride=1, padding=1, out_channels=64)
Conv输出: [1, 64, 224, 224]  ← 通过形状推断得到
  ↓ Relu (不改变形状)
Relu输出: [1, 64, 224, 224]  ← 通过形状推断得到
```

---

## 第五步：图优化

### 代码位置
`src/core/engine.cpp` 的 `InferenceSession::LoadAndOptimizeGraph` 方法

### 5.1 优化流程

```cpp
// 优化图 (参考ONNX Runtime的图优化级别)
if (options_.graph_optimization_level != SessionOptions::GraphOptimizationLevel::NONE) {
    status = optimizer_->Optimize(graph_.get());
    if (!status.IsOk()) {
        return status;
    }
}
```

**优化级别**：
- `NONE`：不优化
- `BASIC`：基础优化（常量折叠、死代码消除）
- `EXTENDED`：扩展优化（算子融合）
- `ALL`：全部优化

### 5.2 优化Pass列表

在 `InferenceSession::Initialize()` 中注册的优化Pass：

```cpp
// 注册默认优化Pass
optimizer_->RegisterPass(std::make_unique<ConstantFoldingPass>());      // 常量折叠
optimizer_->RegisterPass(std::make_unique<DeadCodeEliminationPass>());  // 死代码消除
if (options_.enable_operator_fusion) {
    optimizer_->RegisterPass(std::make_unique<OperatorFusionPass>());   // 算子融合
}
optimizer_->RegisterPass(std::make_unique<MemoryLayoutOptimizationPass>()); // 内存布局优化
```

### 5.3 优化Pass详解

#### 常量折叠（Constant Folding）

**目的**：在编译时计算常量表达式，减少运行时计算

**示例**：
```
优化前：
  const1 = 2.0
  const2 = 3.0
  add_result = Add(const1, const2)  // 运行时计算 2+3=5

优化后：
  add_result = 5.0  // 直接使用结果
```

**实现逻辑**：
1. 遍历所有节点
2. 检查所有输入是否为常量
3. 如果是，执行算子计算
4. 将结果替换为常量节点

#### 死代码消除（Dead Code Elimination）

**目的**：删除不影响输出的节点

**示例**：
```
优化前：
  x → Conv → y → Relu → z (输出)
  x → Add → w  (w不被使用)

优化后：
  x → Conv → y → Relu → z (输出)
  (Add节点被删除)
```

#### 算子融合（Operator Fusion）

**目的**：将多个连续算子合并为一个，减少内存访问和函数调用

**示例**：
```
优化前：
  Input → Conv → BatchNorm → Relu → Output
  (3个节点，3次内存访问)

优化后：
  Input → FusedConvBNReLU → Output
  (1个节点，1次内存访问)
```

**融合模式**：
- `Conv + BatchNorm + ReLU` → `FusedConvBNReLU`
- `MatMul + Add` → `FusedMatMulAdd`
- `Conv + ReLU` → `FusedConvReLU`
- `BatchNorm + ReLU` → `FusedBNReLU`

**融合逻辑**（以 Conv+BN+ReLU 为例）：

```cpp
// 1. 查找融合模式
if (node1->GetOpType() == "Conv") {
    Value* output1 = node1->GetOutputs()[0];
    Node* node2 = output1->GetConsumers()[0];
    if (node2->GetOpType() == "BatchNormalization") {
        Value* output2 = node2->GetOutputs()[0];
        Node* node3 = output2->GetConsumers()[0];
        if (node3->GetOpType() == "Relu") {
            // 找到融合模式，执行融合
            FuseConvBNReLU(graph, node1, node2, node3);
        }
    }
}

// 2. 融合步骤
Status FuseConvBNReLU(Graph* graph, Node* conv, Node* bn, Node* relu) {
    // 创建融合节点
    Node* fused = graph->AddNode("FusedConvBNReLU", 
                                  conv->GetInputs(), 
                                  relu->GetOutputs());
    
    // 连接输入输出
    // ...
    
    // 删除旧节点
    graph->RemoveNode(conv);
    graph->RemoveNode(bn);
    graph->RemoveNode(relu);
}
```

#### 内存布局优化（Memory Layout Optimization）

**目的**：优化数据在内存中的排列方式，提高缓存命中率

**优化内容**：
- 确保Tensor使用对齐的内存（16/64字节对齐）
- 优化NCHW vs NHWC布局选择
- 减少内存碎片

### 5.4 优化效果

优化后的图通常具有：
- **更少的节点**：融合减少了节点数量
- **更少的内存访问**：融合减少了中间结果存储
- **更好的缓存局部性**：内存布局优化提高了缓存命中率

---

## 第六步：执行提供者分配

### 代码位置
`src/core/engine.cpp` 的 `InferenceSession::LoadAndOptimizeGraph` 方法

### 6.1 分配执行提供者

**目的**：为每个节点选择合适的执行后端（CPU、CUDA等）

```cpp
// 分配执行提供者 (参考ONNX Runtime的节点分配)
std::vector<ExecutionProvider*> provider_ptrs;
for (const auto& provider : execution_providers_) {
    provider_ptrs.push_back(provider.get());
}
status = ExecutionProviderSelector::AssignProviders(graph_.get(), provider_ptrs);
```

**分配策略**：
1. 检查每个执行提供者是否支持该算子
2. 选择第一个支持的提供者
3. 将节点标记为该提供者

**示例**：
```
节点: Conv
  - CPUExecutionProvider: 支持 ✓
  - CUDAExecutionProvider: 支持 ✓（如果有CUDA）
  → 选择: CPUExecutionProvider（按优先级）
```

### 6.2 准备执行

**目的**：让执行提供者准备执行环境

```cpp
// 准备执行
for (const auto& provider : execution_providers_) {
    status = provider->PrepareExecution(graph_.get());
    if (!status.IsOk()) {
        return status;
    }
}
```

**准备内容**：
- 分配设备内存（如CUDA）
- 编译算子（如TensorRT）
- 预分配缓冲区
- 初始化执行上下文

### 6.3 执行提供者优先级

在 `SessionOptions` 中配置：

```cpp
SessionOptions options;
options.execution_providers = {
    "CUDAExecutionProvider",  // 优先使用CUDA
    "CPUExecutionProvider"     // 回退到CPU
};
```

**分配逻辑**：
- 按列表顺序尝试
- 如果第一个不支持，尝试下一个
- 如果都不支持，返回错误

---

## 完整流程图

```
用户调用: session->LoadModel("model.onnx")
    ↓
┌─────────────────────────────────────┐
│ 1. LoadModel (engine.cpp)           │
│    - 检测文件格式                    │
│    - 创建ONNXParser                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. LoadFromFile (onnx_parser_impl)  │
│    - 读取文件到内存                  │
│    - 调用LoadFromMemory              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. LoadFromMemory (onnx_parser_impl)│
│    - 解析Protobuf                    │
│    - 提取输入/输出/节点/权重         │
│    - 转换为SimpleONNXModel          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. ConvertToGraph (onnx_parser_impl) │
│    - 创建内部Graph                   │
│    - 处理初始值（权重）              │
│    - 处理图输入                      │
│    - 处理节点                        │
│    - 设置图输出                      │
│    - 验证图                          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 5. LoadModelFromGraph (engine.cpp)  │
│    - 图验证                          │
│    - 形状推断                        │
│    - 图优化                          │
│    - 执行提供者分配                  │
│    - 准备执行                        │
└─────────────────────────────────────┘
    ↓
模型就绪，可以推理 ✓
```

---

## 关键数据结构转换

### ONNX → 内部Graph

| ONNX结构 | 内部结构 | 说明 |
|---------|---------|------|
| `GraphProto.input` | `Graph::inputs_` (Value列表) | 图输入 |
| `GraphProto.output` | `Graph::outputs_` (Value列表) | 图输出 |
| `GraphProto.node` | `Graph::nodes_` (Node列表) | 计算节点 |
| `GraphProto.initializer` | `Value::tensor_` (Tensor) | 权重数据 |
| `NodeProto.op_type` | `Node::op_type_` | 算子类型 |
| `NodeProto.input` | `Node::inputs_` (Value列表) | 节点输入 |
| `NodeProto.output` | `Node::outputs_` (Value列表) | 节点输出 |
| `NodeProto.attribute` | `Node::attributes_` | 节点属性 |

---

## 总结

ONNX 模型加载是一个多步骤的过程：

1. **文件读取**：从磁盘读取ONNX文件
2. **Protobuf解析**：解析ONNX格式的二进制数据
3. **图转换**：将ONNX图结构转换为内部Graph
4. **形状推断**：推断所有节点的输出形状
5. **图优化**：应用各种优化Pass提升性能
6. **提供者分配**：为节点分配执行后端

每个步骤都有明确的职责，通过状态码进行错误处理，确保加载过程的可靠性。

---

**最后更新**: 2026-01-11
