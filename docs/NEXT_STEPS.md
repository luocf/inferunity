# 下一步工作计划

## 当前状态

✅ **编译成功** - 所有组件已编译
✅ **核心功能正常** - Tensor、Graph等核心数据结构工作正常
⚠️ **执行提供者注册问题** - CPUExecutionProvider未自动注册
⚠️ **需要ONNX模型** - 进行完整推理测试

## 优先级任务

### 1. 修复执行提供者注册（高优先级）

**问题**：CPUExecutionProvider的静态注册代码没有被执行

**解决方案**：
- 方案A：在`InferenceSession::Initialize()`中显式调用注册函数
- 方案B：修改注册机制，使用更可靠的初始化方法
- 方案C：在`engine.cpp`中直接引用`cpu_backend.cpp`的符号

**建议**：采用方案A，在`PrepareExecutionProviders()`之前显式初始化

### 2. 转换Qwen2.5-0.5B模型（高优先级）

**当前问题**：
- `torch.onnx.export`遇到`TypeError`和`UnsupportedOperatorError`
- `optimum`库有`ImportError`

**解决方案**：
- 方案A：修复PyTorch ONNX导出问题
  - 检查Qwen2模型的forward方法签名
  - 使用正确的参数调用`torch.onnx.export`
  - 可能需要修改模型包装器
  
- 方案B：使用其他转换工具
  - 使用HuggingFace的`transformers` + `onnxruntime`
  - 使用`onnx`库直接转换
  - 使用预转换的ONNX模型

- 方案C：使用简单的测试模型
  - 创建一个简单的ONNX模型用于测试
  - 使用ONNX官方示例模型

**建议**：先尝试方案C，确保推理引擎可以加载和运行ONNX模型，然后再处理Qwen模型转换

### 3. 测试推理引擎（中优先级）

**需要完成**：
1. 修复执行提供者注册
2. 准备ONNX模型文件
3. 测试模型加载
4. 测试推理执行
5. 验证输出结果

### 4. 完善功能（低优先级）

- 修复Google Test链接问题
- 完善错误处理
- 添加更多测试用例
- 性能优化

## 推荐执行顺序

1. **立即执行**：修复执行提供者注册问题
   - 修改`src/core/engine.cpp`，在`PrepareExecutionProviders()`中显式初始化
   - 测试`InferenceSession::Create()`是否可以成功

2. **然后执行**：创建或获取简单的ONNX测试模型
   - 使用ONNX官方示例或创建一个简单的Add/Mul模型
   - 测试推理引擎可以加载和运行

3. **最后执行**：处理Qwen模型转换
   - 在确保推理引擎工作正常后，再处理复杂的模型转换

## 快速开始

### 修复执行提供者（推荐方案）

在`src/core/engine.cpp`的`PrepareExecutionProviders()`函数开头添加：

```cpp
// 确保CPU执行提供者已注册
// 通过引用一个符号来强制链接cpu_backend
extern void InitializeExecutionProviders();
InitializeExecutionProviders();
```

### 创建简单测试模型

可以使用Python脚本创建一个简单的ONNX模型：

```python
import onnx
from onnx import helper, TensorProto

# 创建一个简单的Add模型
input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3])
input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [2, 3])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

add_node = helper.make_node('Add', ['input1', 'input2'], ['output'])

graph = helper.make_graph([add_node], 'test_model',
                          [input1, input2], [output])
model = helper.make_model(graph)

onnx.save(model, 'test_model.onnx')
```

然后使用这个模型测试推理引擎。

