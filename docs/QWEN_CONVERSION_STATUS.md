# Qwen模型转换状态

## 转换尝试结果

### 尝试1: torch.onnx.export
**结果**: ❌ 失败
**错误**: `Exporting the operator 'aten::__ior_' to ONNX opset version 17 is not supported`
**原因**: PyTorch的ONNX导出器不支持Qwen模型内部使用的某些操作符

### 尝试2: optimum库
**结果**: ❌ 失败  
**错误**: `ImportError: cannot import name 'main_export' from 'optimum.exporters.onnx'`
**原因**: optimum库版本不兼容或安装不完整

### 尝试3: transformers ONNX导出
**结果**: ❌ 失败
**错误**: `qwen2 is not supported yet`
**原因**: transformers库的ONNX导出功能还不支持qwen2模型类型

## 问题分析

`aten::__ior_` 是PyTorch的内部操作符，用于in-place位或运算。这个操作符在Qwen模型的某些实现中被使用，但ONNX标准不支持。

## 解决方案

### 方案1: 使用预转换的ONNX模型（推荐）
- 从HuggingFace Hub下载已转换的ONNX模型
- 或使用其他工具/服务转换

### 方案2: 修改模型代码
- 修改Qwen模型的实现，避免使用不支持的操作符
- 需要深入了解模型架构

### 方案3: 使用更新的工具
- 等待PyTorch更新支持更多操作符
- 或使用其他转换框架（如TensorRT、ONNXRuntime的转换工具）

### 方案4: 先测试简单模型（当前推荐）
- 使用已创建的简单测试模型验证推理引擎
- 确保推理引擎可以正常工作
- 然后再处理Qwen模型转换

## 当前建议

**立即执行**：
1. ✅ 使用已创建的简单测试模型 (`models/test/simple_add.onnx`) 测试推理引擎
2. ✅ 验证推理引擎可以加载和运行ONNX模型
3. ⏸️ 暂时搁置Qwen模型转换，等待更好的转换方案

**后续计划**：
1. 研究Qwen模型的ONNX转换最佳实践
2. 尝试使用HuggingFace的转换服务
3. 或使用其他预转换的模型进行测试

## 测试简单模型

```bash
# 测试简单Add模型
./build/bin/inference_example models/test/simple_add.onnx

# 测试简单Conv模型  
./build/bin/inference_example models/test/simple_conv.onnx
```

这样可以验证推理引擎的基本功能，即使Qwen模型转换暂时无法完成。

