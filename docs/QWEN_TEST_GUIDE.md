# Qwen2.5-0.5B 模型测试指南

## 模型信息

**Qwen2.5-0.5B** 是一个0.5B参数的Transformer语言模型，具有以下特性：

- **参数量**: 0.49B (非嵌入: 0.36B)
- **层数**: 24
- **注意力头**: GQA (Q: 14, KV: 2)
- **上下文长度**: 32,768 tokens
- **架构特性**:
  - RoPE (旋转位置编码)
  - SwiGLU (激活函数)
  - RMSNorm (归一化)
  - Attention QKV偏置
  - 绑定词嵌入

## InferUnity支持情况

### ✅ 已实现的算子

1. **核心Transformer算子**:
   - ✅ MatMul (Attention和FFN)
   - ✅ Add (残差连接)
   - ✅ LayerNormalization
   - ✅ **RMSNorm** (新增)
   - ✅ Softmax (Attention)

2. **激活函数**:
   - ✅ GELU
   - ✅ **SiLU/Swish** (新增，可用于实现SwiGLU)
   - ✅ ReLU, Sigmoid, Tanh

3. **形状操作**:
   - ✅ Reshape, Transpose
   - ✅ Concat, Split
   - ✅ Gather, Slice

4. **词嵌入**:
   - ✅ **Embedding** (新增)

### ⚠️ 可能需要的算子组合

1. **RoPE (旋转位置编码)**:
   - 通常由多个算子组合实现
   - 可能需要: Cos, Sin, Mul, Add等

2. **SwiGLU**:
   - 可由 SiLU + Mul 组合实现
   - 或需要单独的SwiGLU算子

3. **Attention机制**:
   - 通常由 MatMul + Softmax + MatMul 组合
   - 可能需要: Scale, Mask等操作

## 测试步骤

### 1. 准备模型

首先需要将Qwen2.5-0.5B模型导出为ONNX格式：

```python
# 使用transformers库导出ONNX
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B")
model.eval()

# 导出为ONNX
dummy_input = torch.randint(0, 1000, (1, 128))  # [batch, seq_len]
torch.onnx.export(
    model,
    dummy_input,
    "qwen2.5-0.5b.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
    opset_version=14
)
```

### 2. 编译项目

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
```

### 3. 检查模型算子

```bash
# 使用工具检查模型使用的算子
./bin/inferunity_tool info qwen2.5-0.5b.onnx
```

### 4. 运行测试

```bash
# 运行Qwen2.5-0.5B测试程序
./bin/test_qwen qwen2.5-0.5b.onnx
```

测试程序会：
1. 加载模型
2. 显示模型信息（输入/输出形状）
3. 检查算子支持情况
4. 执行推理（使用示例输入）
5. 显示输出结果
6. 性能分析

### 5. 处理可能的错误

如果遇到不支持的算子，可以：

1. **检查算子列表**:
   ```bash
   ./bin/inferunity_tool info qwen2.5-0.5b.onnx | grep -i "op_type"
   ```

2. **添加缺失算子**:
   - 查看ONNX算子定义
   - 在`src/operators/`中实现对应的Operator类
   - 使用`REGISTER_OPERATOR`注册
   - 更新`src/backends/cpu_backend.cpp`的支持列表

3. **常见缺失算子**:
   - **Cos/Sin**: 用于RoPE
   - **Pow**: 幂运算
   - **ReduceMean/ReduceSum**: 归约操作
   - **Where/Select**: 条件选择

## 预期结果

### 成功情况

如果模型加载和推理成功，应该看到：

```
========================================
  Qwen2.5-0.5B 模型测试
========================================
✅ 会话创建成功
✅ 模型加载成功
✅ 推理成功
✅ 输出结果
```

### 可能的问题

1. **不支持的算子**:
   - 错误信息会显示缺失的算子
   - 需要实现对应的算子

2. **输入形状不匹配**:
   - 检查模型期望的输入形状
   - 确保输入数据格式正确

3. **内存不足**:
   - 0.5B模型在CPU上需要足够内存
   - 建议至少8GB RAM

## 性能优化建议

1. **使用小batch size**: CPU推理建议batch_size=1
2. **限制序列长度**: 可以先用较短的序列测试
3. **启用图优化**: 已默认启用
4. **多线程**: 已自动检测CPU核心数

## 下一步

如果测试成功，可以：

1. **集成tokenizer**: 添加文本到token的转换
2. **实现完整推理流程**: 包括生成循环
3. **性能优化**: 针对热点算子进行优化
4. **添加更多算子**: 按需实现缺失的算子

## 参考

- [Qwen2.5模型文档](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- [ONNX导出指南](https://huggingface.co/docs/transformers/serialization)
- [InferUnity API文档](docs/API.md)

