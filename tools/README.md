# InferUnity 工具集

本目录包含 InferUnity 推理引擎的各种工具。

## 工具列表

### 1. inferunity_convert - 模型转换工具

用于转换和验证 ONNX 模型。

**用法：**
```bash
# 验证模型
inferunity_convert validate model.onnx

# 转换模型格式
inferunity_convert convert input.onnx output.ifusion

# 打印模型信息
inferunity_convert info model.onnx
```

**功能：**
- 验证 ONNX 模型的有效性
- 将 ONNX 模型转换为内部格式
- 显示模型的输入输出信息

### 2. inferunity_profiler - 性能分析器

用于分析模型的逐层性能。

**用法：**
```bash
# 基本分析
inferunity_profiler model.onnx

# 详细模式（显示节点级信息）
inferunity_profiler model.onnx -v

# 导出CSV
inferunity_profiler model.onnx -o profile.csv

# 指定迭代次数
inferunity_profiler model.onnx -i 50
```

**功能：**
- 逐层性能分析
- 内存使用分析
- 算子类型统计
- CSV格式导出

### 3. inferunity_benchmark - 性能基准测试

用于测试单个模型的推理性能。

**用法：**
```bash
# 基本测试
inferunity_benchmark model.onnx

# 指定warmup和测试迭代次数
inferunity_benchmark model.onnx 10 100
```

**输出：**
- 平均/最小/最大执行时间
- 吞吐量（inferences/sec）
- 内存使用统计

### 4. inferunity_benchmark_suite - 基准测试套件

用于批量测试多个模型，支持性能对比和回归测试。

**用法：**

#### 运行基准测试
```bash
# 从模型列表文件运行测试
inferunity_benchmark_suite run models.txt results.csv
```

模型列表文件格式（models.txt）：
```
# 注释行以#开头
model1.onnx
model2.onnx
model3.onnx
```

#### 性能对比
```bash
# 对比当前结果与基线
inferunity_benchmark_suite compare current_results.csv baseline_results.csv
```

#### 回归测试
```bash
# 运行回归测试（检查性能是否退化）
inferunity_benchmark_suite regression current_results.csv baseline_results.csv
```

**功能：**
- 批量模型测试
- 性能对比分析
- 自动回归测试
- CSV结果导出

## 使用示例

### 完整工作流

1. **验证模型**
```bash
inferunity_convert validate my_model.onnx
```

2. **性能分析**
```bash
inferunity_profiler my_model.onnx -v -o profile.csv
```

3. **基准测试**
```bash
inferunity_benchmark my_model.onnx 10 100
```

4. **批量测试**
```bash
# 创建模型列表
echo "model1.onnx" > models.txt
echo "model2.onnx" >> models.txt

# 运行测试
inferunity_benchmark_suite run models.txt baseline.csv

# 后续修改代码后，再次测试并对比
inferunity_benchmark_suite run models.txt current.csv
inferunity_benchmark_suite compare current.csv baseline.csv

# 运行回归测试
inferunity_benchmark_suite regression current.csv baseline.csv
```

## 输出格式

### CSV格式

所有工具都支持CSV格式输出，便于后续分析和可视化。

**基准测试结果格式：**
```csv
Model,Avg Time (ms),Min Time (ms),Max Time (ms),Throughput (inferences/sec),Memory (MB),Status
model1.onnx,10.5,9.8,11.2,95.2,128.5,SUCCESS
```

**性能分析结果格式：**
```csv
Node Name,Op Type,Execution Time (ms),Memory Used (bytes),Time %
conv1,Conv,2.5,1048576,25.0
relu1,Relu,0.1,1048576,1.0
```

## 注意事项

1. **模型路径**：确保模型文件路径正确
2. **内存限制**：大模型可能需要较多内存
3. **迭代次数**：更多迭代次数可以获得更准确的结果，但耗时更长
4. **Warmup**：建议至少进行5-10次warmup迭代

## 参考

更多信息请参考：
- [API文档](../docs/API.md)
- [实现计划](../docs/IMPLEMENTATION_PLAN.md)

