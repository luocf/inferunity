# Qwen2.5-0.5B 快速开始指南

## 自动化测试

我们已经为您创建了自动化脚本，可以一键完成所有步骤：

```bash
# 运行自动化脚本
./scripts/setup_and_test.sh
```

这个脚本会自动：
1. ✅ 检查Python环境
2. ✅ 转换模型为ONNX格式
3. ✅ 编译项目
4. ✅ 检查ONNX模型
5. ✅ 运行推理测试

## 手动步骤（如果需要）

### 1. 转换模型为ONNX

```bash
python3 scripts/convert_qwen_to_onnx.py \
    --model_path models/Qwen2.5-0.5B \
    --output models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx \
    --max_length 128
```

### 2. 编译项目

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
cd ..
```

### 3. 运行测试

```bash
./build/bin/test_qwen models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx
```

## 依赖要求

- Python 3.7+
- PyTorch
- transformers
- CMake 3.10+
- C++17编译器

## 故障排除

### 如果模型转换失败

1. 检查模型路径是否正确
2. 确保安装了PyTorch和transformers：
   ```bash
   pip install torch transformers
   ```

### 如果编译失败

1. 检查CMake版本：`cmake --version`
2. 检查编译器：`g++ --version` 或 `clang++ --version`
3. 查看详细错误信息

### 如果测试失败

1. 检查ONNX模型是否存在
2. 查看错误日志
3. 检查是否缺少算子支持

## 下一步

测试成功后，您可以：
1. 集成tokenizer进行文本推理
2. 实现完整的生成循环
3. 优化性能

