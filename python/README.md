# InferUnity Python 绑定

使用 pybind11 提供的 Python 绑定，参考 ONNX Runtime 的 Python API 设计。

## 构建

```bash
cd build
cmake .. -DBUILD_PYTHON=ON
make -j$(nproc)
```

构建完成后，Python 模块将位于 `build/python/` 目录。

## 安装

```bash
# 将模块添加到 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# 或者安装到系统
python setup.py install  # TODO: 创建setup.py
```

## 使用示例

```python
import numpy as np
import inferunity_py as iu

# 创建会话
options = iu.SessionOptions()
options.execution_providers = ["CPUExecutionProvider"]
session = iu.InferenceSession.create(options)

# 加载模型
session.load_model("model.onnx")

# 准备输入
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 执行推理
outputs = session.run([input_data])

# 获取输出
print(f"Output shape: {outputs[0].shape}")
```

## API 参考

### InferenceSession

主要的推理会话类，用于加载模型和执行推理。

- `create(options)` - 创建会话
- `load_model(filepath)` - 加载模型
- `run(inputs)` - 执行推理
- `get_input_shapes()` - 获取输入形状
- `get_output_shapes()` - 获取输出形状

### Tensor

张量类，支持与 NumPy 数组的自动转换。

- `get_shape()` - 获取形状
- `get_data()` - 获取数据（返回 NumPy 数组）
- `get_element_count()` - 获取元素数量

### SessionOptions

会话配置选项。

- `execution_providers` - 执行提供者列表
- `graph_optimization_level` - 图优化级别
- `num_threads` - 线程数
- `enable_profiling` - 是否启用性能分析

## 依赖

- Python 3.6+
- NumPy
- pybind11 (通过 CMake FetchContent 自动下载)

## 注意事项

1. 确保 NumPy 数组的数据类型为 `float32`
2. 输入数组的形状必须与模型输入形状匹配
3. 输出数组与输入 Tensor 共享内存，注意生命周期管理

