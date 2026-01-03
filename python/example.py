# InferUnity Python API 使用示例
# 参考ONNX Runtime的Python API设计

import numpy as np
import inferunity_py as iu

# 1. 创建推理会话
options = iu.SessionOptions()
options.execution_providers = ["CPUExecutionProvider"]
options.graph_optimization_level = iu.GraphOptimizationLevel.ALL

session = iu.InferenceSession.create(options)

# 2. 加载模型
session.load_model("model.onnx")

# 3. 获取模型信息
input_shapes = session.get_input_shapes()
output_shapes = session.get_output_shapes()
print(f"Input shapes: {input_shapes}")
print(f"Output shapes: {output_shapes}")

# 4. 准备输入（NumPy数组）
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
inputs = [input_data]

# 5. 执行推理
outputs = session.run(inputs)

# 6. 获取输出
for i, output in enumerate(outputs):
    print(f"Output {i}: shape={output.shape}, dtype={output.dtype}")

