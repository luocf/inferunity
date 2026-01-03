# 自动化设置指南

## 一键自动化设置

我们已经创建了完整的自动化脚本，可以一键完成所有设置：

```bash
# 运行完整自动化设置（推荐）
./scripts/auto_setup.sh
```

这个脚本会自动：
1. ✅ 创建Python虚拟环境
2. ✅ 安装所有Python依赖（PyTorch, Transformers等）
3. ✅ 转换Qwen2.5-0.5B模型为ONNX格式
4. ✅ 编译C++项目
5. ✅ 检查所有文件
6. ✅ 运行推理测试

## 分步执行

### 步骤1: 设置Python虚拟环境

```bash
# 创建并设置虚拟环境
./scripts/setup_venv.sh

# 或者手动创建
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch transformers numpy
```

### 步骤2: 转换模型（在虚拟环境中）

```bash
# 激活虚拟环境
source venv/bin/activate

# 转换模型
python3 scripts/convert_qwen_to_onnx.py \
    --model_path models/Qwen2.5-0.5B \
    --output models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx \
    --max_length 128
```

### 步骤3: 编译项目

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
cd ..
```

### 步骤4: 运行测试

```bash
./build/bin/test_qwen models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx
```

## 虚拟环境使用

### 激活虚拟环境
```bash
source venv/bin/activate
```

### 退出虚拟环境
```bash
deactivate
```

### 删除虚拟环境（如果需要重新开始）
```bash
rm -rf venv
```

## 故障排除

### 如果虚拟环境创建失败
- 确保已安装Python 3.7+
- 检查是否有`venv`模块：`python3 -m venv --help`

### 如果依赖安装失败
- 检查网络连接
- 尝试使用国内镜像：
  ```bash
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch transformers numpy
  ```

### 如果模型转换失败
- 检查模型文件是否存在：`ls -lh models/Qwen2.5-0.5B/`
- 确保在虚拟环境中运行
- 检查内存是否足够（模型转换需要较多内存）

### 如果编译失败
- 检查CMake版本：`cmake --version`（需要3.10+）
- 检查编译器：`g++ --version` 或 `clang++ --version`
- 查看详细错误信息

## 文件结构

设置完成后，项目结构如下：

```
ifusionengine/
├── venv/                    # Python虚拟环境
├── build/                   # 编译目录
│   └── bin/
│       ├── test_qwen       # 测试程序
│       └── ...
├── models/
│   └── Qwen2.5-0.5B/
│       ├── qwen2.5-0.5b.onnx  # ONNX模型（转换后）
│       └── ...
└── scripts/
    ├── auto_setup.sh        # 完整自动化脚本
    ├── setup_venv.sh        # 虚拟环境设置
    └── convert_qwen_to_onnx.py  # 模型转换脚本
```

## 下一步

设置完成后，您可以：
1. 运行测试验证功能
2. 集成tokenizer进行文本推理
3. 实现完整的生成循环
4. 优化性能

