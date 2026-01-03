# ONNX集成指南

## 依赖安装

### 方法1: 使用系统Protobuf (推荐)

```bash
# Ubuntu/Debian
sudo apt-get install libprotobuf-dev protobuf-compiler

# macOS
brew install protobuf

# 编译时启用
cmake .. -DUSE_SYSTEM_PROTOBUF=ON
```

### 方法2: 使用FetchContent自动下载

CMake会自动下载protobuf（默认方式）。

### 方法3: 使用vcpkg

```bash
vcpkg install protobuf
cmake .. -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

## ONNX Proto文件

### 获取ONNX Proto文件

```bash
# 克隆ONNX仓库
git clone https://github.com/onnx/onnx.git
cd onnx

# 设置环境变量
export ONNX_PROTO_DIR=$(pwd)
```

### 生成Protobuf代码

CMake会自动检测`third_party/onnx/onnx/onnx.proto`文件并生成代码。

或者手动生成：

```bash
protoc --cpp_out=. onnx/onnx.proto
```

## 支持的基础算子

当前实现支持以下ONNX算子的映射：

- **Conv**: 卷积
- **Relu**: ReLU激活
- **Add**: 加法
- **Mul**: 乘法
- **MatMul**: 矩阵乘法
- **MaxPool**: 最大池化
- **AveragePool**: 平均池化
- **BatchNormalization**: 批归一化
- **Softmax**: Softmax
- **Reshape**: 重塑
- **Transpose**: 转置
- **Concat**: 连接

更多算子将在后续版本中添加。

## 使用示例

```cpp
#include "inferunity/engine.h"
#include "frontend/onnx_parser.h"

// 加载ONNX模型
frontend::ONNXParser parser;
parser.LoadFromFile("model.onnx");

// 转换为内部Graph
std::unique_ptr<Graph> graph;
parser.ConvertToGraph(graph);

// 使用InferenceSession
auto session = InferenceSession::Create();
session->LoadModelFromGraph(std::move(graph));
```

## 注意事项

1. ONNX模型必须符合ONNX标准格式
2. 某些高级特性（如动态形状）可能尚未完全支持
3. 建议使用ONNX Runtime验证模型后再转换

