# ONNX Runtime 安装指南

## macOS 安装方式

### 方式1: 使用 Homebrew (推荐)

```bash
# 安装 ONNX Runtime
brew install onnxruntime

# 验证安装
brew list onnxruntime
```

### 方式2: 使用 Python pip (仅Python绑定)

```bash
# 安装 Python 版本的 ONNX Runtime
pip install onnxruntime

# 或安装 GPU 版本（如果有 NVIDIA GPU）
pip install onnxruntime-gpu
```

**注意**: Python 版本主要用于 Python 绑定，C++ 项目需要 C++ 库。

### 方式3: 从源码编译 (完整控制)

```bash
# 克隆 ONNX Runtime 仓库
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# macOS 编译
./build.sh --config Release --build_shared_lib --parallel

# 编译后的库位置
# build/MacOS/Release/libonnxruntime.dylib
# build/MacOS/Release/include/
```

## 配置 CMake

### 使用 Homebrew 安装的版本

```bash
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_DIR=/opt/homebrew/lib/cmake/onnxruntime
```

### 使用源码编译的版本

```bash
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_DIR=/path/to/onnxruntime/build/MacOS/Release/cmake
```

### 手动指定路径

```bash
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_INCLUDE_DIRS=/path/to/include \
         -Donnxruntime_LIBRARIES=/path/to/lib/libonnxruntime.dylib
```

## 验证安装

编译项目后，检查是否启用了 ONNX Runtime 后端：

```bash
cd build
cmake .. -DENABLE_ONNXRUNTIME=ON
make -j$(sysctl -n hw.ncpu)

# 运行测试
./bin/inference_example models/test/simple_add.onnx
```

## 常见问题

### 1. 找不到 ONNX Runtime

**错误**: `ONNX Runtime not found`

**解决**: 
- 确保已安装 ONNX Runtime
- 使用 `-Donnxruntime_DIR` 指定路径
- 检查 `CMAKE_PREFIX_PATH` 环境变量

### 2. 链接错误

**错误**: `undefined reference to Ort::...`

**解决**:
- 确保链接了正确的库文件
- 检查库文件路径是否正确
- 确保使用相同架构（x86_64 或 arm64）

### 3. 运行时错误

**错误**: `Library not loaded: @rpath/libonnxruntime.dylib`

**解决**:
```bash
# 设置动态库路径
export DYLD_LIBRARY_PATH=/path/to/onnxruntime/lib:$DYLD_LIBRARY_PATH

# 或使用 install_name_tool 修改库路径
install_name_tool -change @rpath/libonnxruntime.dylib \
    /path/to/onnxruntime/lib/libonnxruntime.dylib \
    build/bin/inference_example
```

## 当前状态

- ✅ ONNX Runtime 后端代码已实现 (`src/backends/onnxruntime_backend.cpp`)
- ⏳ 需要安装 ONNX Runtime 库才能编译
- ⏳ 安装后可通过 `-DENABLE_ONNXRUNTIME=ON` 启用

---

**注意**: 如果暂时不安装 ONNX Runtime，CPU 后端仍然可用，可以继续开发和测试。

