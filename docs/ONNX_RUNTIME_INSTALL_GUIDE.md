# ONNX Runtime 安装指南

## 概述

ONNX Runtime 是 Microsoft 开发的高性能推理引擎，InferUnity 可以将其作为后端使用，获得更好的性能和算子支持。

## ⚠️ 重要提示

**Python版本 vs C++版本**：
- `pip install onnxruntime` 安装的是 **Python 版本**（用于Python项目）
- C++ 项目需要 **C++ 库版本**（包含 `.dylib` 文件和头文件）
- 两者可以共存，互不影响

**如果已安装Python版本**：
- Python版本不包含C++库文件
- 需要单独安装C++版本才能编译C++项目
- 可以使用Homebrew、预编译版本或conda安装C++版本

## 安装方式

### 方式1: 使用 Homebrew（推荐，macOS）

```bash
# 安装 ONNX Runtime
brew install onnxruntime

# 验证安装
brew list onnxruntime
```

**安装位置**:
- 库文件: `/opt/homebrew/lib/libonnxruntime.dylib` (Apple Silicon) 或 `/usr/local/lib/libonnxruntime.dylib` (Intel)
- 头文件: `/opt/homebrew/include/onnxruntime` (Apple Silicon) 或 `/usr/local/include/onnxruntime` (Intel)
- CMake配置: `/opt/homebrew/lib/cmake/onnxruntime` (Apple Silicon) 或 `/usr/local/lib/cmake/onnxruntime` (Intel)

### 方式2: 使用 Conda（如果使用conda环境）

```bash
# 在conda环境中安装C++版本
conda install -c conda-forge onnxruntime

# 验证安装
conda list onnxruntime
```

**注意**: Conda版本可能包含C++库，但路径可能不同，需要手动指定CMake路径。

### 方式3: 下载预编译版本（快速）

1. 访问 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases)
2. 下载 macOS 版本（选择 `onnxruntime-osx-x64-<version>.tgz` 或 `onnxruntime-osx-arm64-<version>.tgz`）
3. 解压到指定目录：

```bash
# 下载（示例）
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.2/onnxruntime-osx-x64-1.22.2.tgz

# 解压
tar -xzf onnxruntime-osx-x64-1.22.2.tgz

# 移动到系统目录（可选）
sudo cp -r onnxruntime-osx-x64-1.22.2/include/* /usr/local/include/
sudo cp -r onnxruntime-osx-x64-1.22.2/lib/* /usr/local/lib/
```

### 方式4: 从源码编译（完整控制）

```bash
# 克隆仓库
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# macOS 编译
./build.sh --config Release --build_shared_lib --parallel

# 编译后的文件位置
# build/MacOS/Release/libonnxruntime.dylib
# build/MacOS/Release/include/
```

## 配置 CMake

### 使用 Homebrew 安装的版本

```bash
cd build
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_DIR=/opt/homebrew/lib/cmake/onnxruntime  # Apple Silicon
# 或
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_DIR=/usr/local/lib/cmake/onnxruntime  # Intel
```

### 使用预编译版本

```bash
cd build
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_INCLUDE_DIRS=/path/to/onnxruntime/include \
         -Donnxruntime_LIBRARIES=/path/to/onnxruntime/lib/libonnxruntime.dylib
```

### 使用源码编译的版本

```bash
cd build
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_DIR=/path/to/onnxruntime/build/MacOS/Release/cmake
```

## 验证安装

### 1. 检查库文件

```bash
# 检查库文件是否存在
ls -la /opt/homebrew/lib/libonnxruntime*  # Apple Silicon
ls -la /usr/local/lib/libonnxruntime*     # Intel

# 检查头文件
ls -la /opt/homebrew/include/onnxruntime/  # Apple Silicon
ls -la /usr/local/include/onnxruntime/    # Intel
```

### 2. 编译项目

```bash
cd build
cmake .. -DENABLE_ONNXRUNTIME=ON
make -j$(sysctl -n hw.ncpu)

# 检查CMake输出，应该看到：
# -- Found ONNX Runtime: ...
# -- ONNX Runtime found, enabling ONNX Runtime backend
```

### 3. 运行测试

```bash
# 运行CPU后端测试（应该仍然可用）
./bin/test_cpu_backend

# 如果ONNX Runtime后端已集成，可以测试
# ./bin/test_onnxruntime_backend  # （如果已实现）
```

## 常见问题

### 1. CMake找不到ONNX Runtime

**错误**: `ONNX Runtime not found`

**解决方案**:
```bash
# 方法1: 指定CMake路径
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_DIR=/opt/homebrew/lib/cmake/onnxruntime

# 方法2: 设置CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=/opt/homebrew:$CMAKE_PREFIX_PATH
cmake .. -DENABLE_ONNXRUNTIME=ON

# 方法3: 手动指定路径
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_INCLUDE_DIRS=/opt/homebrew/include \
         -Donnxruntime_LIBRARIES=/opt/homebrew/lib/libonnxruntime.dylib
```

### 2. 链接错误

**错误**: `undefined reference to Ort::...`

**解决方案**:
- 确保链接了正确的库文件
- 检查库文件架构是否匹配（x86_64 vs arm64）
- 确保使用相同编译器版本

### 3. 运行时错误

**错误**: `Library not loaded: @rpath/libonnxruntime.dylib`

**解决方案**:
```bash
# 方法1: 设置动态库路径
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# 方法2: 使用install_name_tool修改库路径
install_name_tool -change @rpath/libonnxruntime.dylib \
    /opt/homebrew/lib/libonnxruntime.dylib \
    build/bin/your_executable
```

### 4. 架构不匹配

**错误**: `Bad CPU type in executable`

**解决方案**:
- 确保ONNX Runtime库的架构与项目匹配
- Apple Silicon (arm64) 使用 `onnxruntime-osx-arm64`
- Intel (x86_64) 使用 `onnxruntime-osx-x64`

## 使用ONNX Runtime后端

安装完成后，可以在代码中使用ONNX Runtime后端：

```cpp
#include "inferunity/engine.h"

using namespace inferunity;

int main() {
    // 创建会话选项
    SessionOptions options;
    options.execution_providers = {"ONNXRuntime"};  // 使用ONNX Runtime后端
    
    // 创建推理会话
    auto session = InferenceSession::Create(options);
    
    // 加载模型
    session->LoadModel("model.onnx");
    
    // 运行推理
    // ...
    
    return 0;
}
```

## 当前状态

- ✅ ONNX Runtime后端代码已实现 (`src/backends/onnxruntime_backend.cpp`)
- ⏳ 需要安装ONNX Runtime库才能编译
- ⏳ 安装后可通过 `-DENABLE_ONNXRUNTIME=ON` 启用

## 注意事项

1. **CPU后端仍然可用**: 即使不安装ONNX Runtime，CPU后端仍然可以正常工作
2. **可选功能**: ONNX Runtime后端是可选的，不影响核心功能
3. **生产环境推荐**: 生产环境建议使用ONNX Runtime后端，获得更好的性能和稳定性

---

**最后更新**: 2026-01-06

