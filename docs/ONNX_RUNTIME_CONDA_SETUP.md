# 在Conda环境中使用ONNX Runtime C++库

## 概述

如果你在conda环境中工作，可以使用conda安装ONNX Runtime的C++版本，这样可以直接在C++项目中使用。

## 安装步骤

### 1. 在conda环境中安装ONNX Runtime

```bash
# 激活conda环境
conda activate onnx_env

# 安装ONNX Runtime（包含C++库）
conda install -c conda-forge onnxruntime

# 或安装专门的C++库（如果可用）
conda install -c conda-forge libonnxruntime
```

### 2. 查找C++库和头文件位置

```bash
# 激活环境
conda activate onnx_env

# 查找库文件
find $CONDA_PREFIX -name "libonnxruntime*.dylib"

# 查找头文件
find $CONDA_PREFIX -name "onnxruntime_cxx_api.h"
```

**典型位置**:
- 库文件: `$CONDA_PREFIX/lib/libonnxruntime.dylib`
- 头文件: `$CONDA_PREFIX/include/onnxruntime/onnxruntime_cxx_api.h`
- 或: `$CONDA_PREFIX/lib/python3.x/site-packages/onnxruntime/capi/libonnxruntime*.dylib`

### 3. 配置CMake使用conda环境中的ONNX Runtime

```bash
# 激活conda环境
conda activate onnx_env

# 配置CMake
cd build
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_INCLUDE_DIRS=$CONDA_PREFIX/include \
         -Donnxruntime_LIBRARIES=$CONDA_PREFIX/lib/libonnxruntime.dylib

# 或如果库在site-packages中
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_INCLUDE_DIRS=$CONDA_PREFIX/include \
         -Donnxruntime_LIBRARIES=$(find $CONDA_PREFIX -name "libonnxruntime*.dylib" | head -1)
```

### 4. 编译项目

```bash
# 确保在conda环境中
conda activate onnx_env

# 编译
make -j$(sysctl -n hw.ncpu)
```

## 验证安装

### 检查库文件

```bash
conda activate onnx_env

# 检查库文件是否存在
ls -lh $CONDA_PREFIX/lib/libonnxruntime*.dylib

# 检查头文件
ls -lh $CONDA_PREFIX/include/onnxruntime/*.h
```

### 检查Python版本

```bash
conda activate onnx_env
python -c "import onnxruntime; print('版本:', onnxruntime.__version__)"
```

## 常见问题

### 1. pip安装的版本不包含C++库

**问题**: `pip install onnxruntime` 安装的版本可能只包含Python绑定，不包含C++库。

**解决**: 使用conda安装：
```bash
conda install -c conda-forge onnxruntime
```

### 2. 找不到头文件

**问题**: CMake找不到 `onnxruntime_cxx_api.h`

**解决**: 
- 检查头文件位置：`find $CONDA_PREFIX -name "onnxruntime_cxx_api.h"`
- 手动指定路径：`-Donnxruntime_INCLUDE_DIRS=/path/to/include`

### 3. 库文件路径问题

**问题**: 运行时找不到动态库

**解决**:
```bash
# 设置动态库路径
export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH

# 或在CMake中设置RPATH
cmake .. -DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib
```

## 使用示例

### CMake配置脚本

创建一个 `setup_onnxruntime.sh` 脚本：

```bash
#!/bin/bash
# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate onnx_env

# 查找ONNX Runtime库
ORT_LIB=$(find $CONDA_PREFIX -name "libonnxruntime*.dylib" | head -1)
ORT_INCLUDE=$(find $CONDA_PREFIX -name "onnxruntime_cxx_api.h" | head -1 | xargs dirname)

if [ -z "$ORT_LIB" ] || [ -z "$ORT_INCLUDE" ]; then
    echo "错误: 找不到ONNX Runtime C++库"
    echo "请运行: conda install -c conda-forge onnxruntime"
    exit 1
fi

echo "找到ONNX Runtime库: $ORT_LIB"
echo "找到ONNX Runtime头文件: $ORT_INCLUDE"

# 配置CMake
cd build
cmake .. -DENABLE_ONNXRUNTIME=ON \
         -Donnxruntime_INCLUDE_DIRS=$ORT_INCLUDE \
         -Donnxruntime_LIBRARIES=$ORT_LIB

echo "CMake配置完成！"
```

## 注意事项

1. **环境一致性**: 确保编译和运行时使用相同的conda环境
2. **路径设置**: 可能需要设置 `DYLD_LIBRARY_PATH` 或使用RPATH
3. **版本匹配**: 确保Python版本和C++库版本匹配
4. **架构匹配**: 确保库的架构（x86_64/arm64）与项目匹配

## 替代方案

如果conda环境中的ONNX Runtime不包含C++库，可以使用：

1. **Homebrew安装**（系统级）:
   ```bash
   brew install onnxruntime
   ```

2. **预编译版本**（手动安装）:
   - 从 [GitHub Releases](https://github.com/microsoft/onnxruntime/releases) 下载
   - 解压并配置路径

---

**最后更新**: 2026-01-06

