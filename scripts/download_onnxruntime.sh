#!/bin/bash
# 下载并安装ONNX Runtime预编译版本

set -e

# 检测架构
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    ORT_ARCH="arm64"
    ORT_FILE="onnxruntime-osx-arm64-1.22.2.tgz"
else
    ORT_ARCH="x64"
    ORT_FILE="onnxruntime-osx-x64-1.22.2.tgz"
fi

echo "=== 下载ONNX Runtime预编译版本 ==="
echo "架构: $ORT_ARCH"
echo "文件: $ORT_FILE"
echo ""

# 创建临时目录
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# 下载URL
DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.22.2/$ORT_FILE"

echo "下载地址: $DOWNLOAD_URL"
echo ""

# 下载
echo "正在下载..."
if command -v curl &> /dev/null; then
    curl -L -o "$ORT_FILE" "$DOWNLOAD_URL"
elif command -v wget &> /dev/null; then
    wget "$DOWNLOAD_URL" -O "$ORT_FILE"
else
    echo "错误: 需要 curl 或 wget"
    exit 1
fi

# 解压
echo "正在解压..."
tar -xzf "$ORT_FILE"

# 查找解压后的目录
ORT_DIR=$(find . -maxdepth 1 -type d -name "onnxruntime-*" | head -1)

if [ -z "$ORT_DIR" ]; then
    echo "错误: 找不到解压后的目录"
    exit 1
fi

echo ""
echo "=== 安装位置 ==="
echo "库文件: $ORT_DIR/lib/libonnxruntime.dylib"
echo "头文件: $ORT_DIR/include/onnxruntime/"
echo ""

# 询问是否安装到系统目录
read -p "是否安装到 /usr/local? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在安装到 /usr/local..."
    sudo cp -r "$ORT_DIR/include/onnxruntime" /usr/local/include/
    sudo cp "$ORT_DIR/lib/libonnxruntime.dylib" /usr/local/lib/
    echo "✅ 安装完成！"
    echo ""
    echo "配置CMake:"
    echo "  cmake .. -DENABLE_ONNXRUNTIME=ON \\"
    echo "           -Donnxruntime_INCLUDE_DIRS=/usr/local/include \\"
    echo "           -Donnxruntime_LIBRARIES=/usr/local/lib/libonnxruntime.dylib"
else
    echo "文件已下载到: $TMP_DIR/$ORT_DIR"
    echo ""
    echo "配置CMake时使用:"
    echo "  cmake .. -DENABLE_ONNXRUNTIME=ON \\"
    echo "           -Donnxruntime_INCLUDE_DIRS=$TMP_DIR/$ORT_DIR/include \\"
    echo "           -Donnxruntime_LIBRARIES=$TMP_DIR/$ORT_DIR/lib/libonnxruntime.dylib"
fi

echo ""
echo "清理临时文件..."
rm -rf "$TMP_DIR"

echo "✅ 完成！"

