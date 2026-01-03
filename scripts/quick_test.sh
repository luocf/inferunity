#!/bin/bash
# 快速测试脚本（跳过模型转换，直接编译和测试）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  InferUnity 快速编译和测试"
echo "=========================================="
cd "$PROJECT_ROOT"

# 步骤1: 编译项目
echo -e "\n[1/2] 编译项目..."
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

if [ ! -f "CMakeCache.txt" ]; then
    echo "运行CMake配置..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
fi

echo "编译项目..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

echo "✅ 编译成功"

# 步骤2: 检查是否有ONNX模型
cd "$PROJECT_ROOT"
ONNX_MODEL="models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx"

echo -e "\n[2/2] 检查模型..."
if [ -f "$ONNX_MODEL" ]; then
    echo "✅ 找到ONNX模型: $ONNX_MODEL"
    echo -e "\n运行测试..."
    ./build/bin/test_qwen "$ONNX_MODEL"
else
    echo "⚠️  ONNX模型不存在: $ONNX_MODEL"
    echo ""
    echo "请先转换模型："
    echo "  python3 scripts/convert_qwen_to_onnx.py"
    echo ""
    echo "或者手动转换："
    echo "  python3 -c \"from transformers import AutoModel; import torch; model = AutoModel.from_pretrained('models/Qwen2.5-0.5B'); torch.onnx.export(model, (torch.randint(0, 1000, (1, 128)),), '$ONNX_MODEL', input_names=['input_ids'], opset_version=14)\""
fi

