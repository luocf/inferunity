#!/bin/bash
# 完整的自动化设置脚本（包含虚拟环境）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  InferUnity Qwen2.5-0.5B 完整自动化设置"
echo "=========================================="
cd "$PROJECT_ROOT"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

VENV_DIR="venv"

# 步骤1: 设置Python虚拟环境
echo -e "\n${YELLOW}[1/6] 设置Python虚拟环境...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo "创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

echo "激活虚拟环境..."
source "$VENV_DIR/bin/activate"

echo "安装Python依赖..."
pip install --upgrade pip --quiet
pip install "numpy<2" torch transformers optimum[onnxruntime] --quiet

python3 -c "import torch, transformers; print('✅ PyTorch:', torch.__version__); print('✅ Transformers:', transformers.__version__)" || {
    echo "❌ Python依赖安装失败"
    exit 1
}

# 步骤2: 转换模型为ONNX
echo -e "\n${YELLOW}[2/6] 转换模型为ONNX格式...${NC}"
MODEL_PATH="models/Qwen2.5-0.5B"
ONNX_OUTPUT="models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx"

if [ ! -f "$ONNX_OUTPUT" ]; then
    echo "正在转换模型（这可能需要几分钟）..."
    python3 scripts/convert_qwen_to_onnx.py \
        --model_path "$MODEL_PATH" \
        --output "$ONNX_OUTPUT" \
        --max_length 128
    
    if [ $? -ne 0 ]; then
        echo "❌ 模型转换失败"
        exit 1
    fi
else
    echo -e "${GREEN}✅ ONNX模型已存在${NC}"
fi

# 步骤3: 编译项目
echo -e "\n${YELLOW}[3/6] 编译项目...${NC}"
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

if [ ! -f "CMakeCache.txt" ]; then
    echo "运行CMake配置..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON
fi

echo "编译项目（这可能需要几分钟）..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

echo -e "${GREEN}✅ 编译成功${NC}"

# 步骤4: 检查ONNX模型
cd "$PROJECT_ROOT"
echo -e "\n${YELLOW}[4/6] 检查ONNX模型...${NC}"
if [ -f "$ONNX_OUTPUT" ]; then
    SIZE_MB=$(du -m "$ONNX_OUTPUT" | cut -f1)
    echo -e "${GREEN}✅ ONNX模型: $ONNX_OUTPUT (${SIZE_MB}MB)${NC}"
else
    echo "❌ ONNX模型不存在: $ONNX_OUTPUT"
    exit 1
fi

# 步骤5: 检查编译产物
echo -e "\n${YELLOW}[5/6] 检查编译产物...${NC}"
if [ -f "build/bin/test_qwen" ]; then
    echo -e "${GREEN}✅ 测试程序已编译${NC}"
else
    echo "❌ 测试程序不存在"
    exit 1
fi

# 步骤6: 运行测试
echo -e "\n${YELLOW}[6/6] 运行推理测试...${NC}"
echo "运行Qwen2.5-0.5B测试..."
./build/bin/test_qwen "$ONNX_OUTPUT"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "  ✅ 所有步骤完成！"
    echo "==========================================${NC}"
    echo ""
    echo "后续使用："
    echo "  1. 激活虚拟环境: source $VENV_DIR/bin/activate"
    echo "  2. 运行测试: ./build/bin/test_qwen $ONNX_OUTPUT"
    echo ""
else
    echo ""
    echo "❌ 测试失败，请检查错误信息"
    exit 1
fi

