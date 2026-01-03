#!/bin/bash
# 自动化设置和测试脚本

set -e  # 遇到错误立即退出

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  InferUnity Qwen2.5-0.5B 自动化测试"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 步骤1: 检查Python环境
echo -e "\n${YELLOW}[1/5] 检查Python环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 未安装${NC}"
    exit 1
fi

python3 --version
if ! python3 -c "import torch, transformers" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  需要安装PyTorch和transformers${NC}"
    echo "运行: pip install torch transformers"
    exit 1
fi
echo -e "${GREEN}✅ Python环境检查通过${NC}"

# 步骤2: 转换模型为ONNX
echo -e "\n${YELLOW}[2/5] 转换模型为ONNX格式...${NC}"
MODEL_PATH="models/Qwen2.5-0.5B"
ONNX_OUTPUT="models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx"

if [ ! -f "$ONNX_OUTPUT" ]; then
    echo "正在转换模型..."
    python3 scripts/convert_qwen_to_onnx.py \
        --model_path "$MODEL_PATH" \
        --output "$ONNX_OUTPUT" \
        --max_length 128
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ 模型转换失败${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ ONNX模型已存在: $ONNX_OUTPUT${NC}"
fi

# 步骤3: 编译项目
echo -e "\n${YELLOW}[3/5] 编译项目...${NC}"
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
    echo -e "${RED}❌ 编译失败${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 编译成功${NC}"

# 步骤4: 检查ONNX模型
cd "$PROJECT_ROOT"
echo -e "\n${YELLOW}[4/5] 检查ONNX模型...${NC}"
if [ -f "$ONNX_OUTPUT" ]; then
    if [ -f "build/bin/inferunity_tool" ]; then
        echo "检查模型信息..."
        ./build/bin/inferunity_tool info "$ONNX_OUTPUT" 2>/dev/null || echo "工具检查跳过"
    fi
    SIZE_MB=$(du -m "$ONNX_OUTPUT" | cut -f1)
    echo -e "${GREEN}✅ ONNX模型: $ONNX_OUTPUT (${SIZE_MB}MB)${NC}"
else
    echo -e "${RED}❌ ONNX模型不存在: $ONNX_OUTPUT${NC}"
    exit 1
fi

# 步骤5: 运行测试
echo -e "\n${YELLOW}[5/5] 运行推理测试...${NC}"
if [ -f "build/bin/test_qwen" ]; then
    echo "运行Qwen2.5-0.5B测试..."
    ./build/bin/test_qwen "$ONNX_OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}=========================================="
        echo "  ✅ 测试完成！"
        echo "==========================================${NC}"
    else
        echo -e "\n${RED}=========================================="
        echo "  ❌ 测试失败"
        echo "==========================================${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ 测试程序不存在: build/bin/test_qwen${NC}"
    echo "请检查编译是否成功"
    exit 1
fi

