#!/bin/bash
# 创建Python虚拟环境并安装依赖

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "  设置Python虚拟环境"
echo "=========================================="
cd "$PROJECT_ROOT"

VENV_DIR="venv"

# 检查是否已存在虚拟环境
if [ -d "$VENV_DIR" ]; then
    echo "✅ 虚拟环境已存在: $VENV_DIR"
    echo "激活虚拟环境: source $VENV_DIR/bin/activate"
else
    echo "创建Python虚拟环境..."
    python3 -m venv "$VENV_DIR"
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 升级pip
echo "升级pip..."
pip install --upgrade pip --quiet

# 安装依赖（使用兼容的NumPy版本）
echo "安装依赖..."
pip install "numpy<2" torch transformers optimum[onnxruntime] --quiet

# 验证安装
echo "验证安装..."
python3 -c "import torch, transformers, numpy; print('✅ PyTorch:', torch.__version__); print('✅ Transformers:', transformers.__version__); print('✅ NumPy:', numpy.__version__)"

echo ""
echo "=========================================="
echo "✅ 虚拟环境设置完成！"
echo "=========================================="
echo ""
echo "使用虚拟环境："
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "退出虚拟环境："
echo "  deactivate"
echo ""

