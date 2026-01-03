#!/bin/bash
# 使用optimum-cli转换Qwen模型

set -e

MODEL_PATH="models/Qwen2.5-0.5B"
OUTPUT_DIR="models/Qwen2.5-0.5B/onnx"

echo "=== 使用optimum-cli转换Qwen模型 ==="
echo "模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_DIR"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查optimum-cli是否安装
if ! command -v optimum-cli &> /dev/null; then
    echo "⚠️  optimum-cli未安装，尝试安装..."
    pip install optimum[onnxruntime] --upgrade
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 使用optimum-cli转换
echo "开始转换..."
optimum-cli export onnx \
    --model "$MODEL_PATH" \
    --task text-generation \
    --opset 14 \
    "$OUTPUT_DIR" 2>&1 | tee /tmp/optimum_export.log

if [ $? -eq 0 ]; then
    echo "✅ 转换成功！"
    echo "ONNX文件位置: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"/*.onnx 2>/dev/null || echo "未找到ONNX文件"
else
    echo "❌ 转换失败，查看日志: /tmp/optimum_export.log"
    exit 1
fi

