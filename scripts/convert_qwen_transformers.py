#!/usr/bin/env python3
"""
使用transformers的ONNX导出功能转换Qwen模型
这是HuggingFace推荐的转换方式
"""

import os
import sys
import argparse

try:
    from transformers import AutoModel, AutoTokenizer
    from transformers.onnx import export, FeaturesManager
    USE_TRANSFORMERS_EXPORT = True
except ImportError:
    print("❌ transformers库版本过低，不支持ONNX导出")
    print("   请升级: pip install transformers --upgrade")
    USE_TRANSFORMERS_EXPORT = False

def convert_with_transformers(model_path, output_path):
    """使用transformers的ONNX导出功能"""
    if not USE_TRANSFORMERS_EXPORT:
        return False
    
    print(f"正在加载模型: {model_path}")
    try:
        model = AutoModel.from_pretrained(model_path, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return False
    
    model.eval()
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"正在导出ONNX模型到: {output_path}")
    try:
        # 获取模型配置
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model)
        onnx_config = model_onnx_config(model.config)
        
        # 导出ONNX
        export(
            tokenizer,
            model,
            onnx_config,
            opset=14,  # 使用opset 14
            output=output_path,
        )
        
        print(f"✅ ONNX模型导出成功: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="使用transformers导出Qwen模型为ONNX")
    parser.add_argument("--model_path", type=str, 
                       default="models/Qwen2.5-0.5B",
                       help="模型路径")
    parser.add_argument("--output", type=str,
                       default="models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx",
                       help="输出ONNX文件路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        return 1
    
    success = convert_with_transformers(args.model_path, args.output)
    
    if success and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\n✅ 转换完成!")
        print(f"   文件: {args.output}")
        print(f"   大小: {size_mb:.2f} MB")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

