#!/usr/bin/env python3
"""
使用补丁转换Qwen模型为ONNX
先应用rms_norm补丁，然后使用optimum转换
"""

import sys
import os

# 先应用补丁
sys.path.insert(0, os.path.dirname(__file__))
import patch_torch_rmsnorm

# 然后导入optimum
try:
    from optimum.exporters.onnx.convert import onnx_export_from_model
    from optimum.exporters.onnx.config import TextDecoderOnnxConfig
    from transformers import AutoModel, AutoTokenizer
    import torch
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def convert_qwen_to_onnx(model_path, output_dir, opset=14):
    """转换Qwen模型为ONNX"""
    print(f"正在加载模型: {model_path}")
    
    try:
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备输入
    dummy_input = tokenizer("Hello", return_tensors="pt")
    input_ids = dummy_input["input_ids"]
    
    print(f"正在导出ONNX模型到: {output_dir}")
    try:
        # 使用optimum的导出功能
        onnx_config = TextDecoderOnnxConfig(
            model.config,
            task="text-generation"
        )
        
        onnx_export_from_model(
            model=model,
            output=output_dir,
            opset=opset,
            config=onnx_config,
            input_shapes={"input_ids": input_ids.shape},
        )
        
        print(f"✅ ONNX模型导出成功: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/Qwen2.5-0.5B")
    parser.add_argument("--output", default="models/Qwen2.5-0.5B/onnx")
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()
    
    success = convert_qwen_to_onnx(args.model_path, args.output, args.opset)
    sys.exit(0 if success else 1)

