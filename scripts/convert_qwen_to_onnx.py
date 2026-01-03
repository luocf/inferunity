#!/usr/bin/env python3
"""
将Qwen2.5-0.5B模型转换为ONNX格式
使用optimum库（HuggingFace推荐的方式）
"""

import os
import sys
import argparse

try:
    from optimum.onnxruntime import ORTModelForCausalLM
    from transformers import AutoTokenizer
    USE_OPTIMUM = True
except ImportError:
    print("⚠️  optimum库未安装，尝试使用torch.onnx.export")
    print("   建议安装: pip install optimum[onnxruntime]")
    USE_OPTIMUM = False
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

def convert_to_onnx(model_path, output_path, max_length=128):
    """将Qwen模型转换为ONNX格式"""
    print(f"正在加载模型: {model_path}")
    
    if USE_OPTIMUM:
        # 使用optimum库转换（推荐方式）
        try:
            print("使用optimum库转换模型...")
            # 创建输出目录
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 使用optimum转换
            model = ORTModelForCausalLM.from_pretrained(
                model_path,
                export=True,
                use_io_binding=False
            )
            
            # 保存ONNX模型
            model.save_pretrained(output_dir)
            
            # 查找生成的ONNX文件
            onnx_files = [f for f in os.listdir(output_dir) if f.endswith('.onnx')]
            if onnx_files:
                # 如果生成了多个文件，使用第一个（通常是decoder_model.onnx）
                generated_file = os.path.join(output_dir, onnx_files[0])
                if generated_file != output_path:
                    import shutil
                    shutil.move(generated_file, output_path)
                    print(f"✅ ONNX模型已保存到: {output_path}")
                else:
                    print(f"✅ ONNX模型已保存到: {output_path}")
                return True
            else:
                print("❌ 未找到生成的ONNX文件")
                return False
                
        except Exception as e:
            print(f"❌ 使用optimum转换失败: {e}")
            print("   尝试使用torch.onnx.export...")
            import traceback
            traceback.print_exc()
            # 继续尝试torch.onnx.export
    
    # 使用torch.onnx.export（备用方式）
    try:
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return False
    
    model.eval()
    print("✅ 模型加载成功")
    
    # 创建一个包装类来避免参数冲突
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            outputs = self.model(input_ids=input_ids)
            return outputs.last_hidden_state
    
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()
    
    # 准备输入
    print(f"准备输入 (max_length={max_length})...")
    dummy_text = "Hello, how are you?"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    
    print(f"输入形状: {input_ids.shape}")
    
    # 导出ONNX（使用更高的opset版本）
    print(f"\n正在导出ONNX模型到: {output_path}")
    try:
        torch.onnx.export(
            wrapped_model,
            input_ids,
            output_path,
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=17,  # 使用更高的opset版本
            do_constant_folding=True,
            verbose=False
        )
        print(f"✅ ONNX模型导出成功: {output_path}")
        return True
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="将Qwen2.5-0.5B模型转换为ONNX格式")
    parser.add_argument("--model_path", type=str, 
                       default="models/Qwen2.5-0.5B",
                       help="模型路径")
    parser.add_argument("--output", type=str,
                       default="models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx",
                       help="输出ONNX文件路径")
    parser.add_argument("--max_length", type=int, default=128,
                       help="最大序列长度")
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        return 1
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 转换模型
    success = convert_to_onnx(args.model_path, args.output, args.max_length)
    
    if success:
        # 显示文件信息
        if os.path.exists(args.output):
            size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"\n✅ 转换完成!")
            print(f"   文件: {args.output}")
            print(f"   大小: {size_mb:.2f} MB")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())

