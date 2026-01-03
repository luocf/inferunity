#!/usr/bin/env python3
"""
使用简化的方法转换Qwen模型为ONNX
避免使用不支持的操作符
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

def convert_qwen_simple(model_path, output_path):
    """使用简化的方法转换Qwen模型"""
    print(f"正在加载模型: {model_path}")
    
    try:
        model = AutoModel.from_pretrained(model_path, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建一个更简单的包装器，只导出embedding层
    # 这样可以避免复杂操作符
    class EmbeddingOnlyWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.embedding = model.get_input_embeddings()
        
        def forward(self, input_ids):
            return self.embedding(input_ids)
    
    # 尝试导出embedding层
    print("尝试导出embedding层...")
    embedding_wrapper = EmbeddingOnlyWrapper(model)
    embedding_wrapper.eval()
    
    # 准备输入
    dummy_input = tokenizer("Hello", return_tensors="pt", max_length=8, padding=True, truncation=True)
    input_ids = dummy_input["input_ids"]
    print(f"输入形状: {input_ids.shape}")
    
    try:
        # 导出embedding层
        embedding_path = output_path.replace('.onnx', '_embedding.onnx')
        torch.onnx.export(
            embedding_wrapper,
            input_ids,
            embedding_path,
            input_names=["input_ids"],
            output_names=["embeddings"],
            opset_version=11,
            do_constant_folding=False,
            verbose=False
        )
        print(f"✅ Embedding层导出成功: {embedding_path}")
    except Exception as e:
        print(f"⚠️  Embedding层导出失败: {e}")
    
    # 尝试导出完整模型（使用更短的序列）
    print("\n尝试导出完整模型（简化版）...")
    
    class SimpleWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids):
            # 使用更简单的forward，避免复杂操作
            outputs = self.model(input_ids=input_ids, use_cache=False)
            return outputs.last_hidden_state
    
    simple_wrapper = SimpleWrapper(model)
    simple_wrapper.eval()
    
    # 使用更短的输入序列
    short_input = torch.randint(0, 100, (1, 4), dtype=torch.long)
    print(f"使用短输入序列: {short_input.shape}")
    
    try:
        torch.onnx.export(
            simple_wrapper,
            short_input,
            output_path,
            input_names=["input_ids"],
            output_names=["hidden_states"],
            opset_version=11,
            do_constant_folding=False,
            verbose=True
        )
        print(f"✅ 完整模型导出成功: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 完整模型导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/Qwen2.5-0.5B")
    parser.add_argument("--output", default="models/Qwen2.5-0.5B/qwen2.5-0.5b.onnx")
    args = parser.parse_args()
    
    success = convert_qwen_simple(args.model_path, args.output)
    sys.exit(0 if success else 1)

