#!/usr/bin/env python3
"""
使用optimum库转换Qwen2.5模型为ONNX
optimum库支持更多模型类型，包括qwen2
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from optimum.exporters.onnx import main_export
    OPTIMUM_AVAILABLE = True
    print("✓ 使用optimum.exporters.onnx.main_export")
except ImportError as e:
    print(f"❌ optimum库导入失败: {e}")
    print("   请安装: pip install 'optimum[onnxruntime]' --upgrade")
    OPTIMUM_AVAILABLE = False
    sys.exit(1)

def convert_with_optimum(model_path, output_path):
    """使用optimum库转换模型"""
    if not OPTIMUM_AVAILABLE:
        return False
    
    print(f"正在转换模型: {model_path}")
    print(f"输出路径: {output_path}")
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path) or "."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # 使用main_export转换
        print("开始转换...")
        main_export(
            model_name_or_path=model_path,
            output=output_dir,
            task="text-generation",
            opset=14,
        )
        
        # 查找生成的ONNX文件
        onnx_files = list(Path(output_dir).glob("*.onnx"))
        if onnx_files:
            # 如果生成了多个文件，找到主要的模型文件
            # 通常decoder_model.onnx是主要的
            main_file = None
            for f in onnx_files:
                if "decoder_model" in f.name or "model" in f.name:
                    main_file = f
                    break
            
            if not main_file:
                main_file = onnx_files[0]
            
            # 如果指定了具体文件名，移动或复制
            if str(main_file) != output_path:
                import shutil
                shutil.copy2(str(main_file), output_path)
                print(f"✅ ONNX模型已保存到: {output_path}")
            else:
                print(f"✅ ONNX模型已保存到: {output_path}")
            
            # 显示所有生成的文件
            print(f"\n生成的文件:")
            for f in onnx_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.2f} MB)")
            
            return True
        else:
            print("❌ 未找到生成的ONNX文件")
            print(f"   检查目录: {output_dir}")
            return False
                
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="使用optimum转换Qwen模型为ONNX")
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
    
    success = convert_with_optimum(args.model_path, args.output)
    
    if success and os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"\n✅ 转换完成!")
        print(f"   主文件: {args.output}")
        print(f"   大小: {size_mb:.2f} MB")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
