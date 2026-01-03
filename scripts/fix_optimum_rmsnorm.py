#!/usr/bin/env python3
"""
修复optimum库的rms_norm依赖问题
在导入optimum之前，为torch添加rms_norm函数
"""

import sys

# 在导入torch之前，我们需要先导入torch
# 但问题是optimum在导入时就会检查rms_norm
# 所以我们需要修改optimum的model_patcher.py文件

import os
optimum_path = os.path.join(os.path.dirname(sys.executable), '..', 'lib', 'python3.12', 'site-packages', 'optimum', 'exporters', 'onnx', 'model_patcher.py')
optimum_path = os.path.abspath(optimum_path)

if os.path.exists(optimum_path):
    print(f"找到optimum文件: {optimum_path}")
    
    # 读取文件
    with open(optimum_path, 'r') as f:
        content = f.read()
    
    # 检查是否需要修改
    if 'torch.rms_norm' in content and 'PatchingSpec(torch, "rms_norm"' in content:
        # 找到问题行并修改
        lines = content.split('\n')
        modified = False
        
        for i, line in enumerate(lines):
            if 'PatchingSpec(torch, "rms_norm", onnx_compatible_rms_norm, torch.rms_norm)' in line:
                # 修改为使用hasattr检查
                new_line = '        PatchingSpec(torch, "rms_norm", onnx_compatible_rms_norm, getattr(torch, "rms_norm", None)),'
                lines[i] = new_line
                modified = True
                print(f"修改第{i+1}行: {line.strip()}")
                print(f"改为: {new_line.strip()}")
                break
        
        if modified:
            # 写回文件
            with open(optimum_path, 'w') as f:
                f.write('\n'.join(lines))
            print("✓ 已修复optimum的rms_norm依赖问题")
        else:
            print("⚠️  未找到需要修改的行")
    else:
        print("文件内容可能已更改，需要手动检查")
else:
    print(f"未找到optimum文件: {optimum_path}")

