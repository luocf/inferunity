#!/usr/bin/env python3
"""
临时补丁：为PyTorch 2.2.2添加rms_norm支持
用于解决optimum库的依赖问题
"""

import torch
import torch.nn as nn

# 如果torch没有rms_norm，添加一个简单的实现
if not hasattr(torch, 'rms_norm'):
    def rms_norm(input, normalized_shape, weight=None, eps=1e-5):
        """
        简单的RMSNorm实现
        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
        """
        # 计算均方根
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + eps)
        
        if weight is not None:
            input = input * weight
        
        return input
    
    # 添加到torch模块
    torch.rms_norm = rms_norm
    print("✓ 已为torch添加rms_norm函数")

# 验证
if hasattr(torch, 'rms_norm'):
    print("✓ torch.rms_norm可用")
else:
    print("✗ torch.rms_norm仍不可用")

