#!/usr/bin/env python3
"""
创建一个简单的ONNX测试模型
用于验证推理引擎的基本功能
"""

import onnx
from onnx import helper, TensorProto
import numpy as np

def create_simple_add_model():
    """创建一个简单的Add模型：output = input1 + input2"""
    print("创建简单的Add模型...")
    
    # 定义输入
    input1 = helper.make_tensor_value_info('input1', TensorProto.FLOAT, [2, 3])
    input2 = helper.make_tensor_value_info('input2', TensorProto.FLOAT, [2, 3])
    
    # 定义输出
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])
    
    # 创建Add节点
    add_node = helper.make_node(
        'Add',
        ['input1', 'input2'],
        ['output'],
        name='add_node'
    )
    
    # 创建图
    graph = helper.make_graph(
        [add_node],
        'simple_add_model',
        [input1, input2],
        [output]
    )
    
    # 创建模型
    model = helper.make_model(graph, producer_name='inferunity_test')
    
    # 验证模型
    onnx.checker.check_model(model)
    
    return model

def create_simple_conv_model():
    """创建一个简单的Conv+Relu模型"""
    print("创建简单的Conv+Relu模型...")
    
    # 输入: [1, 1, 5, 5]
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 5, 5])
    
    # 权重: [1, 1, 3, 3]
    weight_data = np.ones((1, 1, 3, 3), dtype=np.float32) * 0.1
    weight = helper.make_tensor('weight', TensorProto.FLOAT, [1, 1, 3, 3], weight_data.flatten())
    
    # Conv节点
    conv_output = helper.make_tensor_value_info('conv_output', TensorProto.FLOAT, [1, 1, 3, 3])
    conv_node = helper.make_node(
        'Conv',
        ['input', 'weight'],
        ['conv_output'],
        name='conv_node',
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1]
    )
    
    # Relu节点
    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 3, 3])
    relu_node = helper.make_node(
        'Relu',
        ['conv_output'],
        ['output'],
        name='relu_node'
    )
    
    # 创建图
    graph = helper.make_graph(
        [conv_node, relu_node],
        'simple_conv_model',
        [input_tensor],
        [output],
        [weight]  # 初始值
    )
    
    # 创建模型
    model = helper.make_model(graph, producer_name='inferunity_test')
    
    # 验证模型
    onnx.checker.check_model(model)
    
    return model

if __name__ == '__main__':
    import sys
    import os
    
    # 创建输出目录
    output_dir = 'models/test'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建Add模型
    add_model = create_simple_add_model()
    add_path = os.path.join(output_dir, 'simple_add.onnx')
    onnx.save(add_model, add_path)
    print(f"✓ Add模型已保存: {add_path}")
    print(f"  输入: input1[2,3], input2[2,3]")
    print(f"  输出: output[2,3] = input1 + input2")
    
    # 创建Conv模型
    try:
        conv_model = create_simple_conv_model()
        conv_path = os.path.join(output_dir, 'simple_conv.onnx')
        onnx.save(conv_model, conv_path)
        print(f"\n✓ Conv模型已保存: {conv_path}")
        print(f"  输入: input[1,1,5,5]")
        print(f"  输出: output[1,1,3,3] = Relu(Conv(input))")
    except Exception as e:
        print(f"\n⚠ Conv模型创建失败: {e}")
        print("  仅创建Add模型")
    
    print(f"\n测试模型已创建在: {output_dir}/")
    print("可以使用以下命令测试:")
    print(f"  ./build/bin/inference_example {add_path}")

