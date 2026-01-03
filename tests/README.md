# InferUnity 测试文档

## 测试框架

InferUnity使用Google Test (GTest)作为测试框架。所有测试都在`tests/`目录下。

## 测试结构

### 单元测试

1. **test_tensor.cpp** - Tensor核心功能测试
   - Tensor创建和基本操作
   - Reshape和Slice操作
   - 序列化/反序列化
   - CopyTo和Fill操作

2. **test_graph.cpp** - Graph核心功能测试
   - Graph创建和节点管理
   - 图验证
   - 拓扑排序
   - 图深拷贝
   - 图序列化

3. **test_memory.cpp** - 内存管理测试
   - 内存分配器
   - 内存统计
   - 对齐分配
   - 生命周期分析
   - 内存复用

4. **test_operators.cpp** - 算子单元测试
   - Reshape算子
   - Concat算子
   - Split算子
   - Transpose算子
   - Add算子
   - Relu算子
   - MatMul算子

### 集成测试

5. **test_runtime.cpp** - 运行时系统集成测试
   - 简单推理流程
   - 多节点推理
   - 形状推断
   - 图优化

6. **test_integration.cpp** - 端到端集成测试
   - 完整推理流程
   - 图优化流程
   - 形状推断流程
   - 多批次推理
   - 内存管理

### 专项测试

7. **test_fused_operators.cpp** - 融合算子测试
   - FusedConvBNReLU
   - FusedMatMulAdd
   - 算子注册

8. **test_operator_fusion.cpp** - 算子融合Pass测试
   - Conv+BN+ReLU融合
   - MatMul+Add融合

9. **test_onnx_parser.cpp** - ONNX解析器测试
   - 形状推断
   - 图验证

## 运行测试

### 编译测试

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON
make -j$(nproc)
```

### 运行所有测试

```bash
cd build
ctest
```

### 运行特定测试

```bash
cd build
./tests/test_core          # 核心功能测试
./tests/test_operators    # 算子测试
./tests/test_runtime      # 运行时测试
./tests/test_integration  # 集成测试
```

### 使用GTest运行

```bash
cd build
./tests/test_core --gtest_filter=*Tensor*  # 只运行Tensor相关测试
./tests/test_core --gtest_list_tests        # 列出所有测试
```

## 测试覆盖率

当前测试覆盖：

- ✅ Tensor核心功能（创建、Reshape、Slice、序列化）
- ✅ Graph核心功能（创建、验证、拓扑排序、深拷贝）
- ✅ 内存管理（分配、统计、生命周期分析）
- ✅ 基础算子（Reshape、Concat、Split、Transpose、Add、Relu、MatMul）
- ✅ 融合算子（FusedConvBNReLU、FusedMatMulAdd）
- ✅ 算子融合Pass
- ✅ 运行时系统（推理流程、形状推断）
- ✅ 集成测试（端到端流程）

## 添加新测试

1. 在`tests/`目录下创建新的测试文件
2. 在`tests/CMakeLists.txt`中添加测试可执行文件
3. 使用GTest的宏编写测试用例：
   - `TEST_F(TestFixture, TestName)` - 使用测试夹具
   - `TEST(TestSuite, TestName)` - 不使用测试夹具
   - `EXPECT_*` - 非致命断言
   - `ASSERT_*` - 致命断言

## 测试最佳实践

1. **独立性**：每个测试应该独立，不依赖其他测试的执行顺序
2. **可重复性**：测试应该可以重复运行，结果一致
3. **快速执行**：单元测试应该快速执行
4. **清晰命名**：测试名称应该清楚描述测试内容
5. **完整覆盖**：测试应该覆盖正常路径和错误路径

## 持续集成

建议在CI/CD流程中：

1. 编译所有测试
2. 运行所有测试
3. 检查测试覆盖率
4. 在PR中要求所有测试通过

