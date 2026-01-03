# InferUnity 测试指南

## 概述

InferUnity使用Google Test (GTest)作为测试框架，提供完整的单元测试和集成测试覆盖。

## 测试结构

### 单元测试

#### 核心功能测试 (`test_core`)
- **test_tensor.cpp**: Tensor创建、Reshape、Slice、序列化等
- **test_graph.cpp**: Graph创建、验证、拓扑排序、深拷贝等
- **test_memory.cpp**: 内存分配、统计、生命周期分析等

#### 算子测试 (`test_operators`)
- 所有基础算子的功能测试
- 形状推断测试
- 执行测试

### 集成测试

#### 运行时测试 (`test_runtime`)
- 简单推理流程
- 多节点推理
- 形状推断流程
- 图优化流程

#### 端到端测试 (`test_integration`)
- 完整推理流程
- 图优化验证
- 多批次推理
- 内存管理验证

### 专项测试

- **test_fused_operators.cpp**: 融合算子测试
- **test_operator_fusion.cpp**: 算子融合Pass测试
- **test_onnx_parser.cpp**: ONNX解析器测试

## 运行测试

### 编译测试

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
make -j$(nproc)
```

### 运行所有测试

```bash
cd build
ctest --output-on-failure
```

### 运行特定测试套件

```bash
cd build
./tests/test_core          # 核心功能测试
./tests/test_operators    # 算子测试
./tests/test_runtime      # 运行时测试
./tests/test_integration  # 集成测试
```

### 使用GTest过滤器

```bash
# 只运行Tensor相关测试
./tests/test_core --gtest_filter=*Tensor*

# 只运行Graph相关测试
./tests/test_core --gtest_filter=*Graph*

# 列出所有测试
./tests/test_core --gtest_list_tests
```

## 测试覆盖率

当前测试覆盖的核心功能：

- ✅ Tensor: 创建、Reshape、Slice、序列化、CopyTo、Fill
- ✅ Graph: 创建、验证、拓扑排序、深拷贝、序列化
- ✅ Memory: 分配、统计、对齐分配、生命周期分析
- ✅ Operators: Reshape、Concat、Split、Transpose、Add、Relu、MatMul
- ✅ Fused Operators: FusedConvBNReLU、FusedMatMulAdd
- ✅ Runtime: 推理流程、形状推断、图优化
- ✅ Integration: 端到端流程、多批次推理

## 添加新测试

### 1. 创建测试文件

在`tests/`目录下创建新的测试文件，例如`test_new_feature.cpp`：

```cpp
#include <gtest/gtest.h>
#include "inferunity/your_header.h"

using namespace inferunity;

class NewFeatureTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试前准备
    }
    
    void TearDown() override {
        // 测试后清理
    }
};

TEST_F(NewFeatureTest, BasicFunctionality) {
    // 测试代码
    EXPECT_TRUE(true);
}
```

### 2. 更新CMakeLists.txt

在`tests/CMakeLists.txt`中添加：

```cmake
add_executable(test_new_feature
    test_new_feature.cpp
)

target_link_libraries(test_new_feature PRIVATE 
    inferunity_core 
    inferunity_operators
    inferunity_optimizers
    inferunity_runtime
    inferunity_backends
    GTest::GTest GTest::Main)
    
add_test(NAME NewFeatureTests COMMAND test_new_feature)
```

## 测试最佳实践

1. **测试独立性**: 每个测试应该独立，不依赖其他测试的执行顺序
2. **可重复性**: 测试应该可以重复运行，结果一致
3. **快速执行**: 单元测试应该快速执行（< 1秒）
4. **清晰命名**: 测试名称应该清楚描述测试内容
5. **完整覆盖**: 测试应该覆盖正常路径和错误路径
6. **使用断言**: 
   - `EXPECT_*` - 非致命断言，测试继续执行
   - `ASSERT_*` - 致命断言，测试立即停止

## 持续集成

建议在CI/CD流程中：

1. 编译所有测试
2. 运行所有测试
3. 检查测试覆盖率
4. 在PR中要求所有测试通过

## 故障排除

### 测试编译失败

- 检查是否安装了GTest: `find_package(GTest)`
- 检查CMakeLists.txt中的链接库是否正确
- 检查头文件包含路径

### 测试运行失败

- 使用`--gtest_filter`运行特定测试
- 使用`--gtest_output=xml`生成XML报告
- 检查测试输出中的错误信息

### 链接错误

- 确保所有依赖库都已链接
- 检查库的编译顺序
- 确保使用了正确的库名称

## 性能测试

性能测试应该单独运行，不在常规测试套件中：

```bash
# 运行性能基准测试
./tools/inferunity_benchmark --model model.onnx --iterations 100

# 运行性能分析
./tools/inferunity_profiler --model model.onnx
```

