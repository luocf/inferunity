// 性能基准测试
#include <gtest/gtest.h>
#include "inferunity/tensor.h"
#include "inferunity/operator.h"
#include "inferunity/types.h"
#include <chrono>
#include <vector>
#include <iostream>
#include <functional>

using namespace inferunity;

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = std::make_unique<ExecutionContext>();
    }
    
    std::unique_ptr<ExecutionContext> ctx_;
    
    double MeasureTime(std::function<void()> func, int iterations = 100) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / static_cast<double>(iterations);
    }
};

// Add算子性能测试
TEST_F(PerformanceTest, AddOperatorPerformance) {
    auto& registry = OperatorRegistry::Instance();
    auto add_op = registry.Create("Add");
    ASSERT_NE(add_op, nullptr);
    
    // 测试不同大小的张量
    std::vector<int64_t> sizes = {100, 1000, 10000, 100000};
    
    for (int64_t size : sizes) {
        Shape shape({size});
        auto input1 = CreateTensor(shape, DataType::FLOAT32);
        auto input2 = CreateTensor(shape, DataType::FLOAT32);
        auto output = CreateTensor(shape, DataType::FLOAT32);
        
        // 填充数据
        float* data1 = static_cast<float*>(input1->GetData());
        float* data2 = static_cast<float*>(input2->GetData());
        for (int64_t i = 0; i < size; ++i) {
            data1[i] = 1.0f;
            data2[i] = 2.0f;
        }
        
        std::vector<Tensor*> inputs = {input1.get(), input2.get()};
        std::vector<Tensor*> outputs = {output.get()};
        
        double avg_time = MeasureTime([&]() {
            add_op->Execute(inputs, outputs, ctx_.get());
        }, 1000);
        
        std::cout << "Add operator (" << size << " elements): " 
                  << avg_time << " us, " 
                  << (size / avg_time) << " elements/us" << std::endl;
    }
}

// MatMul性能测试
TEST_F(PerformanceTest, MatMulOperatorPerformance) {
    auto& registry = OperatorRegistry::Instance();
    auto matmul_op = registry.Create("MatMul");
    if (!matmul_op) {
        GTEST_SKIP() << "MatMul operator not available";
    }
    
    // 测试不同矩阵大小
    std::vector<std::pair<int64_t, int64_t>> sizes = {
        {64, 64}, {128, 128}, {256, 256}, {512, 512}
    };
    
    for (auto [m, n] : sizes) {
        Shape shape1({m, n});
        Shape shape2({n, m});
        Shape output_shape({m, m});
        
        auto input1 = CreateTensor(shape1, DataType::FLOAT32);
        auto input2 = CreateTensor(shape2, DataType::FLOAT32);
        auto output = CreateTensor(output_shape, DataType::FLOAT32);
        
        // 填充数据
        float* data1 = static_cast<float*>(input1->GetData());
        float* data2 = static_cast<float*>(input2->GetData());
        for (int64_t i = 0; i < m * n; ++i) {
            data1[i] = 1.0f;
            data2[i] = 1.0f;
        }
        
        std::vector<Tensor*> inputs = {input1.get(), input2.get()};
        std::vector<Tensor*> outputs = {output.get()};
        
        double avg_time = MeasureTime([&]() {
            matmul_op->Execute(inputs, outputs, ctx_.get());
        }, 100);
        
        std::cout << "MatMul operator (" << m << "x" << n << "): " 
                  << avg_time << " us" << std::endl;
    }
}

// Conv性能测试
TEST_F(PerformanceTest, ConvOperatorPerformance) {
    auto& registry = OperatorRegistry::Instance();
    auto conv_op = registry.Create("Conv");
    if (!conv_op) {
        GTEST_SKIP() << "Conv operator not available";
    }
    
    // 测试不同输入大小
    std::vector<std::tuple<int64_t, int64_t, int64_t>> configs = {
        {1, 1, 32},   // batch=1, channels=1, size=32
        {1, 3, 64},   // batch=1, channels=3, size=64
        {4, 16, 128}  // batch=4, channels=16, size=128
    };
    
    for (auto [batch, channels, size] : configs) {
        Shape input_shape({batch, channels, size, size});
        Shape weight_shape({channels, channels, 3, 3});
        Shape output_shape({batch, channels, size - 2, size - 2});
        
        auto input = CreateTensor(input_shape, DataType::FLOAT32);
        auto weight = CreateTensor(weight_shape, DataType::FLOAT32);
        auto output = CreateTensor(output_shape, DataType::FLOAT32);
        
        std::vector<Tensor*> inputs = {input.get(), weight.get()};
        std::vector<Tensor*> outputs = {output.get()};
        
        double avg_time = MeasureTime([&]() {
            conv_op->Execute(inputs, outputs, ctx_.get());
        }, 10);
        
        std::cout << "Conv operator (batch=" << batch 
                  << ", channels=" << channels 
                  << ", size=" << size << "): " 
                  << avg_time << " us" << std::endl;
    }
}

