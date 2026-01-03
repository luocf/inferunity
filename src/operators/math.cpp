// 基础数学运算算子实现
// 参考TensorFlow Lite的实现

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <cstring>

namespace inferunity {
namespace operators {

// Add算子
class AddOperator : public Operator {
public:
    std::string GetName() const override { return "Add"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Add requires 2 inputs");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        // 广播规则：输出形状是输入形状的广播结果
        // 简化：假设形状相同
        output_shapes.push_back(inputs[0]->GetShape());
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input0 = inputs[0];
        Tensor* input1 = inputs[1];
        Tensor* output = outputs[0];
        
        size_t count = output->GetElementCount();
        const float* data0 = static_cast<const float*>(input0->GetData());
        const float* data1 = static_cast<const float*>(input1->GetData());
        float* out_data = static_cast<float*>(output->GetData());
        
        // 逐元素加法（简化：假设形状相同）
        for (size_t i = 0; i < count; ++i) {
            out_data[i] = data0[i] + data1[i];
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Add", AddOperator);

// Mul算子
class MulOperator : public Operator {
public:
    std::string GetName() const override { return "Mul"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Mul requires 2 inputs");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        output_shapes.push_back(inputs[0]->GetShape());
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input0 = inputs[0];
        Tensor* input1 = inputs[1];
        Tensor* output = outputs[0];
        
        size_t count = output->GetElementCount();
        const float* data0 = static_cast<const float*>(input0->GetData());
        const float* data1 = static_cast<const float*>(input1->GetData());
        float* out_data = static_cast<float*>(output->GetData());
        
        // 逐元素乘法
        for (size_t i = 0; i < count; ++i) {
            out_data[i] = data0[i] * data1[i];
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Mul", MulOperator);

// MatMul算子（矩阵乘法）
class MatMulOperator : public Operator {
public:
    std::string GetName() const override { return "MatMul"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "MatMul requires 2 inputs");
        }
        const Shape& shape0 = inputs[0]->GetShape();
        const Shape& shape1 = inputs[1]->GetShape();
        if (shape0.dims.size() < 2 || shape1.dims.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "MatMul inputs must be at least 2D");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        const Shape& shape0 = inputs[0]->GetShape();
        const Shape& shape1 = inputs[1]->GetShape();
        
        // 简化：假设是2D矩阵 [M, K] x [K, N] = [M, N]
        std::vector<int64_t> output_dims;
        if (shape0.dims.size() == 2 && shape1.dims.size() == 2) {
            output_dims = {shape0.dims[0], shape1.dims[1]};
        } else {
            // 处理batch维度
            output_dims = shape0.dims;
            output_dims.back() = shape1.dims.back();
        }
        
        output_shapes.push_back(Shape(output_dims));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input0 = inputs[0];
        Tensor* input1 = inputs[1];
        Tensor* output = outputs[0];
        
        const Shape& shape0 = input0->GetShape();
        const Shape& shape1 = input1->GetShape();
        
        // 简化：2D矩阵乘法
        int64_t M = shape0.dims[0];
        int64_t K = shape0.dims[1];
        int64_t N = shape1.dims[1];
        
        const float* A = static_cast<const float*>(input0->GetData());
        const float* B = static_cast<const float*>(input1->GetData());
        float* C = static_cast<float*>(output->GetData());
        
        // 初始化输出
        std::memset(C, 0, output->GetSizeInBytes());
        
        // 矩阵乘法: C = A * B
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("MatMul", MatMulOperator);

// Sub算子（减法）
class SubOperator : public Operator {
public:
    std::string GetName() const override { return "Sub"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Sub requires 2 inputs");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        output_shapes.push_back(inputs[0]->GetShape());
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input0 = inputs[0];
        Tensor* input1 = inputs[1];
        Tensor* output = outputs[0];
        
        size_t count = output->GetElementCount();
        const float* data0 = static_cast<const float*>(input0->GetData());
        const float* data1 = static_cast<const float*>(input1->GetData());
        float* out_data = static_cast<float*>(output->GetData());
        
        // 逐元素减法
        for (size_t i = 0; i < count; ++i) {
            out_data[i] = data0[i] - data1[i];
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Sub", SubOperator);

// Div算子（除法）
class DivOperator : public Operator {
public:
    std::string GetName() const override { return "Div"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Div requires 2 inputs");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        output_shapes.push_back(inputs[0]->GetShape());
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input0 = inputs[0];
        Tensor* input1 = inputs[1];
        Tensor* output = outputs[0];
        
        size_t count = output->GetElementCount();
        const float* data0 = static_cast<const float*>(input0->GetData());
        const float* data1 = static_cast<const float*>(input1->GetData());
        float* out_data = static_cast<float*>(output->GetData());
        
        // 逐元素除法（避免除零）
        for (size_t i = 0; i < count; ++i) {
            float divisor = data1[i];
            if (std::abs(divisor) < 1e-8f) {
                out_data[i] = 0.0f;  // 避免除零
            } else {
                out_data[i] = data0[i] / divisor;
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Div", DivOperator);

} // namespace operators
} // namespace inferunity

