// 激活函数算子实现
// 参考NCNN的激活函数实现

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <cmath>

namespace inferunity {
namespace operators {

// ReLU算子
class ReluOperator : public Operator {
public:
    std::string GetName() const override { return "Relu"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
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
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        size_t count = input->GetElementCount();
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // ReLU: max(0, x)
        for (size_t i = 0; i < count; ++i) {
            output_data[i] = std::max(0.0f, input_data[i]);
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Relu", ReluOperator);

// Sigmoid算子
class SigmoidOperator : public Operator {
public:
    std::string GetName() const override { return "Sigmoid"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
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
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        size_t count = input->GetElementCount();
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // Sigmoid: 1 / (1 + exp(-x))
        for (size_t i = 0; i < count; ++i) {
            output_data[i] = 1.0f / (1.0f + std::exp(-input_data[i]));
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Sigmoid", SigmoidOperator);

// Tanh算子
class TanhOperator : public Operator {
public:
    std::string GetName() const override { return "Tanh"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
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
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        size_t count = input->GetElementCount();
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // Tanh: tanh(x)
        for (size_t i = 0; i < count; ++i) {
            output_data[i] = std::tanh(input_data[i]);
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Tanh", TanhOperator);

// GELU算子（Gaussian Error Linear Unit，Transformer常用）
// GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// 近似实现：GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
class GeluOperator : public Operator {
public:
    std::string GetName() const override { return "Gelu"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
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
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        size_t count = input->GetElementCount();
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // GELU近似实现：0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
        const float coeff = 0.044715f;
        
        for (size_t i = 0; i < count; ++i) {
            float x = input_data[i];
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x3);
            output_data[i] = 0.5f * x * (1.0f + std::tanh(inner));
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Gelu", GeluOperator);
// 注意：GELU和Gelu指向同一个算子类，在CPU后端支持列表中已包含

// SiLU算子（Sigmoid Linear Unit，Swish激活函数）
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
class SiluOperator : public Operator {
public:
    std::string GetName() const override { return "Silu"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
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
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        size_t count = input->GetElementCount();
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        for (size_t i = 0; i < count; ++i) {
            float x = input_data[i];
            float sigmoid_x = 1.0f / (1.0f + std::exp(-x));
            output_data[i] = x * sigmoid_x;
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Silu", SiluOperator);
// 注意：SiLU和Swish在CPU后端支持列表中已包含，指向同一个算子类

} // namespace operators
} // namespace inferunity

