// 归一化算子实现
// 参考NCNN的BatchNormalization实现

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <cmath>

namespace inferunity {
namespace operators {

// BatchNormalization算子
class BatchNormalizationOperator : public Operator {
public:
    std::string GetName() const override { return "BatchNormalization"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 5) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "BatchNormalization requires 5 inputs");
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
        if (inputs.size() < 5 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];      // X
        Tensor* scale = inputs[1];      // scale (gamma)
        Tensor* bias = inputs[2];       // B (beta)
        Tensor* mean = inputs[3];      // mean
        Tensor* var = inputs[4];        // variance
        Tensor* output = outputs[0];
        
        const Shape& input_shape = input->GetShape();
        int64_t channels = input_shape.dims[1];
        
        // 获取epsilon（默认1e-5）
        float epsilon = 1e-5f;
        
        const float* input_data = static_cast<const float*>(input->GetData());
        const float* scale_data = static_cast<const float*>(scale->GetData());
        const float* bias_data = static_cast<const float*>(bias->GetData());
        const float* mean_data = static_cast<const float*>(mean->GetData());
        const float* var_data = static_cast<const float*>(var->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        size_t count = input->GetElementCount();
        int64_t spatial_size = count / (input_shape.dims[0] * channels);
        
        // BatchNorm: y = scale * (x - mean) / sqrt(var + epsilon) + bias
        for (size_t i = 0; i < count; ++i) {
            int64_t c = (i / spatial_size) % channels;
            float normalized = (input_data[i] - mean_data[c]) / std::sqrt(var_data[c] + epsilon);
            output_data[i] = scale_data[c] * normalized + bias_data[c];
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("BatchNormalization", BatchNormalizationOperator);

// LayerNormalization算子（Transformer常用）
class LayerNormalizationOperator : public Operator {
public:
    std::string GetName() const override { return "LayerNormalization"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "LayerNormalization requires at least 2 inputs (input, scale)");
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
        
        Tensor* input = inputs[0];
        Tensor* scale = inputs[1];
        Tensor* bias = inputs.size() > 2 ? inputs[2] : nullptr;
        Tensor* output = outputs[0];
        
        // 获取axis属性（默认-1，即最后一个维度）
        int axis = -1;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
        }
        
        // 获取epsilon属性（默认1e-5）
        float epsilon = 1e-5f;
        auto epsilon_attr = GetAttribute("epsilon");
        if (epsilon_attr.GetType() == AttributeValue::Type::FLOAT) {
            epsilon = epsilon_attr.GetFloat();
        }
        
        const Shape& input_shape = input->GetShape();
        int rank = static_cast<int>(input_shape.dims.size());
        
        if (axis < 0) {
            axis += rank;
        }
        
        // 计算归一化维度的大小
        int64_t norm_size = 1;
        for (int i = axis; i < rank; ++i) {
            norm_size *= input_shape.dims[i];
        }
        
        const float* input_data = static_cast<const float*>(input->GetData());
        const float* scale_data = static_cast<const float*>(scale->GetData());
        const float* bias_data = bias ? static_cast<const float*>(bias->GetData()) : nullptr;
        float* output_data = static_cast<float*>(output->GetData());
        
        size_t total_count = input->GetElementCount();
        int64_t num_groups = total_count / norm_size;
        
        // LayerNorm: y = scale * (x - mean) / sqrt(var + eps) + bias
        for (int64_t g = 0; g < num_groups; ++g) {
            const float* group_input = input_data + g * norm_size;
            float* group_output = output_data + g * norm_size;
            
            // 计算mean
            float mean = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                mean += group_input[i];
            }
            mean /= norm_size;
            
            // 计算var
            float var = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float diff = group_input[i] - mean;
                var += diff * diff;
            }
            var /= norm_size;
            
            // 归一化
            float inv_std = 1.0f / std::sqrt(var + epsilon);
            for (int64_t i = 0; i < norm_size; ++i) {
                float normalized = (group_input[i] - mean) * inv_std;
                float scaled = scale_data[i] * normalized;
                group_output[i] = bias_data ? scaled + bias_data[i] : scaled;
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("LayerNormalization", LayerNormalizationOperator);

// RMSNorm算子（Root Mean Square Layer Normalization，部分Transformer模型使用）
// RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * scale
class RMSNormOperator : public Operator {
public:
    std::string GetName() const override { return "RMSNorm"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "RMSNorm requires at least 2 inputs (input, scale)");
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
        
        Tensor* input = inputs[0];
        Tensor* scale = inputs[1];
        Tensor* output = outputs[0];
        
        // 获取epsilon属性（默认1e-6）
        float epsilon = 1e-6f;
        auto epsilon_attr = GetAttribute("epsilon");
        if (epsilon_attr.GetType() == AttributeValue::Type::FLOAT) {
            epsilon = epsilon_attr.GetFloat();
        }
        
        const Shape& input_shape = input->GetShape();
        int rank = static_cast<int>(input_shape.dims.size());
        
        // 默认在最后一个维度归一化
        int axis = rank - 1;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
            if (axis < 0) axis += rank;
        }
        
        // 计算归一化维度的大小
        int64_t norm_size = 1;
        for (int i = axis; i < rank; ++i) {
            norm_size *= input_shape.dims[i];
        }
        
        const float* input_data = static_cast<const float*>(input->GetData());
        const float* scale_data = static_cast<const float*>(scale->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        size_t total_count = input->GetElementCount();
        int64_t num_groups = total_count / norm_size;
        
        // RMSNorm: y = (x / sqrt(mean(x^2) + eps)) * scale
        for (int64_t g = 0; g < num_groups; ++g) {
            const float* group_input = input_data + g * norm_size;
            float* group_output = output_data + g * norm_size;
            
            // 计算mean(x^2)
            float mean_sq = 0.0f;
            for (int64_t i = 0; i < norm_size; ++i) {
                float val = group_input[i];
                mean_sq += val * val;
            }
            mean_sq /= static_cast<float>(norm_size);
            
            // 归一化：计算inv_rms，避免除零
            float rms_sq = mean_sq + epsilon;
            if (rms_sq <= 0.0f) {
                rms_sq = epsilon;  // 防止数值问题
            }
            float inv_rms = 1.0f / std::sqrt(rms_sq);
            
            // 应用归一化和scale
            // scale的形状应该匹配归一化维度
            const Shape& scale_shape = scale->GetShape();
            int64_t scale_size = scale_shape.GetElementCount();
            
            for (int64_t i = 0; i < norm_size; ++i) {
                // scale索引：如果scale_size == norm_size，直接使用i
                // 否则使用 i % scale_size 来处理广播
                int64_t scale_idx = (scale_size == norm_size) ? i : (i % scale_size);
                group_output[i] = group_input[i] * inv_rms * scale_data[scale_idx];
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("RMSNorm", RMSNormOperator);

} // namespace operators
} // namespace inferunity

