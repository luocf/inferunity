// Softmax算子实现
// 参考TensorFlow Lite的实现

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <cmath>

namespace inferunity {
namespace operators {

// Softmax算子
class SoftmaxOperator : public Operator {
public:
    std::string GetName() const override { return "Softmax"; }
    
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
        
        const Shape& input_shape = input->GetShape();
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // 获取axis（默认-1，最后一个维度）
        int axis = -1;
        if (input_shape.dims.size() > 0) {
            axis = axis < 0 ? input_shape.dims.size() + axis : axis;
        }
        
        // 计算softmax维度
        int64_t outer_size = 1;
        int64_t inner_size = 1;
        int64_t softmax_size = input_shape.dims[axis];
        
        for (int i = 0; i < axis; ++i) {
            outer_size *= input_shape.dims[i];
        }
        for (int i = axis + 1; i < static_cast<int>(input_shape.dims.size()); ++i) {
            inner_size *= input_shape.dims[i];
        }
        
        // Softmax计算
        for (int64_t outer = 0; outer < outer_size; ++outer) {
            for (int64_t inner = 0; inner < inner_size; ++inner) {
                // 找到最大值（数值稳定性）
                float max_val = -std::numeric_limits<float>::max();
                for (int64_t i = 0; i < softmax_size; ++i) {
                    int64_t idx = (outer * softmax_size + i) * inner_size + inner;
                    max_val = std::max(max_val, input_data[idx]);
                }
                
                // 计算exp和sum
                float sum = 0.0f;
                for (int64_t i = 0; i < softmax_size; ++i) {
                    int64_t idx = (outer * softmax_size + i) * inner_size + inner;
                    output_data[idx] = std::exp(input_data[idx] - max_val);
                    sum += output_data[idx];
                }
                
                // 归一化
                for (int64_t i = 0; i < softmax_size; ++i) {
                    int64_t idx = (outer * softmax_size + i) * inner_size + inner;
                    output_data[idx] /= sum;
                }
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Softmax", SoftmaxOperator);

} // namespace operators
} // namespace inferunity

