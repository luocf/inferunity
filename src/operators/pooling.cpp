// 池化算子实现
// 参考NCNN的池化实现

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <cmath>
#include <climits>

namespace inferunity {
namespace operators {

// MaxPool算子
class MaxPoolOperator : public Operator {
public:
    std::string GetName() const override { return "MaxPool"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
        }
        if (inputs[0]->GetShape().dims.size() < 4) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "MaxPool input must be 4D (NCHW)");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        const Shape& input_shape = inputs[0]->GetShape();
        
        // 获取属性（简化：默认值）
        int64_t kernel_h = 2, kernel_w = 2;
        int64_t stride_h = 2, stride_w = 2;
        int64_t pad_h = 0, pad_w = 0;
        
        int64_t batch = input_shape.dims[0];
        int64_t channels = input_shape.dims[1];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        
        int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
        
        output_shapes.push_back(Shape({batch, channels, out_h, out_w}));
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
        const Shape& output_shape = output->GetShape();
        
        int64_t batch = input_shape.dims[0];
        int64_t channels = input_shape.dims[1];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        int64_t out_h = output_shape.dims[2];
        int64_t out_w = output_shape.dims[3];
        
        // 默认参数
        int64_t kernel_h = 2, kernel_w = 2;
        int64_t stride_h = 2, stride_w = 2;
        int64_t pad_h = 0, pad_w = 0;
        
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float max_val = -std::numeric_limits<float>::max();
                        
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                int64_t ih = oh * stride_h + kh - pad_h;
                                int64_t iw = ow * stride_w + kw - pad_w;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int64_t idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                                    max_val = std::max(max_val, input_data[idx]);
                                }
                            }
                        }
                        
                        int64_t out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                        output_data[out_idx] = max_val;
                    }
                }
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("MaxPool", MaxPoolOperator);

// AveragePool算子
class AveragePoolOperator : public Operator {
public:
    std::string GetName() const override { return "AveragePool"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
        }
        if (inputs[0]->GetShape().dims.size() < 4) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "AveragePool input must be 4D (NCHW)");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        const Shape& input_shape = inputs[0]->GetShape();
        
        int64_t kernel_h = 2, kernel_w = 2;
        int64_t stride_h = 2, stride_w = 2;
        int64_t pad_h = 0, pad_w = 0;
        
        int64_t batch = input_shape.dims[0];
        int64_t channels = input_shape.dims[1];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        
        int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
        
        output_shapes.push_back(Shape({batch, channels, out_h, out_w}));
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
        const Shape& output_shape = output->GetShape();
        
        int64_t batch = input_shape.dims[0];
        int64_t channels = input_shape.dims[1];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        int64_t out_h = output_shape.dims[2];
        int64_t out_w = output_shape.dims[3];
        
        int64_t kernel_h = 2, kernel_w = 2;
        int64_t stride_h = 2, stride_w = 2;
        int64_t pad_h = 0, pad_w = 0;
        
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float sum = 0.0f;
                        int count = 0;
                        
                        for (int64_t kh = 0; kh < kernel_h; ++kh) {
                            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                int64_t ih = oh * stride_h + kh - pad_h;
                                int64_t iw = ow * stride_w + kw - pad_w;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int64_t idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                                    sum += input_data[idx];
                                    count++;
                                }
                            }
                        }
                        
                        int64_t out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                        output_data[out_idx] = count > 0 ? sum / count : 0.0f;
                    }
                }
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("AveragePool", AveragePoolOperator);

} // namespace operators
} // namespace inferunity

