// Conv算子实现
// 参考NCNN的卷积实现，但简化版本

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace inferunity {
namespace operators {

class ConvOperator : public Operator {
public:
    std::string GetName() const override { return "Conv"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Conv requires at least 2 inputs (input and weight)");
        }
        if (inputs[0]->GetDataType() != DataType::FLOAT32) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Conv only supports FLOAT32");
        }
        // 检查输入维度（Conv需要4D输入：NCHW）
        const Shape& input_shape = inputs[0]->GetShape();
        if (input_shape.dims.size() < 4) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Conv input must be 4D (NCHW format)");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
        }
        
        const Shape& input_shape = inputs[0]->GetShape();
        if (input_shape.dims.size() < 4) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Conv input must be 4D (NCHW)");
        }
        
        // 获取属性
        int64_t kernel_h = 3, kernel_w = 3;
        int64_t stride_h = 1, stride_w = 1;
        int64_t pad_h = 0, pad_w = 0;
        int64_t dilation_h = 1, dilation_w = 1;
        
        // 解析属性（从Node获取，这里简化处理）
        // 实际应该从Node的attributes获取
        
        int64_t batch = input_shape.dims[0];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        
        if (inputs.size() >= 2) {
            const Shape& weight_shape = inputs[1]->GetShape();
            if (weight_shape.dims.size() >= 2) {
                kernel_h = weight_shape.dims[2];
                kernel_w = weight_shape.dims[3];
            }
        }
        
        // 计算输出尺寸
        int64_t out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int64_t out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
        int64_t out_c = inputs.size() >= 2 ? inputs[1]->GetShape().dims[0] : input_shape.dims[1];
        
        output_shapes.push_back(Shape({batch, out_c, out_h, out_w}));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* weight = inputs[1];
        Tensor* bias = inputs.size() > 2 ? inputs[2] : nullptr;
        Tensor* output = outputs[0];
        
        const Shape& input_shape = input->GetShape();
        const Shape& weight_shape = weight->GetShape();
        const Shape& output_shape = output->GetShape();
        
        // 简化的卷积实现（参考NCNN，但更基础）
        // 实际应该使用im2col + GEMM或直接卷积
        int64_t batch = input_shape.dims[0];
        int64_t in_c = input_shape.dims[1];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        int64_t out_c = output_shape.dims[1];
        int64_t out_h = output_shape.dims[2];
        int64_t out_w = output_shape.dims[3];
        int64_t kernel_h = weight_shape.dims[2];
        int64_t kernel_w = weight_shape.dims[3];
        
        // 获取属性
        int64_t stride_h = 1, stride_w = 1;
        int64_t pad_h = 0, pad_w = 0;
        int64_t dilation_h = 1, dilation_w = 1;
        
        float* input_data = static_cast<float*>(input->GetData());
        float* weight_data = static_cast<float*>(weight->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        float* bias_data = bias ? static_cast<float*>(bias->GetData()) : nullptr;
        
        // 初始化输出
        std::memset(output_data, 0, output->GetSizeInBytes());
        
        // 简化的卷积计算（参考NCNN的基础实现）
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t oc = 0; oc < out_c; ++oc) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float sum = bias_data ? bias_data[oc] : 0.0f;
                        
                        for (int64_t ic = 0; ic < in_c; ++ic) {
                            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                    int64_t ih = oh * stride_h + kh * dilation_h - pad_h;
                                    int64_t iw = ow * stride_w + kw * dilation_w - pad_w;
                                    
                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        int64_t input_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                                        int64_t weight_idx = ((oc * in_c + ic) * kernel_h + kh) * kernel_w + kw;
                                        sum += input_data[input_idx] * weight_data[weight_idx];
                                    }
                                }
                            }
                        }
                        
                        int64_t output_idx = ((n * out_c + oc) * out_h + oh) * out_w + ow;
                        output_data[output_idx] = sum;
                    }
                }
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Conv", ConvOperator);

} // namespace operators
} // namespace inferunity

