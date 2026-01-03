// 融合算子实现
// 参考TVM和ONNX Runtime的融合算子实现

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include "simd_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace inferunity {
namespace operators {

// FusedConvBNReLU算子：融合Conv+BatchNorm+ReLU
// 参考NCNN的融合实现
class FusedConvBNReLUOperator : public Operator {
public:
    std::string GetName() const override { return "FusedConvBNReLU"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        // 输入：input, weight, bias(可选), scale, B, mean, var
        if (inputs.size() < 6) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "FusedConvBNReLU requires at least 6 inputs");
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
                               "Input must be 4D (NCHW)");
        }
        
        // 输出形状与输入相同（除了通道数可能不同）
        int64_t batch = input_shape.dims[0];
        int64_t out_c = inputs.size() >= 2 ? inputs[1]->GetShape().dims[0] : input_shape.dims[1];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        
        // 简化的输出尺寸计算
        int64_t kernel_h = inputs[1]->GetShape().dims[2];
        int64_t kernel_w = inputs[1]->GetShape().dims[3];
        int64_t stride_h = 1, stride_w = 1;
        int64_t pad_h = 0, pad_w = 0;
        
        int64_t out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        int64_t out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
        
        output_shapes.push_back(Shape({batch, out_c, out_h, out_w}));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 6 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* weight = inputs[1];
        Tensor* bias = inputs.size() > 2 ? inputs[2] : nullptr;
        Tensor* scale = inputs[3];
        Tensor* B = inputs[4];
        Tensor* mean = inputs[5];
        Tensor* var = inputs[6];
        Tensor* output = outputs[0];
        
        // 融合计算：Conv -> BN -> ReLU
        // 优化实现：预计算BN参数，优化内存访问模式，单次遍历完成所有计算
        
        // 先执行Conv（简化版）
        const Shape& input_shape = input->GetShape();
        const Shape& weight_shape = weight->GetShape();
        const Shape& output_shape = output->GetShape();
        
        int64_t batch = input_shape.dims[0];
        int64_t in_c = input_shape.dims[1];
        int64_t in_h = input_shape.dims[2];
        int64_t in_w = input_shape.dims[3];
        int64_t out_c = output_shape.dims[1];
        int64_t out_h = output_shape.dims[2];
        int64_t out_w = output_shape.dims[3];
        int64_t kernel_h = weight_shape.dims[2];
        int64_t kernel_w = weight_shape.dims[3];
        
        float* input_data = static_cast<float*>(input->GetData());
        float* weight_data = static_cast<float*>(weight->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        float* bias_data = bias ? static_cast<float*>(bias->GetData()) : nullptr;
        float* scale_data = static_cast<float*>(scale->GetData());
        float* B_data = static_cast<float*>(B->GetData());
        float* mean_data = static_cast<float*>(mean->GetData());
        float* var_data = static_cast<float*>(var->GetData());
        
        float epsilon = 1e-5f;
        
        // 优化：预计算BN参数，减少循环内计算
        // BN公式：y = scale * (x - mean) / sqrt(var + eps) + B
        // 可以重写为：y = a * x + b，其中：
        // a = scale / sqrt(var + eps)
        // b = B - scale * mean / sqrt(var + eps)
        std::vector<float> bn_scale(out_c);
        std::vector<float> bn_bias(out_c);
        for (int64_t oc = 0; oc < out_c; ++oc) {
            float inv_std = 1.0f / std::sqrt(var_data[oc] + epsilon);
            bn_scale[oc] = scale_data[oc] * inv_std;
            bn_bias[oc] = B_data[oc] - scale_data[oc] * mean_data[oc] * inv_std;
        }
        
        // 融合计算：在卷积循环中直接应用BN和ReLU
        // 优化：单次遍历，减少内存访问
        std::memset(output_data, 0, output->GetSizeInBytes());
        
        // 优化：按通道处理，改善缓存局部性
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t oc = 0; oc < out_c; ++oc) {
                // 预计算BN参数（每个通道只需计算一次）
                float bn_a = bn_scale[oc];
                float bn_b = bn_bias[oc];
                float conv_bias = bias_data ? bias_data[oc] : 0.0f;
                
                // 合并BN bias和conv bias
                float combined_bias = bn_a * conv_bias + bn_b;
                
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float sum = 0.0f;
                        
                        // Conv计算（优化内存访问模式）
                        for (int64_t ic = 0; ic < in_c; ++ic) {
                            const float* input_channel = input_data + ((n * in_c + ic) * in_h) * in_w;
                            const float* weight_channel = weight_data + ((oc * in_c + ic) * kernel_h) * kernel_w;
                            
                            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                                int64_t ih = oh + kh;
                                if (ih < 0 || ih >= in_h) continue;
                                
                                const float* input_row = input_channel + ih * in_w;
                                const float* weight_row = weight_channel + kh * kernel_w;
                                
                                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                                    int64_t iw = ow + kw;
                                    if (iw >= 0 && iw < in_w) {
                                        sum += input_row[iw] * weight_row[kw];
                                    }
                                }
                            }
                        }
                        
                        // 融合BN和ReLU：y = max(0, a * (x + conv_bias) + bn_b)
                        // 由于已经合并了bias，简化为：y = max(0, bn_a * x + combined_bias)
                        float fused_output = bn_a * sum + combined_bias;
                        float relu_output = (fused_output > 0.0f) ? fused_output : 0.0f;
                        
                        int64_t output_idx = ((n * out_c + oc) * out_h + oh) * out_w + ow;
                        output_data[output_idx] = relu_output;
                    }
                }
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("FusedConvBNReLU", FusedConvBNReLUOperator);

// FusedMatMulAdd算子：融合MatMul+Add（类似GEMM）
// 参考ONNX Runtime的Gemm融合
class FusedMatMulAddOperator : public Operator {
public:
    std::string GetName() const override { return "FusedMatMulAdd"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 3) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "FusedMatMulAdd requires 3 inputs (A, B, bias)");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        const Shape& shape0 = inputs[0]->GetShape();
        const Shape& shape1 = inputs[1]->GetShape();
        
        if (shape0.dims.size() < 2 || shape1.dims.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Inputs must be at least 2D");
        }
        
        // [M, K] x [K, N] = [M, N]
        std::vector<int64_t> output_dims = {shape0.dims[0], shape1.dims[1]};
        output_shapes.push_back(Shape(output_dims));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 3 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* A = inputs[0];
        Tensor* B = inputs[1];
        Tensor* bias = inputs[2];
        Tensor* output = outputs[0];
        
        const Shape& shape0 = A->GetShape();
        const Shape& shape1 = B->GetShape();
        
        int64_t M = shape0.dims[0];
        int64_t K = shape0.dims[1];
        int64_t N = shape1.dims[1];
        
        const float* A_data = static_cast<const float*>(A->GetData());
        const float* B_data = static_cast<const float*>(B->GetData());
        const float* bias_data = static_cast<const float*>(bias->GetData());
        float* C_data = static_cast<float*>(output->GetData());
        
        // 融合计算：MatMul + Add（类似GEMM）
        // 使用SIMD优化（如果可用）
        if (M * N * K > 1000) {  // 对于大矩阵使用SIMD
            // 先计算MatMul
            simd::MatMulSIMD(A_data, B_data, C_data, M, K, N);
            // 然后添加bias
            if (bias_data) {
                for (int64_t i = 0; i < M; ++i) {
                    simd::AddSIMD(C_data + i * N, bias_data, C_data + i * N, N);
                }
            }
        } else {
            // 小矩阵使用标准实现
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    float sum = bias_data ? bias_data[j] : 0.0f;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A_data[i * K + k] * B_data[k * N + j];
                    }
                    C_data[i * N + j] = sum;
                }
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("FusedMatMulAdd", FusedMatMulAddOperator);

} // namespace operators
} // namespace inferunity

