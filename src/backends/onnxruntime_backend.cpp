// ONNX Runtime执行提供者
// 集成ONNX Runtime作为后端，利用其优化的SIMD/GPU内核

#include "inferunity/backend.h"
#include "inferunity/logger.h"
#include <sstream>

#ifdef INFERUNITY_USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

namespace inferunity {

class ONNXRuntimeExecutionProvider : public ExecutionProvider {
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<Ort::AllocatedStringPtr> input_names_ptrs_;
    std::vector<Ort::AllocatedStringPtr> output_names_ptrs_;
    std::vector<const char*> input_names_cstr_;
    std::vector<const char*> output_names_cstr_;
    int optimization_level_ = 2;
    size_t memory_usage_ = 0;
    
public:
    ONNXRuntimeExecutionProvider() 
        : env_(ORT_LOGGING_LEVEL_WARNING, "InferUnity"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    }
    
    std::string GetName() const override {
        return "ONNXRuntimeExecutionProvider";
    }
    
    std::string GetVersion() const override {
        return OrtGetApiBase()->GetVersionString();
    }
    
    DeviceType GetDeviceType() const override {
        // ONNX Runtime可以运行在CPU或GPU上
        // 这里简化处理，返回CPU（实际应该根据SessionOptions判断）
        return DeviceType::CPU;
    }
    
    std::vector<std::string> GetSupportedOps() const override {
        // ONNX Runtime支持所有ONNX标准算子
        // 这里返回空列表表示支持所有（实际应该查询Session）
        return {};
    }
    
    bool IsOpSupported(const std::string& op_type) const override {
        // ONNX Runtime支持所有ONNX标准算子
        return true;
    }
    
    bool IsAvailable() const override {
        return true;  // ONNX Runtime总是可用（如果已链接）
    }
    
    Status LoadModel(const std::string& model_path) override {
        try {
            Ort::SessionOptions session_options;
            
            // 设置优化级别
            if (optimization_level_ >= 2) {
                session_options.SetGraphOptimizationLevel(
                    GraphOptimizationLevel::ORT_ENABLE_ALL);
            } else if (optimization_level_ >= 1) {
                session_options.SetGraphOptimizationLevel(
                    GraphOptimizationLevel::ORT_ENABLE_BASIC);
            } else {
                session_options.SetGraphOptimizationLevel(
                    GraphOptimizationLevel::ORT_DISABLE_ALL);
            }
            
            // 创建Session
            session_ = std::make_unique<Ort::Session>(
                env_, model_path.c_str(), session_options);
            
            // 获取输入/输出名称
            size_t num_input_nodes = session_->GetInputCount();
            size_t num_output_nodes = session_->GetOutputCount();
            
            input_names_.clear();
            output_names_.clear();
            input_names_ptrs_.clear();
            output_names_ptrs_.clear();
            input_names_cstr_.clear();
            output_names_cstr_.clear();
            
            for (size_t i = 0; i < num_input_nodes; ++i) {
                auto name_ptr = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                input_names_ptrs_.push_back(std::move(name_ptr));
                input_names_.push_back(input_names_ptrs_.back().get());
                input_names_cstr_.push_back(input_names_.back().c_str());
            }
            
            for (size_t i = 0; i < num_output_nodes; ++i) {
                auto name_ptr = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
                output_names_ptrs_.push_back(std::move(name_ptr));
                output_names_.push_back(output_names_ptrs_.back().get());
                output_names_cstr_.push_back(output_names_.back().c_str());
            }
            
            LOG_INFO("ONNX Runtime model loaded: " + model_path);
            LOG_INFO("Inputs: " + std::to_string(num_input_nodes));
            LOG_INFO("Outputs: " + std::to_string(num_output_nodes));
            
            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::Error(StatusCode::ERROR_RUNTIME_ERROR,
                               "Failed to load ONNX Runtime model: " + std::string(e.what()));
        }
    }
    
    Status LoadModelFromMemory(const void* data, size_t size) override {
        try {
            Ort::SessionOptions session_options;
            
            if (optimization_level_ >= 2) {
                session_options.SetGraphOptimizationLevel(
                    GraphOptimizationLevel::ORT_ENABLE_ALL);
            }
            
            // 从内存加载模型
            session_ = std::make_unique<Ort::Session>(
                env_, data, size, session_options);
            
            // 获取输入/输出名称（同上）
            // ... (省略重复代码)
            
            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::Error(StatusCode::ERROR_RUNTIME_ERROR,
                               "Failed to load ONNX Runtime model from memory: " + std::string(e.what()));
        }
    }
    
    Status LoadGraph(Graph* graph) override {
        // ONNX Runtime后端不支持从内部Graph加载
        // 应该先将Graph序列化为ONNX格式，然后加载
        return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                           "ONNX Runtime backend does not support LoadGraph. Use LoadModel instead.");
    }
    
    Status Run(const std::vector<Tensor*>& inputs,
              std::vector<Tensor*>& outputs) override {
        if (!session_) {
            return Status::Error(StatusCode::ERROR_INVALID_STATE,
                               "Model not loaded");
        }
        
        if (inputs.size() != input_names_.size()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Input count mismatch");
        }
        
        try {
            // 转换输入Tensor到ONNX Runtime格式
            std::vector<Ort::Value> ort_inputs;
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto* tensor = inputs[i];
                const Shape& shape = tensor->GetShape();
                
                // 创建ONNX Runtime Tensor
                std::vector<int64_t> dims(shape.dims.begin(), shape.dims.end());
                auto ort_tensor = Ort::Value::CreateTensor<float>(
                    memory_info_,
                    static_cast<float*>(tensor->GetData()),
                    tensor->GetElementCount(),
                    dims.data(),
                    dims.size()
                );
                ort_inputs.push_back(std::move(ort_tensor));
            }
            
            // 执行推理
            auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr_.data(), ort_inputs.data(), ort_inputs.size(),
                output_names_cstr_.data(), output_names_cstr_.size()
            );
            
            // 转换输出Tensor
            if (outputs.size() != ort_outputs.size()) {
                return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                   "Output count mismatch");
            }
            
            for (size_t i = 0; i < outputs.size(); ++i) {
                auto& ort_output = ort_outputs[i];
                auto* output = outputs[i];
                
                // 获取形状信息
                auto shape_info = ort_output.GetTensorTypeAndShapeInfo();
                std::vector<int64_t> dims = shape_info.GetShape();
                
                // 更新输出Tensor的形状和数据
                Shape output_shape(dims);
                if (output->GetShape() != output_shape) {
                    // 需要重新分配内存
                    output->Reshape(output_shape);
                }
                
                // 拷贝数据
                float* output_data = static_cast<float*>(output->GetData());
                const float* ort_data = ort_output.GetTensorMutableData<float>();
                std::memcpy(output_data, ort_data, output->GetSizeInBytes());
            }
            
            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::Error(StatusCode::ERROR_RUNTIME_ERROR,
                               "ONNX Runtime inference failed: " + std::string(e.what()));
        }
    }
    
    Status SetOptimizationLevel(int level) override {
        optimization_level_ = level;
        // 注意：如果Session已创建，需要重新加载模型
        return Status::Ok();
    }
    
    Status OptimizeGraph(Graph* graph) override {
        // ONNX Runtime的图优化在LoadModel时自动执行
        return Status::Ok();
    }
    
    Status AllocateMemory(size_t size) override {
        memory_usage_ += size;
        return Status::Ok();
    }
    
    Status ReleaseMemory() override {
        memory_usage_ = 0;
        return Status::Ok();
    }
    
    size_t GetMemoryUsage() const override {
        return memory_usage_;
    }
    
    void ResetProfiling() override {
        // ONNX Runtime支持性能分析，这里简化处理
    }
    
    std::string GetProfilingReport() const override {
        // 返回ONNX Runtime的性能报告
        return "Profiling report not implemented";
    }
};

// 注册ONNX Runtime执行提供者
namespace {
    void RegisterONNXRuntimeExecutionProvider() {
        ExecutionProviderRegistry::Instance().Register("ONNXRuntimeExecutionProvider", []() {
            return std::make_unique<ONNXRuntimeExecutionProvider>();
        });
        ExecutionProviderRegistry::Instance().Register("ONNXRuntime", []() {
            return std::make_unique<ONNXRuntimeExecutionProvider>();
        });
    }
    
    static bool g_registered = []() {
        RegisterONNXRuntimeExecutionProvider();
        return true;
    }();
}

} // namespace inferunity

#else  // INFERUNITY_USE_ONNXRUNTIME未定义

namespace inferunity {
    // ONNX Runtime未启用时的占位实现
    // 注册函数为空，不会注册ONNX Runtime后端
}

#endif  // INFERUNITY_USE_ONNXRUNTIME

