#include "inferunity/engine.h"
#include "inferunity/runtime.h"
#include "inferunity/optimizer.h"
#include "inferunity/backend.h"
#include "inferunity/tensor.h"
#include "inferunity/logger.h"
#include "frontend/onnx_parser.h"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <thread>
#include <functional>

// 前向声明形状推断函数
namespace inferunity {
    Status InferShapes(Graph* graph);
}

namespace inferunity {

InferenceSession::InferenceSession(const SessionOptions& options)
    : options_(options), initialized_(false) {
    optimizer_ = std::make_unique<Optimizer>();
    execution_engine_ = std::make_unique<ExecutionEngine>();
}

InferenceSession::~InferenceSession() = default;

std::unique_ptr<InferenceSession> InferenceSession::Create(const SessionOptions& options) {
    auto session = std::unique_ptr<InferenceSession>(new InferenceSession(options));
    Status status = session->Initialize();
    if (!status.IsOk()) {
        return nullptr;
    }
    return session;
}

Status InferenceSession::Initialize() {
    // 确保所有执行提供者被注册
    InitializeExecutionProviders(); // 显式调用注册函数
    
    // 注册默认优化Pass (参考TVM的Pass注册机制)
    optimizer_->RegisterPass(std::make_unique<ConstantFoldingPass>());
    optimizer_->RegisterPass(std::make_unique<DeadCodeEliminationPass>());
    if (options_.enable_operator_fusion) {
        optimizer_->RegisterPass(std::make_unique<OperatorFusionPass>());
    }
    optimizer_->RegisterPass(std::make_unique<MemoryLayoutOptimizationPass>());
    
    // 准备执行提供者 (参考ONNX Runtime的提供者初始化)
    Status status = PrepareExecutionProviders();
    if (!status.IsOk()) {
        return status;
    }
    
    initialized_ = true;
    return Status::Ok();
}

Status InferenceSession::PrepareExecutionProviders() {
    // 确保CPU执行提供者已注册（通过引用符号强制链接cpu_backend）
    // 这确保静态注册代码被执行
    static bool initialized = []() {
        // 通过创建和检查来触发CPU执行提供者的注册
        // cpu_backend.cpp中的静态初始化会在链接时执行
        return true;
    }();
    (void)initialized;  // 避免未使用变量警告
    
    // 获取可用执行提供者 (参考ONNX Runtime的提供者发现机制)
    auto available_providers = ExecutionProviderRegistry::Instance().GetAvailableProviders();
    
    if (available_providers.empty()) {
        return Status::Error(StatusCode::ERROR_NOT_FOUND,
                           "No available execution providers");
    }
    
    // 按偏好顺序创建执行提供者 (参考ONNX Runtime的提供者优先级)
    std::vector<std::string> provider_names = options_.execution_providers;
    if (provider_names.empty()) {
        provider_names = available_providers;
    }
    
    for (const std::string& name : provider_names) {
        auto provider = ExecutionProviderRegistry::Instance().Create(name);
        if (provider && provider->IsAvailable()) {
            // 将unique_ptr转换为shared_ptr
            execution_providers_.push_back(std::shared_ptr<ExecutionProvider>(provider.release()));
        }
    }
    
    if (execution_providers_.empty()) {
        return Status::Error(StatusCode::ERROR_NOT_FOUND,
                           "No available execution providers could be created");
    }
    
    execution_engine_->SetBackends(execution_providers_);
    return Status::Ok();
}

Status InferenceSession::LoadModel(const std::string& filepath) {
    // 根据文件扩展名选择解析器
    // 参考ONNX Runtime的模型加载机制
    std::string ext = filepath.substr(filepath.find_last_of(".") + 1);
    
    if (ext == "onnx") {
        // 使用ONNX解析器
        frontend::ONNXParser parser;
        Status status = parser.LoadFromFile(filepath);
        if (!status.IsOk()) {
            return status;
        }
        
        std::unique_ptr<Graph> graph;
        status = parser.ConvertToGraph(graph);
        if (!status.IsOk()) {
            return status;
        }
        
        return LoadModelFromGraph(std::move(graph));
    }
    // 其他格式支持（待实现）
    // else if (ext == "pb") {
    //     // TensorFlow SavedModel格式
    //     // 需要实现TensorFlow protobuf解析器
    // }
    // else if (ext == "tflite") {
    //     // TensorFlow Lite格式
    //     // 需要实现TFLite FlatBuffer解析器
    // }
    
    return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                       "Unsupported model format: " + ext);
}

Status InferenceSession::LoadModelFromMemory(const void* data, size_t size) {
    if (!data || size == 0) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Invalid data or size");
    }
    
    // 尝试检测模型格式（简化实现，实际应该更智能）
    // 检查ONNX格式的magic number（protobuf格式）
    // ONNX模型通常以protobuf格式存储，可以通过解析来判断
    
    // 首先尝试ONNX格式
    frontend::ONNXParser parser;
    Status status = parser.LoadFromMemory(data, size);
    if (status.IsOk()) {
        std::unique_ptr<Graph> graph;
        status = parser.ConvertToGraph(graph);
        if (status.IsOk()) {
            return LoadModelFromGraph(std::move(graph));
        }
    }
    
    // 其他格式支持（待实现）
    // TensorFlow Lite (.tflite): 需要实现FlatBuffer解析器
    // TensorFlow SavedModel (.pb): 需要实现protobuf解析器
    // 目前只支持ONNX格式
    
    return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED,
                       "Unsupported model format in memory. Only ONNX format is supported.");
}

Status InferenceSession::LoadModelFromGraph(std::unique_ptr<Graph> graph) {
    graph_ = std::move(graph);
    return LoadAndOptimizeGraph();
}

Status InferenceSession::LoadAndOptimizeGraph() {
    if (!graph_) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Graph is null");
    }
    
    // 验证图
    Status status = graph_->Validate();
    if (!status.IsOk()) {
        return status;
    }
    
    // 形状推断（参考ONNX Runtime的ShapeInference）
    status = InferShapes(graph_.get());
    if (!status.IsOk()) {
        // 形状推断失败不影响加载，只记录警告
        LOG_WARNING("Shape inference failed: " + status.Message());
    }
    
    // 优化图 (参考ONNX Runtime的图优化级别)
    if (options_.graph_optimization_level != SessionOptions::GraphOptimizationLevel::NONE) {
        status = optimizer_->Optimize(graph_.get());
        if (!status.IsOk()) {
            return status;
        }
    }
    
    // 分配执行提供者 (参考ONNX Runtime的节点分配)
    std::vector<ExecutionProvider*> provider_ptrs;
    for (const auto& provider : execution_providers_) {
        provider_ptrs.push_back(provider.get());
    }
    status = ExecutionProviderSelector::AssignProviders(graph_.get(), provider_ptrs);
    if (!status.IsOk()) {
        return status;
    }
    
    // 准备执行
    for (const auto& provider : execution_providers_) {
        status = provider->PrepareExecution(graph_.get());
        if (!status.IsOk()) {
            return status;
        }
    }
    
    return Status::Ok();
}

std::vector<Shape> InferenceSession::GetInputShapes() const {
    std::vector<Shape> shapes;
    if (graph_) {
        for (Value* input : graph_->GetInputs()) {
            shapes.push_back(input->GetShape());
        }
    }
    return shapes;
}

std::vector<Shape> InferenceSession::GetOutputShapes() const {
    std::vector<Shape> shapes;
    if (graph_) {
        for (Value* output : graph_->GetOutputs()) {
            shapes.push_back(output->GetShape());
        }
    }
    return shapes;
}

std::vector<std::string> InferenceSession::GetInputNames() const {
    std::vector<std::string> names;
    if (graph_) {
        for (Value* input : graph_->GetInputs()) {
            std::string name = input->GetName();
            if (name.empty()) {
                // 如果没有名称，使用默认名称
                name = "input_" + std::to_string(names.size());
            }
            names.push_back(name);
        }
    }
    return names;
}

std::vector<std::string> InferenceSession::GetOutputNames() const {
    std::vector<std::string> names;
    if (graph_) {
        for (Value* output : graph_->GetOutputs()) {
            std::string name = output->GetName();
            if (name.empty()) {
                // 如果没有名称，使用默认名称
                name = "output_" + std::to_string(names.size());
            }
            names.push_back(name);
        }
    }
    return names;
}

Status InferenceSession::Run(const std::vector<Tensor*>& inputs, std::vector<Tensor*>& outputs) {
    if (!graph_) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Model not loaded");
    }
    
    ExecutionOptions options;
    options.enable_profiling = options_.enable_profiling;
    
    return execution_engine_->Execute(graph_.get(), inputs, outputs, options);
}

Status InferenceSession::Run(const std::unordered_map<std::string, Tensor*>& inputs,
                            std::unordered_map<std::string, Tensor*>& outputs) {
    if (!graph_) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Model not loaded");
    }
    
    // 按名称映射输入
    std::vector<Tensor*> input_tensors;
    std::vector<Value*> graph_inputs = graph_->GetInputs();
    
    for (Value* input_value : graph_inputs) {
        std::string name = input_value->GetName();
        if (name.empty()) {
            // 如果没有名称，使用索引
            size_t index = input_tensors.size();
            name = "input_" + std::to_string(index);
        }
        
        auto it = inputs.find(name);
        if (it != inputs.end()) {
            input_tensors.push_back(it->second);
        } else {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Missing input: " + name);
        }
    }
    
    // 执行推理
    std::vector<Tensor*> output_tensors;
    Status status = Run(input_tensors, output_tensors);
    if (!status.IsOk()) {
        return status;
    }
    
    // 按名称映射输出
    std::vector<Value*> graph_outputs = graph_->GetOutputs();
    for (size_t i = 0; i < output_tensors.size() && i < graph_outputs.size(); ++i) {
        std::string name = graph_outputs[i]->GetName();
        if (name.empty()) {
            name = "output_" + std::to_string(i);
        }
        outputs[name] = output_tensors[i];
    }
    
    return Status::Ok();
}

std::future<Status> InferenceSession::RunAsync(const std::vector<Tensor*>& inputs,
                                             std::vector<Tensor*>& outputs) {
    ExecutionOptions options;
    options.mode = ExecutionMode::ASYNCHRONOUS;
    options.enable_profiling = options_.enable_profiling;
    
    return execution_engine_->ExecuteAsync(graph_.get(), inputs, outputs, options);
}

std::shared_ptr<Tensor> InferenceSession::CreateInputTensor(size_t input_index) {
    if (!graph_ || input_index >= graph_->GetInputs().size()) {
        return nullptr;
    }
    
    Value* input_value = graph_->GetInputs()[input_index];
    Shape shape = input_value->GetShape();
    DataType dtype = input_value->GetDataType();
    
    return CreateTensor(shape, dtype);
}

std::shared_ptr<Tensor> InferenceSession::CreateInputTensor(const std::string& input_name) {
    if (!graph_) {
        return nullptr;
    }
    
    Value* input_value = graph_->FindValueByName(input_name);
    if (!input_value) {
        return nullptr;
    }
    
    Shape shape = input_value->GetShape();
    DataType dtype = input_value->GetDataType();
    
    return CreateTensor(shape, dtype);
}

std::shared_ptr<Tensor> InferenceSession::GetOutputTensor(size_t output_index) {
    if (!graph_ || output_index >= graph_->GetOutputs().size()) {
        return nullptr;
    }
    
    Value* output_value = graph_->GetOutputs()[output_index];
    return output_value->GetTensor();
}

std::shared_ptr<Tensor> InferenceSession::GetOutputTensor(const std::string& output_name) {
    if (!graph_) {
        return nullptr;
    }
    
    Value* output_value = graph_->FindValueByName(output_name);
    if (!output_value) {
        return nullptr;
    }
    
    return output_value->GetTensor();
}

Status InferenceSession::Profile(ProfilingResult& result) {
    if (!graph_) {
        return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                           "Model not loaded");
    }
    
    // 创建虚拟输入
    std::vector<Tensor*> inputs;
    for (Value* input_value : graph_->GetInputs()) {
        auto tensor = CreateTensor(input_value->GetShape(), input_value->GetDataType());
        tensor->FillZero();
        inputs.push_back(tensor.get());
    }
    
    return execution_engine_->Profile(graph_.get(), inputs, result);
}

void InferenceSession::SetOptions(const SessionOptions& options) {
    options_ = options;
    // 重新初始化 (参考ONNX Runtime的配置更新)
    Initialize();
}

} // namespace inferunity

