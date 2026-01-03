#include "inferunity/backend.h"
#include "inferunity/operator.h"
#include "inferunity/memory.h"
#include "inferunity/graph.h"
#include "inferunity/tensor.h"
#include <thread>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

// 前向声明形状推断函数
namespace inferunity {
    Status InferShapes(Graph* graph);
}

namespace inferunity {

// CPU设备实现
class CPUDevice : public Device {
public:
    DeviceType GetType() const override { return DeviceType::CPU; }
    std::string GetName() const override { return "CPU"; }
    
    void* Allocate(size_t size) override {
        return std::malloc(size);
    }
    
    void Free(void* ptr) override {
        std::free(ptr);
    }
    
    void* AllocateAligned(size_t size, size_t alignment) override {
        void* ptr = nullptr;
        #ifdef _WIN32
            ptr = _aligned_malloc(size, alignment);
        #elif defined(__APPLE__)
            // macOS使用posix_memalign
            if (posix_memalign(&ptr, alignment, size) != 0) {
                return nullptr;
            }
        #else
            // Linux使用aligned_alloc或posix_memalign
            #if __STDC_VERSION__ >= 201112L
                ptr = aligned_alloc(alignment, size);
            #else
                if (posix_memalign(&ptr, alignment, size) != 0) {
                    return nullptr;
                }
            #endif
        #endif
        return ptr;
    }
    
    Status Copy(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        return Status::Ok();
    }
    
    Status CopyFromHost(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        return Status::Ok();
    }
    
    Status CopyToHost(void* dst, const void* src, size_t size) override {
        std::memcpy(dst, src, size);
        return Status::Ok();
    }
    
    Status Synchronize() override {
        return Status::Ok();
    }
    
    void* CreateStream() override {
        return nullptr;  // CPU不需要流
    }
    
    void DestroyStream(void* stream) override {
        (void)stream;
    }
    
    Status SynchronizeStream(void* stream) override {
        (void)stream;
        return Status::Ok();
    }
};

// CPU执行提供者实现 (参考ONNX Runtime的CPUExecutionProvider)
class CPUExecutionProvider : public ExecutionProvider {
public:
    std::string GetName() const override { return "CPU"; }
    DeviceType GetDeviceType() const override { return DeviceType::CPU; }
    
    bool IsAvailable() const override {
        return true;  // CPU总是可用
    }
    
    std::shared_ptr<Device> GetDevice(int device_id = 0) override {
        (void)device_id;
        if (!device_) {
            device_ = std::make_shared<CPUDevice>();
        }
        return device_;
    }
    
    int GetDeviceCount() const override {
        return std::thread::hardware_concurrency();
    }
    
    bool SupportsOperator(const std::string& op_type) const override {
        // CPU支持的基础算子列表（参考ONNX Runtime的CPU算子支持）
        // 维护支持列表：包含所有已实现的算子
        static const std::unordered_set<std::string> supported_operators = {
            // 基础算子
            "Conv", "Relu", "Sigmoid", "Tanh", "Gelu", "GELU", "Silu", "SiLU", "Swish",
            "MatMul", "Add", "Mul", "Sub",
            "MaxPool", "AvgPool", "AveragePool", "GlobalMaxPool", "GlobalAvgPool",
            "BatchNormalization", "LayerNormalization", "RMSNorm",
            "Softmax", "LogSoftmax",
            // 形状操作
            "Reshape", "Concat", "Split", "Transpose", "Gather", "Slice",
            // Transformer专用
            "Embedding",
            // 融合算子
            "FusedConvBNReLU", "FusedMatMulAdd", "FusedConvReLU", "FusedBNReLU",
            // 其他常用算子
            "Dropout", "Flatten", "Pad", "Resize"
        };
        
        // 检查是否在支持列表中
        if (supported_operators.find(op_type) != supported_operators.end()) {
            return true;
        }
        
        // 也检查算子注册表（支持动态注册的算子）
        return OperatorRegistry::Instance().IsRegistered(op_type);
    }
    
    std::unique_ptr<Operator> CreateOperator(const std::string& op_type) override {
        // 从全局算子注册表创建算子
        // 算子实现在src/operators中，通过REGISTER_OPERATOR宏自动注册
        return OperatorRegistry::Instance().Create(op_type);
    }
    
    Status OptimizeGraph(Graph* graph) override {
        // CPU特定的图优化（参考ONNX Runtime的CPU优化策略）
        if (!graph) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
        }
        
        // 1. 形状推断（确保所有节点都有正确的形状信息）
        Status shape_status = InferShapes(graph);
        if (!shape_status.IsOk()) {
            // 形状推断失败不是致命错误，继续优化
        }
        
        // 2. 验证图结构
        Status validate_status = graph->Validate();
        if (!validate_status.IsOk()) {
            return validate_status;
        }
        
        // 3. CPU特定的优化：
        // - 确保所有节点都分配到CPU设备
        for (const auto& node : graph->GetNodes()) {
            node->SetDevice(DeviceType::CPU);
        }
        
        // 4. 内存布局优化（已在优化器中实现，这里可以再次应用）
        // 注意：实际优化应该在Optimizer中完成，这里只是确保设备分配
        
        return Status::Ok();
    }
    
    Status CompileNode(Node* node) override {
        // CPU节点编译（参考ONNX Runtime的节点编译）
        // CPU后端通常不需要JIT编译，但可以：
        // 1. 验证算子是否支持
        // 2. 预分配内存
        // 3. 选择最优的内核实现
        
        if (!node) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Node is null");
        }
        
        // 检查算子是否支持
        if (!SupportsOperator(node->GetOpType())) {
            return Status::Error(StatusCode::ERROR_NOT_FOUND,
                               "Operator not supported: " + node->GetOpType());
        }
        
        // 验证节点输入输出
        if (node->GetInputs().empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Node has no inputs");
        }
        
        // CPU后端不需要特殊的编译步骤
        // 实际的内核选择在ExecuteNode时动态决定
        
        return Status::Ok();
    }
    
    Status PrepareExecution(Graph* graph) override {
        // 准备执行（参考ONNX Runtime的执行准备）
        // 1. 分配内存
        // 2. 编译节点
        // 3. 预计算常量
        
        if (!graph) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Graph is null");
        }
        
        // 1. 形状推断（应该在图优化阶段完成）
        // 这里只验证形状是否已推断
        
        // 2. 为所有输出Value预分配Tensor（可选优化）
        // 这样可以避免在执行时动态分配
        for (Value* output : graph->GetOutputs()) {
            if (!output->GetTensor()) {
                const Shape& shape = output->GetShape();
                if (shape.GetElementCount() > 0) {
                    DataType dtype = output->GetDataType();
                    if (dtype == DataType::UNKNOWN) {
                        dtype = DataType::FLOAT32;  // 默认类型
                    }
                    auto tensor = CreateTensor(shape, dtype, DeviceType::CPU);
                    output->SetTensor(tensor);
                }
            }
        }
        
        // 3. 编译所有节点（验证和准备）
        for (const auto& node : graph->GetNodes()) {
            Status compile_status = CompileNode(node.get());
            if (!compile_status.IsOk()) {
                // 编译失败，记录但继续
                // 实际执行时可能会失败
            }
        }
        
        // 4. 内存预分配（可选，由内存管理器处理）
        // 这里可以触发内存生命周期分析
        
        return Status::Ok();
    }
    
    Status ExecuteNode(Node* node, ExecutionContext* ctx) override {
        // 执行节点
        auto op = CreateOperator(node->GetOpType());
        if (!op) {
            return Status::Error(StatusCode::ERROR_NOT_FOUND,
                               "Operator not found: " + node->GetOpType());
        }
        
        // 将Node的属性复制到Operator（参考ONNX Runtime的实现）
        for (const auto& attr : node->GetAttributes()) {
            // 将字符串属性转换为AttributeValue
            // 简化实现：尝试解析为int或float，否则作为字符串
            AttributeValue attr_value;
            try {
                // 尝试解析为整数
                int64_t int_val = std::stoll(attr.second);
                attr_value = AttributeValue(int_val);
            } catch (...) {
                try {
                    // 尝试解析为浮点数
                    float float_val = std::stof(attr.second);
                    attr_value = AttributeValue(float_val);
                } catch (...) {
                    // 作为字符串
                    attr_value = AttributeValue(attr.second);
                }
            }
            op->SetAttribute(attr.first, attr_value);
        }
        
        // 准备输入输出
        std::vector<Tensor*> inputs;
        std::vector<Tensor*> outputs;
        
        for (Value* input : node->GetInputs()) {
            if (input->GetTensor()) {
                inputs.push_back(input->GetTensor().get());
            }
        }
        
        for (Value* output : node->GetOutputs()) {
            if (output->GetTensor()) {
                outputs.push_back(output->GetTensor().get());
            }
        }
        
        // 执行
        return op->Execute(inputs, outputs, ctx);
    }
    
private:
    std::shared_ptr<Device> device_;
};

// 注册CPU执行提供者 (参考ONNX Runtime的注册方式)
// 使用函数来确保注册代码被执行（避免静态变量被优化）
namespace {
    void RegisterCPUExecutionProvider() {
        ExecutionProviderRegistry::Instance().Register("CPUExecutionProvider", []() {
            return std::make_unique<CPUExecutionProvider>();
        });
        // 同时注册"CPU"别名以保持兼容性
        ExecutionProviderRegistry::Instance().Register("CPU", []() {
            return std::make_unique<CPUExecutionProvider>();
        });
    }
    
    // 使用lambda立即执行来确保注册代码被执行（避免静态变量被优化）
    // 这个lambda会在文件加载时立即执行
    [[maybe_unused]] static bool g_registered = []() {
        RegisterCPUExecutionProvider();
        return true;
    }();
}

// 实现InitializeExecutionProviders函数
void InitializeExecutionProviders() {
    // 调用注册函数，确保CPU执行提供者被注册
    RegisterCPUExecutionProvider();
}

} // namespace inferunity

