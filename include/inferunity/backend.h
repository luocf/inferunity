#pragma once

#include "types.h"
#include "tensor.h"
#include "operator.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>

namespace inferunity {

// 前向声明
class Graph;
class Node;

// 设备接口
class Device {
public:
    virtual ~Device() = default;
    
    virtual DeviceType GetType() const = 0;
    virtual std::string GetName() const = 0;
    virtual int GetDeviceId() const { return 0; }
    
    // 内存管理
    virtual void* Allocate(size_t size) = 0;
    virtual void Free(void* ptr) = 0;
    virtual void* AllocateAligned(size_t size, size_t alignment) = 0;
    
    // 数据传输
    virtual Status Copy(void* dst, const void* src, size_t size) = 0;
    virtual Status CopyFromHost(void* dst, const void* src, size_t size) = 0;
    virtual Status CopyToHost(void* dst, const void* src, size_t size) = 0;
    
    // 同步
    virtual Status Synchronize() = 0;
    
    // 流管理
    virtual void* CreateStream() = 0;
    virtual void DestroyStream(void* stream) = 0;
    virtual Status SynchronizeStream(void* stream) = 0;
};

// 执行提供者接口 (参考ONNX Runtime的ExecutionProvider设计)
// ExecutionProvider是ONNX Runtime的核心抽象，我们采用相同的命名和设计理念
class ExecutionProvider {
public:
    virtual ~ExecutionProvider() = default;
    
    // 提供者信息
    virtual std::string GetName() const = 0;
    virtual DeviceType GetDeviceType() const = 0;
    virtual bool IsAvailable() const = 0;
    
    // 设备管理
    virtual std::shared_ptr<Device> GetDevice(int device_id = 0) = 0;
    virtual int GetDeviceCount() const = 0;
    
    // 算子支持 (参考TensorFlow Lite的算子注册机制)
    virtual bool SupportsOperator(const std::string& op_type) const = 0;
    virtual std::unique_ptr<Operator> CreateOperator(const std::string& op_type) = 0;
    
    // 图优化 (参考TVM的图优化Pass)
    virtual Status OptimizeGraph(Graph* graph) = 0;
    
    // 编译和准备 (参考TVM的编译流程)
    virtual Status CompileNode(Node* node) = 0;
    virtual Status PrepareExecution(Graph* graph) = 0;
    
    // 执行节点
    virtual Status ExecuteNode(Node* node, ExecutionContext* ctx) = 0;
    
    // 量化支持
    virtual bool SupportsQuantization() const { return false; }
    virtual Status QuantizeModel(Graph* graph, DataType target_dtype) {
        (void)graph; (void)target_dtype;
        return Status::Error(StatusCode::ERROR_NOT_IMPLEMENTED);
    }
};

// 为了向后兼容，保留Backend作为别名
using Backend = ExecutionProvider;

// 执行提供者注册表 (参考ONNX Runtime的ExecutionProvider注册机制)
class ExecutionProviderRegistry {
public:
    using ProviderFactory = std::function<std::unique_ptr<ExecutionProvider>()>;
    
    static ExecutionProviderRegistry& Instance() {
        static ExecutionProviderRegistry instance;
        return instance;
    }
    
    // 注册提供者 (参考ONNX Runtime的注册方式)
    void Register(const std::string& name, ProviderFactory factory) {
        factories_[name] = factory;
    }
    
    // 创建提供者
    std::unique_ptr<ExecutionProvider> Create(const std::string& name) const {
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }
    
    // 获取可用的提供者列表
    std::vector<std::string> GetAvailableProviders() const {
        std::vector<std::string> providers;
        for (const auto& pair : factories_) {
            auto provider = pair.second();
            if (provider && provider->IsAvailable()) {
                providers.push_back(pair.first);
            }
        }
        return providers;
    }
    
    // 获取所有已注册的提供者名称（不检查可用性）
    std::vector<std::string> GetRegisteredProviders() const {
        std::vector<std::string> providers;
        for (const auto& pair : factories_) {
            providers.push_back(pair.first);
        }
        return providers;
    }
    
private:
    std::unordered_map<std::string, ProviderFactory> factories_;
};

// 为了向后兼容，保留BackendRegistry作为别名
using BackendRegistry = ExecutionProviderRegistry;

// 提供者选择器 (参考ONNX Runtime的节点分配策略)
class ExecutionProviderSelector {
public:
    // 为节点选择最佳执行提供者 (参考ONNX Runtime的节点分配算法)
    static ExecutionProvider* SelectProvider(Node* node, 
                                             const std::vector<ExecutionProvider*>& available_providers) {
        if (available_providers.empty()) {
            return nullptr;
        }
        
        // 策略1: 选择第一个支持该算子的提供者 (参考ONNX Runtime的默认策略)
        for (ExecutionProvider* provider : available_providers) {
            if (provider->SupportsOperator(node->GetOpType())) {
                return provider;
            }
        }
        
        // 策略2: 回退到CPU提供者 (参考ONNX Runtime的fallback机制)
        for (ExecutionProvider* provider : available_providers) {
            if (provider->GetDeviceType() == DeviceType::CPU) {
                return provider;
            }
        }
        
        // 策略3: 使用第一个可用提供者
        return available_providers[0];
    }
    
    // 为整个图分配执行提供者 (参考ONNX Runtime的图分区)
    static Status AssignProviders(Graph* graph,
                                  const std::vector<ExecutionProvider*>& available_providers) {
        for (const auto& node_ptr : graph->GetNodes()) {
            Node* node = node_ptr.get();
            ExecutionProvider* provider = SelectProvider(node, available_providers);
            if (provider) {
                node->SetDevice(provider->GetDeviceType());
            } else {
                return Status::Error(StatusCode::ERROR_NOT_FOUND,
                                   "No execution provider available for operator: " + node->GetOpType());
            }
        }
        return Status::Ok();
    }
};

// 为了向后兼容，保留BackendSelector作为别名
using BackendSelector = ExecutionProviderSelector;

// 显式初始化所有执行提供者（确保注册代码被执行）
void InitializeExecutionProviders();

} // namespace inferunity

