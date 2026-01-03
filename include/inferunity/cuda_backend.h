// CUDA执行提供者
// 参考ONNX Runtime的CUDAExecutionProvider设计

#pragma once

#ifdef ENABLE_CUDA
#include "backend.h"
#include <cuda_runtime.h>
#include <memory>

namespace inferunity {

// CUDA设备实现
class CUDADevice : public Device {
public:
    explicit CUDADevice(int device_id = 0);
    ~CUDADevice();
    
    DeviceType GetType() const override { return DeviceType::CUDA; }
    std::string GetName() const override;
    int GetDeviceId() const override { return device_id_; }
    
    // 内存管理
    void* Allocate(size_t size) override;
    void Free(void* ptr) override;
    void* AllocateAligned(size_t size, size_t alignment) override;
    
    // 数据传输
    Status Copy(void* dst, const void* src, size_t size) override;
    Status CopyFromHost(void* dst, const void* src, size_t size) override;
    Status CopyToHost(void* dst, const void* src, size_t size) override;
    
    // 同步
    Status Synchronize() override;
    
    // 流管理
    void* CreateStream() override;
    void DestroyStream(void* stream) override;
    Status SynchronizeStream(void* stream) override;
    
private:
    int device_id_;
    cudaStream_t default_stream_;
};

// CUDA执行提供者
class CUDAExecutionProvider : public ExecutionProvider {
public:
    explicit CUDAExecutionProvider(int device_id = 0);
    ~CUDAExecutionProvider();
    
    // 提供者信息
    std::string GetName() const override { return "CUDAExecutionProvider"; }
    DeviceType GetDeviceType() const override { return DeviceType::CUDA; }
    bool IsAvailable() const override;
    
    // 设备管理
    std::shared_ptr<Device> GetDevice(int device_id = 0) override;
    int GetDeviceCount() const override;
    
    // 算子支持
    bool SupportsOperator(const std::string& op_type) const override;
    std::unique_ptr<Operator> CreateOperator(const std::string& op_type) override;
    
    // 图优化
    Status OptimizeGraph(Graph* graph) override;
    
    // 编译和准备
    Status CompileNode(Node* node) override;
    Status PrepareExecution(Graph* graph) override;
    
    // 执行节点
    Status ExecuteNode(Node* node, ExecutionContext* ctx) override;
    
private:
    int device_id_;
    std::shared_ptr<CUDADevice> device_;
    
    // 检查CUDA是否可用
    static bool CheckCUDAAvailable();
};

} // namespace inferunity

#endif // ENABLE_CUDA

