#pragma once

#include "types.h"
#include <cstddef>
#include <memory>

namespace inferunity {

// 前向声明
class Graph;

// 内存分配器接口
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;
    
    // 分配内存
    virtual void* Allocate(size_t size) = 0;
    
    // 释放内存
    virtual void Free(void* ptr) = 0;
    
    // 对齐分配
    virtual void* AllocateAligned(size_t size, size_t alignment) {
        // 默认实现：分配额外空间用于对齐
        size_t total_size = size + alignment - 1;
        void* ptr = Allocate(total_size);
        if (ptr) {
            uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
            uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
            return reinterpret_cast<void*>(aligned_addr);
        }
        return nullptr;
    }
    
    // 获取分配的内存大小（可选）
    virtual size_t GetAllocatedSize(void* ptr) const { return 0; }
};

// CPU内存分配器
class CPUAllocator : public MemoryAllocator {
public:
    void* Allocate(size_t size) override;
    void Free(void* ptr) override;
    size_t GetAllocatedSize(void* ptr) const override;
};

// 内存池分配器（用于减少分配开销）
class MemoryPool {
public:
    explicit MemoryPool(size_t block_size, size_t initial_blocks = 16);
    ~MemoryPool();
    
    void* Allocate(size_t size);
    void Free(void* ptr);
    void Reset();  // 释放所有块，但保留池
    
private:
    struct Block {
        void* data;
        bool in_use;
    };
    
    size_t block_size_;
    std::vector<Block> blocks_;
    std::vector<void*> free_blocks_;
    
    void* AllocateNewBlock();
};

// 获取设备对应的内存分配器
std::shared_ptr<MemoryAllocator> GetMemoryAllocator(DeviceType device);

// 内存统计
struct MemoryStats {
    size_t allocated_bytes;
    size_t peak_allocated_bytes;
    size_t allocation_count;
    size_t free_count;
};

MemoryStats GetMemoryStats(DeviceType device);

// 张量生命周期分析（参考NCNN的BlobAllocator）
struct TensorLifetime {
    int64_t birth;   // 出生时间（节点执行顺序）
    int64_t death;   // 死亡时间（最后使用时间）
    void* value_ptr;  // Value指针（避免循环依赖）
};

std::vector<TensorLifetime> AnalyzeTensorLifetimes(const Graph* graph);

// 内存复用分配（参考NCNN的内存复用机制）
Status AllocateMemoryWithReuse(Graph* graph);

// 内存池公共接口（新增）
void* AllocateMemory(size_t size, size_t alignment = 16);
void FreeMemory(void* ptr);
void ReleaseUnusedMemory();
MemoryStats GetMemoryStats();  // CPU设备的内存统计

} // namespace inferunity

