#include "inferunity/memory.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace inferunity {

// CPU分配器实现
// 使用内存池进行分配，以便统计和管理
void* CPUAllocator::Allocate(size_t size) {
    // 使用内存池分配（带对齐）
    return AllocateMemory(size, 16);  // 16字节对齐
}

void CPUAllocator::Free(void* ptr) {
    // 使用内存池释放
    FreeMemory(ptr);
}

size_t CPUAllocator::GetAllocatedSize(void* ptr) const {
    // 标准malloc不提供此功能，需要自定义实现
    (void)ptr;
    return 0;
}

// 内存池实现
MemoryPool::MemoryPool(size_t block_size, size_t initial_blocks)
    : block_size_(block_size) {
    blocks_.reserve(initial_blocks);
    for (size_t i = 0; i < initial_blocks; ++i) {
        AllocateNewBlock();
    }
}

MemoryPool::~MemoryPool() {
    Reset();
    for (auto& block : blocks_) {
        if (block.data) {
            std::free(block.data);
        }
    }
}

void* MemoryPool::Allocate(size_t size) {
    if (size > block_size_) {
        // 大块直接分配
        return std::malloc(size);
    }
    
    // 从空闲块中查找
    if (!free_blocks_.empty()) {
        void* ptr = free_blocks_.back();
        free_blocks_.pop_back();
        return ptr;
    }
    
    // 分配新块
    return AllocateNewBlock();
}

void MemoryPool::Free(void* ptr) {
    if (!ptr) return;
    
    // 检查是否是大块
    bool found = false;
    for (auto& block : blocks_) {
        if (block.data == ptr) {
            if (block.in_use) {
                block.in_use = false;
                free_blocks_.push_back(ptr);
                found = true;
            }
            break;
        }
    }
    
    if (!found) {
        // 可能是大块，直接释放
        std::free(ptr);
    }
}

void MemoryPool::Reset() {
    for (auto& block : blocks_) {
        block.in_use = false;
    }
    free_blocks_.clear();
    for (auto& block : blocks_) {
        if (block.data) {
            free_blocks_.push_back(block.data);
        }
    }
}

void* MemoryPool::AllocateNewBlock() {
    void* data = std::malloc(block_size_);
    if (data) {
        blocks_.push_back({data, true});
    }
    return data;
}

// 全局分配器管理
namespace {
    std::mutex allocator_mutex;
    std::unordered_map<DeviceType, std::shared_ptr<MemoryAllocator>> allocators;
    
    std::shared_ptr<MemoryAllocator> GetOrCreateAllocator(DeviceType device) {
        std::lock_guard<std::mutex> lock(allocator_mutex);
        auto it = allocators.find(device);
        if (it != allocators.end()) {
            return it->second;
        }
        
        std::shared_ptr<MemoryAllocator> allocator;
        switch (device) {
            case DeviceType::CPU:
                allocator = std::make_shared<CPUAllocator>();
                break;
            // 其他设备的分配器（CUDA等）
            // 可以通过ExecutionProviderRegistry获取相应设备的分配器
            // 目前只实现CPU分配器，其他设备分配器在相应后端中实现
            default:
                allocator = std::make_shared<CPUAllocator>();
                break;
        }
        
        allocators[device] = allocator;
        return allocator;
    }
}

std::shared_ptr<MemoryAllocator> GetMemoryAllocator(DeviceType device) {
    return GetOrCreateAllocator(device);
}

// 内存统计（线程安全）
namespace {
    std::mutex stats_mutex;
    std::unordered_map<DeviceType, MemoryStats> device_stats;
    
    void UpdateStats(DeviceType device, size_t allocated, bool is_alloc) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        MemoryStats& stats = device_stats[device];
        if (is_alloc) {
            stats.allocated_bytes += allocated;
            stats.allocation_count++;
            if (stats.allocated_bytes > stats.peak_allocated_bytes) {
                stats.peak_allocated_bytes = stats.allocated_bytes;
            }
        } else {
            stats.allocated_bytes -= allocated;
            stats.free_count++;
        }
    }
}

MemoryStats GetMemoryStats(DeviceType device) {
    // 对于CPU设备，使用内存池统计
    if (device == DeviceType::CPU) {
        return ::inferunity::GetMemoryStats();
    }
    // 其他设备使用设备特定统计
    std::lock_guard<std::mutex> lock(stats_mutex);
    auto it = device_stats.find(device);
    if (it != device_stats.end()) {
        return it->second;
    }
    return MemoryStats{0, 0, 0, 0};
}

// 内存碎片整理功能（简化实现）
// 注意：由于MemoryPool的blocks_是private，这里提供概念性实现
// 实际使用时需要在MemoryPool类中添加Defragment()方法

} // namespace inferunity

