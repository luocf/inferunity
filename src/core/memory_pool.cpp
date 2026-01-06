// 内存池实现
// 参考 ONNX Runtime 和 TensorFlow 的内存池设计

#include "inferunity/memory.h"
#include "inferunity/logger.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <chrono>

namespace inferunity {

// 内存块
struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
    std::chrono::steady_clock::time_point allocated_time;
    
    MemoryBlock() : ptr(nullptr), size(0), in_use(false),
                    allocated_time(std::chrono::steady_clock::now()) {}
    
    MemoryBlock(void* p, size_t s) 
        : ptr(p), size(s), in_use(true),
          allocated_time(std::chrono::steady_clock::now()) {}
};

// 内存池实现
class MemoryPoolImpl {
private:
    std::mutex mutex_;
    std::unordered_map<void*, MemoryBlock> blocks_;
    size_t total_allocated_ = 0;
    size_t peak_allocated_ = 0;
    size_t current_allocated_ = 0;
    
    // 对齐分配
    void* AllocateAligned(size_t size, size_t alignment) {
        size_t total_size = size + alignment - 1;
        void* raw_ptr = std::malloc(total_size);
        if (!raw_ptr) {
            return nullptr;
        }
        
        // 计算对齐后的地址
        uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
        uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
        void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
        
        // 存储原始指针（用于释放）
        uintptr_t* header = reinterpret_cast<uintptr_t*>(aligned_ptr) - 1;
        *header = reinterpret_cast<uintptr_t>(raw_ptr);
        
        return aligned_ptr;
    }
    
    void FreeAligned(void* ptr) {
        if (!ptr) return;
        
        uintptr_t* header = reinterpret_cast<uintptr_t*>(ptr) - 1;
        void* raw_ptr = reinterpret_cast<void*>(*header);
        std::free(raw_ptr);
    }
    
public:
    void* Allocate(size_t size, size_t alignment = 16) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 查找可重用的内存块
        for (auto& pair : blocks_) {
            if (!pair.second.in_use && pair.second.size >= size) {
                pair.second.in_use = true;
                pair.second.allocated_time = std::chrono::steady_clock::now();
                current_allocated_ += pair.second.size;
                return pair.second.ptr;
            }
        }
        
        // 分配新内存块
        void* ptr = AllocateAligned(size, alignment);
        if (!ptr) {
            LOG_ERROR("Memory allocation failed: size=" + std::to_string(size));
            return nullptr;
        }
        
        blocks_[ptr] = MemoryBlock(ptr, size);
        total_allocated_ += size;
        current_allocated_ += size;
        peak_allocated_ = std::max(peak_allocated_, current_allocated_);
        
        return ptr;
    }
    
    void Free(void* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = blocks_.find(ptr);
        if (it != blocks_.end()) {
            it->second.in_use = false;
            current_allocated_ -= it->second.size;
            // 注意：这里不立即释放内存，而是标记为未使用以便重用
            // 实际释放可以在内存压力大时进行
        }
    }
    
    void ReleaseUnused() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = blocks_.begin();
        while (it != blocks_.end()) {
            if (!it->second.in_use) {
                FreeAligned(it->second.ptr);
                current_allocated_ -= it->second.size;
                it = blocks_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    MemoryStats GetStats() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(mutex_));
        
        MemoryStats stats;
        stats.allocated_bytes = total_allocated_;
        stats.peak_allocated_bytes = peak_allocated_;
        stats.allocation_count = blocks_.size();
        
        size_t unused_count = 0;
        for (const auto& pair : blocks_) {
            if (!pair.second.in_use) {
                unused_count++;
            }
        }
        stats.free_count = unused_count;
        
        return stats;
    }
    
    ~MemoryPoolImpl() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : blocks_) {
            FreeAligned(pair.second.ptr);
        }
        blocks_.clear();
    }
};

// 全局内存池实例
static MemoryPoolImpl* g_memory_pool = nullptr;
static std::once_flag g_pool_init_flag;

MemoryPoolImpl* GetMemoryPool() {
    std::call_once(g_pool_init_flag, []() {
        g_memory_pool = new MemoryPoolImpl();
    });
    return g_memory_pool;
}

} // namespace inferunity

// 公共接口实现
namespace inferunity {

void* AllocateMemory(size_t size, size_t alignment) {
    return GetMemoryPool()->Allocate(size, alignment);
}

void FreeMemory(void* ptr) {
    GetMemoryPool()->Free(ptr);
}

void ReleaseUnusedMemory() {
    GetMemoryPool()->ReleaseUnused();
}

MemoryStats GetMemoryStats() {
    return GetMemoryPool()->GetStats();
}

} // namespace inferunity

