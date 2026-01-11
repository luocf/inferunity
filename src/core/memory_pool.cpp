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
    size_t unused_memory_ = 0;  // 未使用内存总量
    size_t max_pool_size_ = 0;  // 最大池大小（0表示无限制）
    double release_threshold_ = 0.5;  // 释放阈值：当未使用内存超过总内存的50%时触发释放
    
    // 对齐分配
    void* AllocateAligned(size_t size, size_t alignment) {
        // 需要额外空间存储原始指针（sizeof(uintptr_t)）和对齐（alignment - 1）
        size_t header_size = sizeof(uintptr_t);
        size_t total_size = size + alignment - 1 + header_size;
        void* raw_ptr = std::malloc(total_size);
        if (!raw_ptr) {
            return nullptr;
        }
        
        // 计算对齐后的地址（在header之后）
        uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
        uintptr_t aligned_addr = (addr + header_size + alignment - 1) & ~(alignment - 1);
        void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
        
        // 在对齐指针之前存储原始指针（用于释放）
        uintptr_t* header = reinterpret_cast<uintptr_t*>(aligned_ptr) - 1;
        *header = reinterpret_cast<uintptr_t>(raw_ptr);
        
        return aligned_ptr;
    }
    
    void FreeAligned(void* ptr) {
        if (!ptr) return;
        
        // 验证指针是否有效（基本检查）
        // 注意：这里不能完全验证，但可以避免明显的错误
        try {
            uintptr_t* header = reinterpret_cast<uintptr_t*>(ptr) - 1;
            void* raw_ptr = reinterpret_cast<void*>(*header);
            
            // 基本验证：raw_ptr应该在合理范围内
            if (raw_ptr && raw_ptr < ptr) {
                std::free(raw_ptr);
            } else {
                LOG_ERROR("Invalid raw pointer in FreeAligned: " + 
                         std::to_string(reinterpret_cast<uintptr_t>(raw_ptr)));
            }
        } catch (...) {
            LOG_ERROR("Exception in FreeAligned for pointer: " + 
                     std::to_string(reinterpret_cast<uintptr_t>(ptr)));
        }
    }
    
public:
    void* Allocate(size_t size, size_t alignment = 16) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 查找可重用的内存块（优先使用大小最接近的块以减少碎片）
        void* best_fit = nullptr;
        size_t best_size = SIZE_MAX;
        
        for (auto& pair : blocks_) {
            if (!pair.second.in_use && pair.second.size >= size) {
                // 优先选择大小最接近的块（最佳适配）
                if (pair.second.size < best_size) {
                    best_fit = pair.second.ptr;
                    best_size = pair.second.size;
                }
            }
        }
        
        if (best_fit) {
            auto it = blocks_.find(best_fit);
            if (it != blocks_.end()) {
                it->second.in_use = true;
                it->second.allocated_time = std::chrono::steady_clock::now();
                unused_memory_ -= it->second.size;
                current_allocated_ += it->second.size;
                return it->second.ptr;
            }
        }
        
        // 检查是否超过最大池大小限制
        if (max_pool_size_ > 0 && total_allocated_ + size > max_pool_size_) {
            // 尝试释放未使用的内存
            ReleaseUnused();
            
            // 如果仍然超过限制，尝试碎片整理
            if (total_allocated_ + size > max_pool_size_) {
                Defragment();
            }
            
            // 再次检查
            if (max_pool_size_ > 0 && total_allocated_ + size > max_pool_size_) {
                LOG_WARNING("Memory pool size limit reached: " + 
                           std::to_string(max_pool_size_) + " bytes");
                // 继续分配，但记录警告
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
            unused_memory_ += it->second.size;
            
            // 检查是否需要自动释放未使用内存
            if (total_allocated_ > 0 && 
                static_cast<double>(unused_memory_) / total_allocated_ > release_threshold_) {
                // 异步释放，避免在Free中执行耗时操作
                // 这里只标记，实际释放由ReleaseUnused执行
            }
        } else {
            // 指针不在blocks_中，可能是无效指针或已经被释放
            // 不执行任何操作，避免崩溃
            LOG_WARNING("Attempted to free unknown pointer: " + 
                       std::to_string(reinterpret_cast<uintptr_t>(ptr)));
        }
    }
    
    void ReleaseUnused() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t released = 0;
        auto it = blocks_.begin();
        while (it != blocks_.end()) {
            if (!it->second.in_use) {
                FreeAligned(it->second.ptr);
                unused_memory_ -= it->second.size;
                total_allocated_ -= it->second.size;
                released += it->second.size;
                it = blocks_.erase(it);
            } else {
                ++it;
            }
        }
        
        if (released > 0) {
            LOG_INFO("Released " + std::to_string(released) + " bytes of unused memory");
        }
    }
    
    // 内存碎片整理：合并相邻的未使用内存块
    // 注意：由于对齐分配的实现，实际内存块可能不连续，这里只做概念性实现
    void Defragment() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 简化实现：只释放长时间未使用的内存块
        // 真正的碎片整理需要更复杂的实现（如buddy allocator）
        auto now = std::chrono::steady_clock::now();
        auto max_age = std::chrono::seconds(60);  // 60秒未使用则释放
        
        size_t merged_count = 0;
        auto it = blocks_.begin();
        while (it != blocks_.end()) {
            if (!it->second.in_use) {
                auto age = now - it->second.allocated_time;
                if (age > max_age) {
                    // 释放长时间未使用的块
                    FreeAligned(it->second.ptr);
                    unused_memory_ -= it->second.size;
                    total_allocated_ -= it->second.size;
                    it = blocks_.erase(it);
                    merged_count++;
                } else {
                    ++it;
                }
            } else {
                ++it;
            }
        }
        
        if (merged_count > 0) {
            LOG_INFO("Defragmented memory: released " + std::to_string(merged_count) + " old blocks");
        }
    }
    
    // 设置最大池大小
    void SetMaxPoolSize(size_t max_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        max_pool_size_ = max_size;
        
        // 如果当前大小超过限制，释放未使用的内存
        if (max_pool_size_ > 0 && total_allocated_ > max_pool_size_) {
            ReleaseUnused();
        }
    }
    
    // 设置释放阈值
    void SetReleaseThreshold(double threshold) {
        std::lock_guard<std::mutex> lock(mutex_);
        release_threshold_ = std::max(0.0, std::min(1.0, threshold));
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
        
        // 计算碎片率（未使用内存占总内存的比例）
        // 注意：这里使用unused_memory_而不是遍历计算，因为可能有合并的块
        
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

void DefragmentMemory() {
    GetMemoryPool()->Defragment();
}

void SetMemoryPoolMaxSize(size_t max_size) {
    GetMemoryPool()->SetMaxPoolSize(max_size);
}

void SetMemoryReleaseThreshold(double threshold) {
    GetMemoryPool()->SetReleaseThreshold(threshold);
}

MemoryStats GetMemoryStats() {
    return GetMemoryPool()->GetStats();
}

} // namespace inferunity

