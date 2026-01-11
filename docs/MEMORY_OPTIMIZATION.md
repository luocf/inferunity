# 内存优化文档

## 概述

InferUnity 实现了多种内存优化策略，包括内存池、碎片整理、自动释放阈值等，以提高内存使用效率和减少碎片。

## 功能特性

### 1. 内存池 ✅

**实现位置**: `src/core/memory_pool.cpp`

**功能**:
- 内存块重用（减少分配/释放开销）
- 对齐内存分配（支持SIMD优化，16字节对齐）
- 内存统计（总分配、峰值、当前使用、重用次数）
- 未使用内存释放

**优化特性**:
- **最佳适配算法**：优先使用大小最接近的内存块，减少碎片
- **智能重用**：查找大小匹配且未使用的内存块
- **对齐支持**：自动处理内存对齐，支持AVX/SSE
- **延迟释放**：标记为未使用而非立即释放，提高重用率

### 2. 内存碎片整理 ✅

**功能**:
- 合并相邻的未使用内存块
- 减少内存碎片
- 提高内存利用率

**API**:
```cpp
// 执行碎片整理
DefragmentMemory();
```

**工作原理**:
1. 按地址排序所有内存块
2. 查找相邻的未使用块
3. 合并相邻块，减少碎片

### 3. 自动释放阈值 ✅

**功能**:
- 当未使用内存超过总内存的指定比例时，自动触发释放
- 可配置阈值（0.0-1.0）

**API**:
```cpp
// 设置释放阈值（例如：0.5表示50%）
SetMemoryReleaseThreshold(0.5);
```

**使用场景**:
- 长时间运行的推理服务
- 内存压力较大的场景
- 需要控制内存占用的应用

### 4. 最大池大小限制 ✅

**功能**:
- 限制内存池的最大大小
- 超过限制时自动释放未使用内存
- 可选的碎片整理

**API**:
```cpp
// 设置最大池大小（字节）
SetMemoryPoolMaxSize(10 * 1024 * 1024);  // 10MB
```

**使用场景**:
- 嵌入式设备
- 内存受限环境
- 需要严格控制内存使用的场景

### 5. Tensor生命周期分析 ✅

**实现位置**: `src/core/tensor_lifetime_optimizer.cpp`

**功能**:
- 分析Tensor的出生时间（产生节点）
- 分析Tensor的死亡时间（最后使用节点）
- 基于生命周期的内存复用分配

**API**:
```cpp
// 分析Tensor生命周期
std::vector<TensorLifetime> lifetimes = AnalyzeTensorLifetimes(graph);

// 基于生命周期进行内存复用分配
Status status = AllocateMemoryWithReuse(graph);
```

## 使用示例

### 基本使用

```cpp
#include "inferunity/memory.h"

// 分配内存（自动对齐）
void* ptr1 = AllocateMemory(1024, 16);  // 1KB，16字节对齐
void* ptr2 = AllocateMemory(2048, 32);  // 2KB，32字节对齐

// 使用内存
// ...

// 释放内存（可重用）
FreeMemory(ptr1);
FreeMemory(ptr2);

// 获取统计信息
MemoryStats stats = GetMemoryStats();
std::cout << "总分配: " << stats.allocated_bytes << " bytes" << std::endl;
std::cout << "峰值: " << stats.peak_allocated_bytes << " bytes" << std::endl;
std::cout << "分配次数: " << stats.allocation_count << std::endl;
std::cout << "未使用块数: " << stats.free_count << std::endl;
```

### 碎片整理

```cpp
// 分配多个内存块
std::vector<void*> ptrs;
for (int i = 0; i < 10; ++i) {
    ptrs.push_back(AllocateMemory(1024, 16));
}

// 释放部分内存块（创建碎片）
for (size_t i = 0; i < ptrs.size(); i += 2) {
    FreeMemory(ptrs[i]);
}

// 执行碎片整理
DefragmentMemory();

// 释放未使用的内存
ReleaseUnusedMemory();
```

### 配置内存池

```cpp
// 设置最大池大小为10MB
SetMemoryPoolMaxSize(10 * 1024 * 1024);

// 设置释放阈值为50%（当未使用内存超过50%时自动释放）
SetMemoryReleaseThreshold(0.5);

// 正常使用
void* ptr = AllocateMemory(1024, 16);
// ...
FreeMemory(ptr);
```

### Tensor生命周期分析

```cpp
#include "inferunity/memory.h"
#include "inferunity/graph.h"

// 分析生命周期
auto lifetimes = AnalyzeTensorLifetimes(graph);

for (const auto& lifetime : lifetimes) {
    std::cout << "Tensor出生: " << lifetime.birth 
              << ", 死亡: " << lifetime.death << std::endl;
}

// 基于生命周期进行内存复用分配
Status status = AllocateMemoryWithReuse(graph);
if (!status.IsOk()) {
    std::cerr << "内存分配失败: " << status.GetMessage() << std::endl;
}
```

## 性能优化

### 内存重用优势

1. **减少分配开销**: 重用已分配的内存块，避免频繁malloc/free
2. **提高缓存命中率**: 重用内存块可能仍在CPU缓存中
3. **降低内存碎片**: 延迟释放，减少内存碎片

### 碎片整理优势

1. **合并相邻块**: 减少碎片，提高内存利用率
2. **提高重用率**: 合并后的块更容易被重用
3. **降低内存占用**: 减少总内存占用

### 最佳适配算法优势

1. **减少碎片**: 优先使用大小最接近的块
2. **提高利用率**: 避免大块被小请求占用
3. **降低浪费**: 最小化内存浪费

## 配置建议

### 对齐大小

```cpp
// 默认16字节对齐（支持AVX）
size_t alignment = 16;

// 对于AVX-512，使用32字节对齐
size_t alignment = 32;

// 分配内存
void* ptr = AllocateMemory(size, alignment);
```

### 释放阈值

```cpp
// 保守策略：30%（更频繁释放，内存占用更小）
SetMemoryReleaseThreshold(0.3);

// 平衡策略：50%（默认）
SetMemoryReleaseThreshold(0.5);

// 激进策略：70%（更少释放，重用率更高）
SetMemoryReleaseThreshold(0.7);
```

### 最大池大小

```cpp
// 嵌入式设备：1MB
SetMemoryPoolMaxSize(1 * 1024 * 1024);

// 桌面应用：100MB
SetMemoryPoolMaxSize(100 * 1024 * 1024);

// 服务器：1GB
SetMemoryPoolMaxSize(1024 * 1024 * 1024);

// 无限制（默认）
SetMemoryPoolMaxSize(0);
```

## 性能监控

### 内存统计

```cpp
MemoryStats stats = GetMemoryStats();

std::cout << "总分配: " << stats.allocated_bytes << " bytes" << std::endl;
std::cout << "峰值: " << stats.peak_allocated_bytes << " bytes" << std::endl;
std::cout << "分配次数: " << stats.allocation_count << std::endl;
std::cout << "未使用块数: " << stats.free_count << std::endl;

// 计算碎片率
double fragmentation = stats.free_count > 0 ? 
    (static_cast<double>(stats.free_count) / stats.allocation_count) : 0.0;
std::cout << "碎片率: " << (fragmentation * 100) << "%" << std::endl;
```

## 最佳实践

1. **定期释放未使用内存**: 在长时间运行的推理服务中，定期调用`ReleaseUnusedMemory()`
2. **使用碎片整理**: 在内存压力大时，调用`DefragmentMemory()`
3. **设置合理的阈值**: 根据应用场景设置释放阈值和最大池大小
4. **监控内存使用**: 定期检查内存统计，优化配置
5. **利用生命周期分析**: 对于复杂模型，使用生命周期分析进行内存复用

## 更新日志

- 2026-01-06: 实现内存碎片整理功能
- 2026-01-06: 添加自动释放阈值机制
- 2026-01-06: 实现最大池大小限制
- 2026-01-06: 优化最佳适配算法

