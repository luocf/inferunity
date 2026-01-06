// 测试CPU后端功能

#include "inferunity/engine.h"
#include "inferunity/backend.h"
#include "inferunity/tensor.h"
#include "inferunity/memory.h"
#include "inferunity/runtime.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace inferunity;

int main() {
    std::cout << "=== CPU后端功能测试 ===" << std::endl;
    
    // 1. 测试执行提供者注册
    std::cout << "\n1. 测试执行提供者注册..." << std::endl;
    InitializeExecutionProviders();
    auto& registry = ExecutionProviderRegistry::Instance();
    auto providers = registry.GetRegisteredProviders();
    std::cout << "  已注册的提供者数量: " << providers.size() << std::endl;
    for (const auto& name : providers) {
        std::cout << "    - " << name << std::endl;
    }
    
    // 2. 测试CPU执行提供者
    std::cout << "\n2. 测试CPU执行提供者..." << std::endl;
    auto cpu_provider = registry.Create("CPU");
    if (cpu_provider) {
        std::cout << "  ✓ CPU执行提供者创建成功" << std::endl;
        std::cout << "  名称: " << cpu_provider->GetName() << std::endl;
        std::cout << "  设备类型: " << (int)cpu_provider->GetDeviceType() << std::endl;
        std::cout << "  是否可用: " << (cpu_provider->IsAvailable() ? "是" : "否") << std::endl;
        std::cout << "  设备数量: " << cpu_provider->GetDeviceCount() << std::endl;
        
        // 测试算子支持
        std::cout << "\n  支持的算子测试:" << std::endl;
        std::vector<std::string> test_ops = {"Add", "Mul", "Conv", "Relu", "MatMul"};
        for (const auto& op : test_ops) {
            bool supported = cpu_provider->SupportsOperator(op);
            std::cout << "    " << op << ": " << (supported ? "✓" : "✗") << std::endl;
        }
    } else {
        std::cout << "  ✗ CPU执行提供者创建失败" << std::endl;
        return 1;
    }
    
    // 3. 测试内存管理
    std::cout << "\n3. 测试内存管理..." << std::endl;
    MemoryStats stats = GetMemoryStats(DeviceType::CPU);
    std::cout << "  总分配: " << stats.allocated_bytes << " bytes" << std::endl;
    std::cout << "  峰值: " << stats.peak_allocated_bytes << " bytes" << std::endl;
    std::cout << "  分配次数: " << stats.allocation_count << std::endl;
    std::cout << "  释放次数: " << stats.free_count << std::endl;
    
    // 4. 测试线程池
    std::cout << "\n4. 测试线程池..." << std::endl;
    std::cout << "  线程数: " << ThreadPool::GetThreadCount() << std::endl;
    std::cout << "  待处理任务: " << ThreadPool::GetPendingTaskCount() << std::endl;
    
    // 提交一些测试任务
    std::atomic<int> counter{0};
    for (int i = 0; i < 10; ++i) {
        ThreadPool::EnqueueTask([&counter, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter++;
            std::cout << "    任务 " << i << " 完成" << std::endl;
        });
    }
    
    ThreadPool::WaitAll();
    std::cout << "  所有任务完成，计数器: " << counter.load() << std::endl;
    
    // 5. 测试InferenceSession
    std::cout << "\n5. 测试InferenceSession..." << std::endl;
    SessionOptions options;
    options.execution_providers = {"CPU"};
    auto session = InferenceSession::Create(options);
    if (session) {
        std::cout << "  ✓ InferenceSession创建成功" << std::endl;
    } else {
        std::cout << "  ✗ InferenceSession创建失败" << std::endl;
        return 1;
    }
    
    std::cout << "\n✅ CPU后端功能测试完成！" << std::endl;
    return 0;
}

