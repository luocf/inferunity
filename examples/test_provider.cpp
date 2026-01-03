// 测试执行提供者注册
#include "inferunity/backend.h"
#include <iostream>

using namespace inferunity;

int main() {
    std::cout << "=== 测试执行提供者注册 ===" << std::endl;
    
    // 显式初始化所有执行提供者（确保注册代码被执行）
    InitializeExecutionProviders();
    
    auto& registry = ExecutionProviderRegistry::Instance();
    
    // 先检查已注册的提供者（不检查可用性）
    auto registered = registry.GetRegisteredProviders();
    std::cout << "\n已注册的提供者数量: " << registered.size() << std::endl;
    for (const auto& name : registered) {
        std::cout << "  - " << name << std::endl;
    }
    
    // 获取可用提供者
    auto providers = registry.GetAvailableProviders();
    std::cout << "\n可用执行提供者数量: " << providers.size() << std::endl;
    
    for (const auto& name : providers) {
        std::cout << "  - " << name << std::endl;
        
        // 尝试创建提供者
        auto provider = registry.Create(name);
        if (provider) {
            std::cout << "    ✓ 创建成功" << std::endl;
            std::cout << "    设备类型: " << static_cast<int>(provider->GetDeviceType()) << std::endl;
            std::cout << "    是否可用: " << (provider->IsAvailable() ? "是" : "否") << std::endl;
        } else {
            std::cout << "    ✗ 创建失败" << std::endl;
        }
    }
    
    // 测试创建CPUExecutionProvider
    std::cout << "\n测试创建CPUExecutionProvider..." << std::endl;
    auto cpu_provider = registry.Create("CPUExecutionProvider");
    if (cpu_provider) {
        std::cout << "  ✓ CPUExecutionProvider创建成功" << std::endl;
    } else {
        std::cout << "  ✗ CPUExecutionProvider创建失败" << std::endl;
    }
    
    // 测试创建CPU别名
    std::cout << "\n测试创建CPU别名..." << std::endl;
    auto cpu_alias = registry.Create("CPU");
    if (cpu_alias) {
        std::cout << "  ✓ CPU别名创建成功" << std::endl;
    } else {
        std::cout << "  ✗ CPU别名创建失败" << std::endl;
    }
    
    return 0;
}

