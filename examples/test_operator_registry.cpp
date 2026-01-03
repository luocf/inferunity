// 测试算子注册表
#include "inferunity/operator.h"
#include "inferunity/backend.h"
#include <iostream>
#include <vector>

using namespace inferunity;

int main() {
    std::cout << "=== 测试算子注册表 ===" << std::endl;
    
    // 获取算子注册表实例
    auto& registry = OperatorRegistry::Instance();
    
    // 获取所有已注册的算子
    auto registered_ops = registry.GetRegisteredOps();
    std::cout << "\n已注册的算子数量: " << registered_ops.size() << std::endl;
    std::cout << "已注册的算子列表:" << std::endl;
    for (const auto& op : registered_ops) {
        std::cout << "  - " << op << std::endl;
    }
    
    // 测试Add算子
    std::cout << "\n测试Add算子:" << std::endl;
    bool is_registered = registry.IsRegistered("Add");
    std::cout << "  IsRegistered('Add'): " << (is_registered ? "是" : "否") << std::endl;
    
    auto add_op = registry.Create("Add");
    if (add_op) {
        std::cout << "  ✓ Create('Add')成功" << std::endl;
        std::cout << "  算子名称: " << add_op->GetName() << std::endl;
    } else {
        std::cout << "  ✗ Create('Add')失败" << std::endl;
    }
    
    // 测试其他算子
    std::vector<std::string> test_ops = {"Mul", "Conv", "Relu", "MatMul"};
    std::cout << "\n测试其他算子:" << std::endl;
    for (const auto& op_name : test_ops) {
        bool reg = registry.IsRegistered(op_name);
        auto op = registry.Create(op_name);
        std::cout << "  " << op_name << ": " 
                  << (reg ? "已注册" : "未注册") << ", "
                  << (op ? "可创建" : "不可创建") << std::endl;
    }
    
    // 测试执行提供者
    std::cout << "\n测试执行提供者:" << std::endl;
    InitializeExecutionProviders();
    auto& ep_registry = ExecutionProviderRegistry::Instance();
    auto cpu_provider = ep_registry.Create("CPUExecutionProvider");
    if (cpu_provider) {
        std::cout << "  ✓ CPUExecutionProvider创建成功" << std::endl;
        std::cout << "  支持Add算子: " << (cpu_provider->SupportsOperator("Add") ? "是" : "否") << std::endl;
        
        auto add_op_from_provider = cpu_provider->CreateOperator("Add");
        if (add_op_from_provider) {
            std::cout << "  ✓ 从执行提供者创建Add算子成功" << std::endl;
        } else {
            std::cout << "  ✗ 从执行提供者创建Add算子失败" << std::endl;
        }
    } else {
        std::cout << "  ✗ CPUExecutionProvider创建失败" << std::endl;
    }
    
    return 0;
}

