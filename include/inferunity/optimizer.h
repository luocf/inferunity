#pragma once

#include "types.h"
#include "graph.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace inferunity {

// 优化Pass接口
class OptimizationPass {
public:
    virtual ~OptimizationPass() = default;
    
    virtual std::string GetName() const = 0;
    virtual Status Run(Graph* graph) = 0;
    
    // Pass依赖关系
    virtual std::vector<std::string> GetDependencies() const { return {}; }
    
    // 是否可重复运行
    virtual bool IsRepeatable() const { return false; }
};

// 优化器管理器
class Optimizer {
public:
    Optimizer();
    ~Optimizer();
    
    // 注册Pass
    void RegisterPass(std::unique_ptr<OptimizationPass> pass);
    
    // 运行所有Pass
    Status Optimize(Graph* graph);
    
    // 运行特定Pass
    Status RunPass(const std::string& pass_name, Graph* graph);
    
    // 获取已注册的Pass列表
    std::vector<std::string> GetRegisteredPasses() const;
    
private:
    std::vector<std::unique_ptr<OptimizationPass>> passes_;
    std::unordered_map<std::string, OptimizationPass*> pass_map_;
    
    // 按依赖关系排序Pass
    std::vector<OptimizationPass*> SortPasses() const;
};

// 常见优化Pass实现

// 常量折叠
class ConstantFoldingPass : public OptimizationPass {
public:
    std::string GetName() const override { return "ConstantFolding"; }
    Status Run(Graph* graph) override;
};

// 死代码消除
class DeadCodeEliminationPass : public OptimizationPass {
public:
    std::string GetName() const override { return "DeadCodeElimination"; }
    Status Run(Graph* graph) override;
};

// 算子融合
class OperatorFusionPass : public OptimizationPass {
public:
    std::string GetName() const override { return "OperatorFusion"; }
    Status Run(Graph* graph) override;
    
    // 融合模式
    bool CanFuseConvBNReLU(Node* conv, Node* bn, Node* relu) const;
    bool CanFuseMatMulAdd(Node* matmul, Node* add) const;
    bool CanFuseConvReLU(Node* conv, Node* relu) const;
    bool CanFuseBNReLU(Node* bn, Node* relu) const;
    
private:
    Status FuseConvBNReLU(Graph* graph, Node* conv, Node* bn, Node* relu);
    Status FuseMatMulAdd(Graph* graph, Node* matmul, Node* add);
    Status FuseConvReLU(Graph* graph, Node* conv, Node* relu);
    Status FuseBNReLU(Graph* graph, Node* bn, Node* relu);
};

// 内存布局优化
class MemoryLayoutOptimizationPass : public OptimizationPass {
public:
    std::string GetName() const override { return "MemoryLayoutOptimization"; }
    Status Run(Graph* graph) override;
};

// 子图替换
class SubgraphReplacementPass : public OptimizationPass {
public:
    std::string GetName() const override { return "SubgraphReplacement"; }
    Status Run(Graph* graph) override;
    
private:
    // 辅助函数：检查张量是否全为0
    bool IsZeroTensor(Tensor* tensor) const;
};

} // namespace inferunity

