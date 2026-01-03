// 模型转换工具
// 参考ONNX Runtime的模型转换实现

#include "inferunity/engine.h"
#include "inferunity/graph.h"
#include "inferunity/optimizer.h"
#include "frontend/onnx_parser.h"
// 前向声明形状推断函数
namespace inferunity {
    Status InferShapes(Graph* graph);
}
#include <iostream>
#include <fstream>
#include <string>

using namespace inferunity;
using namespace inferunity::frontend;

// 验证模型
Status ValidateModel(const std::string& model_path) {
    std::cout << "Validating model: " << model_path << std::endl;
    
    // 创建解析器
    ONNXParser parser;
    Status status = parser.LoadFromFile(model_path);
    if (!status.IsOk()) {
        return status;
    }
    
    // 转换为图
    std::unique_ptr<Graph> graph;
    status = parser.ConvertToGraph(graph);
    if (!status.IsOk()) {
        return status;
    }
    
    // 验证图
    status = graph->Validate();
    if (!status.IsOk()) {
        return status;
    }
    
    // 形状推断（使用全局函数）
    status = InferShapes(graph.get());
    if (!status.IsOk()) {
        std::cerr << "Warning: Shape inference failed: " << status.Message() << std::endl;
    }
    
    // 打印模型信息
    std::cout << "Model validation successful!" << std::endl;
    std::cout << "  Inputs: " << graph->GetInputs().size() << std::endl;
    std::cout << "  Outputs: " << graph->GetOutputs().size() << std::endl;
    std::cout << "  Nodes: " << graph->GetNodes().size() << std::endl;
    
    return Status::Ok();
}

// 转换模型格式
Status ConvertModel(const std::string& input_path, const std::string& output_path) {
    std::cout << "Converting model:" << std::endl;
    std::cout << "  Input:  " << input_path << std::endl;
    std::cout << "  Output: " << output_path << std::endl;
    
    // 加载ONNX模型
    ONNXParser parser;
    Status status = parser.LoadFromFile(input_path);
    if (!status.IsOk()) {
        return status;
    }
    
    // 转换为内部图格式
    std::unique_ptr<Graph> graph;
    status = parser.ConvertToGraph(graph);
    if (!status.IsOk()) {
        return status;
    }
    
    // 优化图（可选）
    Optimizer optimizer;
    // 注册优化Pass
    optimizer.RegisterPass(std::make_unique<ConstantFoldingPass>());
    optimizer.RegisterPass(std::make_unique<DeadCodeEliminationPass>());
    optimizer.RegisterPass(std::make_unique<OperatorFusionPass>());
    
    status = optimizer.Optimize(graph.get());
    if (!status.IsOk()) {
        std::cerr << "Warning: Graph optimization failed: " << status.Message() << std::endl;
    }
    
    // 序列化图
    status = graph->Serialize(output_path);
    if (!status.IsOk()) {
        return status;
    }
    
    std::cout << "Model conversion successful!" << std::endl;
    return Status::Ok();
}

// 打印模型信息
Status PrintModelInfo(const std::string& model_path) {
    std::cout << "Model Information: " << model_path << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // 创建会话
    SessionOptions options;
    options.execution_providers = {"CPUExecutionProvider"};
    
    auto session = InferenceSession::Create(options);
    if (!session) {
        return Status::Error(StatusCode::ERROR_RUNTIME_ERROR, "Failed to create session");
    }
    
    // 加载模型
    Status status = session->LoadModel(model_path);
    if (!status.IsOk()) {
        return status;
    }
    
    // 获取模型信息
    auto input_shapes = session->GetInputShapes();
    auto output_shapes = session->GetOutputShapes();
    auto input_names = session->GetInputNames();
    auto output_names = session->GetOutputNames();
    
    std::cout << "\nInputs (" << input_shapes.size() << "):" << std::endl;
    for (size_t i = 0; i < input_shapes.size(); ++i) {
        std::cout << "  [" << i << "] ";
        if (i < input_names.size()) {
            std::cout << input_names[i] << ": ";
        }
        std::cout << "Shape(";
        for (size_t j = 0; j < input_shapes[i].dims.size(); ++j) {
            std::cout << input_shapes[i].dims[j];
            if (j < input_shapes[i].dims.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    std::cout << "\nOutputs (" << output_shapes.size() << "):" << std::endl;
    for (size_t i = 0; i < output_shapes.size(); ++i) {
        std::cout << "  [" << i << "] ";
        if (i < output_names.size()) {
            std::cout << output_names[i] << ": ";
        }
        std::cout << "Shape(";
        for (size_t j = 0; j < output_shapes[i].dims.size(); ++j) {
            std::cout << output_shapes[i].dims[j];
            if (j < output_shapes[i].dims.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    // 获取图信息
    const Graph* graph = session->GetGraph();
    if (graph) {
        std::cout << "\nGraph Statistics:" << std::endl;
        std::cout << "  Total nodes: " << graph->GetNodes().size() << std::endl;
        std::cout << "  Total values: " << graph->GetValues().size() << std::endl;
    }
    
    std::cout << std::string(50, '=') << std::endl;
    
    return Status::Ok();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <command> [options]" << std::endl;
        std::cerr << "\nCommands:" << std::endl;
        std::cerr << "  validate <model_path>     Validate ONNX model" << std::endl;
        std::cerr << "  convert <input> <output>  Convert ONNX to internal format" << std::endl;
        std::cerr << "  info <model_path>         Print model information" << std::endl;
        return 1;
    }
    
    std::string command = argv[1];
    
    if (command == "validate") {
        if (argc < 3) {
            std::cerr << "Error: Missing model path" << std::endl;
            return 1;
        }
        std::string model_path = argv[2];
        Status status = ValidateModel(model_path);
        if (!status.IsOk()) {
            std::cerr << "Validation failed: " << status.Message() << std::endl;
            return 1;
        }
    } else if (command == "convert") {
        if (argc < 4) {
            std::cerr << "Error: Missing input or output path" << std::endl;
            return 1;
        }
        std::string input_path = argv[2];
        std::string output_path = argv[3];
        Status status = ConvertModel(input_path, output_path);
        if (!status.IsOk()) {
            std::cerr << "Conversion failed: " << status.Message() << std::endl;
            return 1;
        }
    } else if (command == "info") {
        if (argc < 3) {
            std::cerr << "Error: Missing model path" << std::endl;
            return 1;
        }
        std::string model_path = argv[2];
        Status status = PrintModelInfo(model_path);
        if (!status.IsOk()) {
            std::cerr << "Failed to get model info: " << status.Message() << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: Unknown command: " << command << std::endl;
        return 1;
    }
    
    return 0;
}

