// Python绑定（使用pybind11）
// 参考ONNX Runtime的Python API设计

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "inferunity/engine.h"
#include "inferunity/tensor.h"
#include "inferunity/types.h"
#include "inferunity/graph.h"

namespace py = pybind11;
using namespace inferunity;

// NumPy数组到Tensor的转换
std::shared_ptr<Tensor> numpy_to_tensor(py::array_t<float> arr) {
    auto buf = arr.request();
    
    // 获取形状
    std::vector<int64_t> shape;
    for (size_t i = 0; i < buf.ndim; ++i) {
        shape.push_back(buf.shape[i]);
    }
    
    // 创建Tensor
    Shape tensor_shape(shape);
    auto tensor = CreateTensor(tensor_shape, DataType::FLOAT32, DeviceType::CPU);
    
    // 复制数据
    float* tensor_data = static_cast<float*>(tensor->GetData());
    float* numpy_data = static_cast<float*>(buf.ptr);
    std::memcpy(tensor_data, numpy_data, buf.size * sizeof(float));
    
    return tensor;
}

// Tensor到NumPy数组的转换
py::array_t<float> tensor_to_numpy(std::shared_ptr<Tensor> tensor) {
    if (!tensor) {
        throw std::runtime_error("Tensor is null");
    }
    
    const Shape& shape = tensor->GetShape();
    std::vector<size_t> numpy_shape;
    std::vector<size_t> numpy_strides;
    
    size_t stride = sizeof(float);
    for (int64_t i = shape.dims.size() - 1; i >= 0; --i) {
        numpy_shape.push_back(static_cast<size_t>(shape.dims[i]));
        numpy_strides.push_back(stride);
        stride *= shape.dims[i];
    }
    std::reverse(numpy_shape.begin(), numpy_shape.end());
    std::reverse(numpy_strides.begin(), numpy_strides.end());
    
    float* data = static_cast<float*>(tensor->GetData());
    size_t size = tensor->GetElementCount();
    
    return py::array_t<float>(
        numpy_shape,
        numpy_strides,
        data,
        py::cast(tensor)  // 保持Tensor对象存活
    );
}

// SessionOptions绑定
void bind_session_options(py::module& m) {
    py::class_<SessionOptions>(m, "SessionOptions")
        .def(py::init<>())
        .def_readwrite("execution_providers", &SessionOptions::execution_providers)
        .def_readwrite("device_id", &SessionOptions::device_id)
        .def_readwrite("graph_optimization_level", &SessionOptions::graph_optimization_level)
        .def_readwrite("enable_operator_fusion", &SessionOptions::enable_operator_fusion)
        .def_readwrite("num_threads", &SessionOptions::num_threads)
        .def_readwrite("max_batch_size", &SessionOptions::max_batch_size)
        .def_readwrite("enable_profiling", &SessionOptions::enable_profiling);
    
    py::enum_<SessionOptions::GraphOptimizationLevel>(m, "GraphOptimizationLevel")
        .value("NONE", SessionOptions::GraphOptimizationLevel::NONE)
        .value("BASIC", SessionOptions::GraphOptimizationLevel::BASIC)
        .value("EXTENDED", SessionOptions::GraphOptimizationLevel::EXTENDED)
        .value("ALL", SessionOptions::GraphOptimizationLevel::ALL);
}

// Status绑定
void bind_status(py::module& m) {
    py::class_<Status>(m, "Status")
        .def("is_ok", &Status::IsOk)
        .def("message", &Status::Message)
        .def("__bool__", &Status::IsOk)
        .def("__str__", &Status::Message);
}

// Shape绑定
void bind_shape(py::module& m) {
    py::class_<Shape>(m, "Shape")
        .def(py::init<>())
        .def(py::init<const std::vector<int64_t>&>())
        .def_readwrite("dims", &Shape::dims)
        .def_readwrite("is_dynamic", &Shape::is_dynamic)
        .def("get_element_count", &Shape::GetElementCount)
        .def("is_dynamic", &Shape::IsDynamic)
        .def("get_static_shape", &Shape::GetStaticShape);
}

// InferenceSession绑定
void bind_inference_session(py::module& m) {
    py::class_<InferenceSession, std::unique_ptr<InferenceSession>>(m, "InferenceSession")
        .def_static("create", &InferenceSession::Create,
                   py::arg("options") = SessionOptions(),
                   "Create an inference session")
        .def("load_model", py::overload_cast<const std::string&>(&InferenceSession::LoadModel),
             "Load model from file")
        .def("get_input_shapes", &InferenceSession::GetInputShapes,
             "Get input shapes")
        .def("get_output_shapes", &InferenceSession::GetOutputShapes,
             "Get output shapes")
        .def("get_input_names", &InferenceSession::GetInputNames,
             "Get input names")
        .def("get_output_names", &InferenceSession::GetOutputNames,
             "Get output names")
        .def("run", [](InferenceSession& session, 
                       const std::vector<py::array_t<float>>& inputs) {
            // 转换输入
            std::vector<Tensor*> input_tensors;
            std::vector<std::shared_ptr<Tensor>> input_storage;
            
            for (const auto& arr : inputs) {
                auto tensor = numpy_to_tensor(arr);
                input_storage.push_back(tensor);
                input_tensors.push_back(tensor.get());
            }
            
            // 执行推理
            std::vector<Tensor*> outputs;
            Status status = session.Run(input_tensors, outputs);
            
            if (!status.IsOk()) {
                throw std::runtime_error("Run failed: " + status.Message());
            }
            
            // 转换输出
            std::vector<py::array_t<float>> result;
            for (Tensor* output : outputs) {
                // 注意：这里需要保持Tensor的引用
                // 简化实现，实际应该从session获取shared_ptr
                result.push_back(py::array_t<float>());  // TODO: 完整实现
            }
            
            return result;
        }, "Run inference")
        .def("create_input_tensor", [](InferenceSession& session, size_t index) {
            return session.CreateInputTensor(index);
        }, "Create input tensor");
}

// Tensor绑定
void bind_tensor(py::module& m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def("get_shape", &Tensor::GetShape, py::return_value_policy::reference_internal)
        .def("get_data", [](Tensor& tensor) {
            return tensor_to_numpy(std::shared_ptr<Tensor>(&tensor, [](Tensor*){}));
        }, "Get data as NumPy array")
        .def("get_element_count", &Tensor::GetElementCount)
        .def("get_size_in_bytes", &Tensor::GetSizeInBytes);
}

// 主模块
PYBIND11_MODULE(inferunity_py, m) {
    m.doc() = "InferUnity Python bindings";
    
    // 绑定类型
    bind_status(m);
    bind_shape(m);
    bind_session_options(m);
    bind_tensor(m);
    bind_inference_session(m);
    
    // 数据类型枚举
    py::enum_<DataType>(m, "DataType")
        .value("FLOAT32", DataType::FLOAT32)
        .value("FLOAT16", DataType::FLOAT16)
        .value("INT32", DataType::INT32)
        .value("INT64", DataType::INT64)
        .value("INT8", DataType::INT8)
        .value("UINT8", DataType::UINT8)
        .value("UNKNOWN", DataType::UNKNOWN);
    
    // 设备类型枚举
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .value("TENSORRT", DeviceType::TENSORRT)
        .value("VULKAN", DeviceType::VULKAN)
        .value("METAL", DeviceType::METAL);
    
    // 工具函数
    m.def("create_tensor", [](const std::vector<int64_t>& shape, DataType dtype) {
        Shape s(shape);
        return CreateTensor(s, dtype, DeviceType::CPU);
    }, "Create a tensor", py::arg("shape"), py::arg("dtype") = DataType::FLOAT32);
}

