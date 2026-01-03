// 形状操作算子实现
// 参考NCNN和ONNX Runtime的实现

#include "inferunity/operator.h"
#include "inferunity/tensor.h"
#include <algorithm>
#include <numeric>
#include <cstring>

namespace inferunity {
namespace operators {

// Reshape算子
class ReshapeOperator : public Operator {
public:
    std::string GetName() const override { return "Reshape"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, 
                               "Reshape requires at least 2 inputs");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, 
                               "Reshape requires at least 2 inputs");
        }
        
        const Shape& input_shape = inputs[0]->GetShape();
        Tensor* shape_tensor = inputs[1];
        
        // 从shape tensor获取目标形状
        if (shape_tensor->GetDataType() != DataType::INT64) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Shape tensor must be INT64");
        }
        
        const int64_t* shape_data = static_cast<const int64_t*>(shape_tensor->GetData());
        size_t shape_size = shape_tensor->GetElementCount();
        
        std::vector<int64_t> target_shape(shape_data, shape_data + shape_size);
        
        // 处理-1维度（自动推断）
        int64_t total_elements = input_shape.GetElementCount();
        int64_t known_elements = 1;
        int unknown_dim = -1;
        
        for (size_t i = 0; i < target_shape.size(); ++i) {
            if (target_shape[i] == -1) {
                if (unknown_dim != -1) {
                    return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                       "Only one dimension can be -1");
                }
                unknown_dim = static_cast<int>(i);
            } else if (target_shape[i] > 0) {
                known_elements *= target_shape[i];
            }
        }
        
        if (unknown_dim != -1) {
            if (known_elements == 0 || total_elements % known_elements != 0) {
                return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                   "Cannot infer dimension size");
            }
            target_shape[unknown_dim] = total_elements / known_elements;
        }
        
        // 验证元素总数
        int64_t new_total = 1;
        for (int64_t dim : target_shape) {
            new_total *= dim;
        }
        
        if (new_total != total_elements) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Shape mismatch: total elements don't match");
        }
        
        output_shapes.push_back(Shape(target_shape));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        // Reshape只是改变视图，不复制数据
        // 实际实现中，Tensor应该支持视图（共享内存）
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        // 验证元素总数
        if (input->GetElementCount() != output->GetElementCount()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Element count mismatch");
        }
        
        // 如果支持视图，直接共享内存
        // 否则需要复制数据（简化实现）
        size_t size = input->GetSizeInBytes();
        std::memcpy(output->GetData(), input->GetData(), size);
        
        return Status::Ok();
    }
};

// Concat算子
class ConcatOperator : public Operator {
public:
    std::string GetName() const override { return "Concat"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, 
                               "Concat requires at least 1 input");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
        }
        
        // 获取axis属性（默认axis=0）
        int axis = 0;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
        }
        
        const Shape& first_shape = inputs[0]->GetShape();
        if (axis < 0 || axis >= static_cast<int>(first_shape.dims.size())) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid axis");
        }
        
        std::vector<int64_t> output_dims = first_shape.dims;
        int64_t concat_dim = first_shape.dims[axis];
        
        // 验证所有输入形状（除了axis维度）都相同
        for (size_t i = 1; i < inputs.size(); ++i) {
            const Shape& shape = inputs[i]->GetShape();
            if (shape.dims.size() != first_shape.dims.size()) {
                return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                   "Shape rank mismatch");
            }
            
            for (int j = 0; j < static_cast<int>(shape.dims.size()); ++j) {
                if (j != axis && shape.dims[j] != first_shape.dims[j]) {
                    return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                       "Shape mismatch");
                }
            }
            
            concat_dim += shape.dims[axis];
        }
        
        output_dims[axis] = concat_dim;
        output_shapes.push_back(Shape(output_dims));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        int axis = 0;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
        }
        Tensor* output = outputs[0];
        const Shape& output_shape = output->GetShape();
        
        // 计算每个输入在axis维度上的大小
        size_t element_size = GetDataTypeSize(inputs[0]->GetDataType());
        size_t axis_stride = element_size;
        for (int i = static_cast<int>(output_shape.dims.size()) - 1; i > axis; --i) {
            axis_stride *= output_shape.dims[i];
        }
        
        size_t axis_size = output_shape.dims[axis];
        size_t axis_offset = 0;
        
        // 拼接数据
        uint8_t* output_data = static_cast<uint8_t*>(output->GetData());
        
        for (Tensor* input : inputs) {
            size_t input_axis_size = input->GetShape().dims[axis];
            size_t input_size = input_axis_size * axis_stride;
            
            for (size_t i = 0; i < axis_stride / element_size; ++i) {
                size_t src_offset = i * input_axis_size * element_size;
                size_t dst_offset = (i * axis_size + axis_offset) * element_size;
                
                std::memcpy(output_data + dst_offset,
                           static_cast<uint8_t*>(input->GetData()) + src_offset,
                           input_axis_size * element_size);
            }
            
            axis_offset += input_axis_size;
        }
        
        return Status::Ok();
    }
    
private:
    size_t GetDataTypeSize(DataType dtype) {
        switch (dtype) {
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT16: return 2;
            case DataType::INT32: return 4;
            case DataType::INT64: return 8;
            case DataType::INT8: return 1;
            case DataType::UINT8: return 1;
            default: return 4;
        }
    }
};

// Split算子
class SplitOperator : public Operator {
public:
    std::string GetName() const override { return "Split"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, 
                               "Split requires at least 1 input");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
        }
        
        // 从属性获取split和axis
        int axis = 0;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
        }
        
        std::vector<int64_t> splits;
        auto split_attr = GetAttribute("split");
        if (split_attr.GetType() == AttributeValue::Type::INTS) {
            splits = split_attr.GetInts();
        }
        
        const Shape& input_shape = inputs[0]->GetShape();
        if (axis < 0 || axis >= static_cast<int>(input_shape.dims.size())) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid axis");
        }
        
        // 如果没有指定splits，平均分割
        if (splits.empty()) {
            // 默认分割为2个相等的部分
            int64_t dim_size = input_shape.dims[axis];
            splits = {dim_size / 2, dim_size - dim_size / 2};
        }
        
        // 验证splits总和
        int64_t total = 0;
        for (int64_t s : splits) {
            total += s;
        }
        if (total != input_shape.dims[axis]) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Split sizes don't match axis dimension");
        }
        
        // 为每个split创建输出形状
        for (int64_t split_size : splits) {
            std::vector<int64_t> output_dims = input_shape.dims;
            output_dims[axis] = split_size;
            output_shapes.push_back(Shape(output_dims));
        }
        
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        // 从属性获取axis和splits
        int axis = 0;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
        }
        
        std::vector<int64_t> splits;
        auto split_attr = GetAttribute("split");
        if (split_attr.GetType() == AttributeValue::Type::INTS) {
            splits = split_attr.GetInts();
        }
        
        Tensor* input = inputs[0];
        const Shape& input_shape = input->GetShape();
        
        // 如果没有指定splits，平均分割
        if (splits.empty()) {
            int64_t dim_size = input_shape.dims[axis];
            int64_t num_outputs = outputs.size();
            int64_t base_size = dim_size / num_outputs;
            int64_t remainder = dim_size % num_outputs;
            
            for (int64_t i = 0; i < num_outputs; ++i) {
                splits.push_back(base_size + (i < remainder ? 1 : 0));
            }
        }
        
        // 验证splits数量
        if (splits.size() != outputs.size()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Split sizes don't match output count");
        }
        
        // 计算每个输出在axis维度上的大小
        size_t element_size = Tensor::GetDataTypeSize(input->GetDataType());
        size_t axis_stride = element_size;
        for (int i = static_cast<int>(input_shape.dims.size()) - 1; i > axis; --i) {
            axis_stride *= input_shape.dims[i];
        }
        
        size_t axis_offset = 0;
        const uint8_t* input_data = static_cast<const uint8_t*>(input->GetData());
        
        // 分割数据到各个输出
        for (size_t i = 0; i < outputs.size(); ++i) {
            Tensor* output = outputs[i];
            size_t output_axis_size = splits[i];
            size_t output_size = output_axis_size * axis_stride;
            
            uint8_t* output_data = static_cast<uint8_t*>(output->GetData());
            
            for (size_t j = 0; j < axis_stride / element_size; ++j) {
                size_t src_offset = (j * input_shape.dims[axis] + axis_offset) * element_size;
                size_t dst_offset = j * output_axis_size * element_size;
                
                std::memcpy(output_data + dst_offset,
                           input_data + src_offset,
                           output_axis_size * element_size);
            }
            
            axis_offset += output_axis_size;
        }
        
        return Status::Ok();
    }
};

// Transpose算子
class TransposeOperator : public Operator {
public:
    std::string GetName() const override { return "Transpose"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, 
                               "Transpose requires at least 1 input");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "No inputs");
        }
        
        const Shape& input_shape = inputs[0]->GetShape();
        std::vector<int64_t> perm;
        
        // 从属性获取perm，如果没有则默认反转
        auto perm_attr = GetAttribute("perm");
        if (perm_attr.GetType() == AttributeValue::Type::INTS) {
            perm = perm_attr.GetInts();
        }
        
        if (perm.empty()) {
            perm.resize(input_shape.dims.size());
            for (size_t i = 0; i < perm.size(); ++i) {
                perm[i] = static_cast<int64_t>(perm.size() - 1 - i);
            }
        }
        
        if (perm.size() != input_shape.dims.size()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Perm size mismatch");
        }
        
        std::vector<int64_t> output_dims(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] < 0 || perm[i] >= static_cast<int64_t>(input_shape.dims.size())) {
                return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                   "Invalid perm value");
            }
            output_dims[i] = input_shape.dims[perm[i]];
        }
        
        output_shapes.push_back(Shape(output_dims));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        // 实现通用的转置操作（支持任意维度）
        const Shape& input_shape = input->GetShape();
        const float* in_data = static_cast<const float*>(input->GetData());
        float* out_data = static_cast<float*>(output->GetData());
        
        // 获取perm属性
        std::vector<int64_t> perm;
        auto perm_attr = GetAttribute("perm");
        if (perm_attr.GetType() == AttributeValue::Type::INTS) {
            perm = perm_attr.GetInts();
        }
        
        if (perm.empty()) {
            // 默认反转所有维度
            perm.resize(input_shape.dims.size());
            for (size_t i = 0; i < perm.size(); ++i) {
                perm[i] = static_cast<int64_t>(perm.size() - 1 - i);
            }
        }
        
        // 计算输入和输出的步长
        std::vector<int64_t> input_strides(input_shape.dims.size());
        std::vector<int64_t> output_strides(input_shape.dims.size());
        
        int64_t input_stride = 1;
        int64_t output_stride = 1;
        for (int i = static_cast<int>(input_shape.dims.size()) - 1; i >= 0; --i) {
            input_strides[i] = input_stride;
            input_stride *= input_shape.dims[i];
            
            int64_t output_dim = input_shape.dims[perm[i]];
            output_strides[perm[i]] = output_stride;
            output_stride *= output_dim;
        }
        
        // 计算输出步长（按perm顺序）
        std::vector<int64_t> perm_output_strides(input_shape.dims.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            perm_output_strides[i] = input_strides[perm[i]];
        }
        
        // 转置数据
        int64_t total_elements = input_shape.GetElementCount();
        std::vector<int64_t> indices(input_shape.dims.size(), 0);
        
        for (int64_t flat_idx = 0; flat_idx < total_elements; ++flat_idx) {
            // 计算输入索引
            int64_t input_idx = 0;
            int64_t temp = flat_idx;
            for (int i = static_cast<int>(input_shape.dims.size()) - 1; i >= 0; --i) {
                indices[i] = temp % input_shape.dims[i];
                temp /= input_shape.dims[i];
                input_idx += indices[i] * input_strides[i];
            }
            
            // 计算输出索引（按perm重新排列）
            int64_t output_idx = 0;
            for (size_t i = 0; i < perm.size(); ++i) {
                output_idx += indices[perm[i]] * perm_output_strides[i];
            }
            
            out_data[output_idx] = in_data[input_idx];
        }
        
        return Status::Ok();
    }
};

// 注册算子（在命名空间内）
// Gather算子 - 从输入张量中根据索引收集元素
class GatherOperator : public Operator {
public:
    std::string GetName() const override { return "Gather"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Gather requires at least 2 inputs");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Gather requires at least 2 inputs");
        }
        
        const Shape& data_shape = inputs[0]->GetShape();
        const Shape& indices_shape = inputs[1]->GetShape();
        
        // 获取axis属性（默认为0）
        int axis = 0;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
        }
        
        if (axis < 0) {
            axis += static_cast<int>(data_shape.dims.size());
        }
        
        if (axis < 0 || axis >= static_cast<int>(data_shape.dims.size())) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Invalid axis for Gather");
        }
        
        // 输出形状 = data_shape的前axis维 + indices_shape + data_shape的后(rank-axis-1)维
        Shape output_shape;
        for (int i = 0; i < axis; ++i) {
            output_shape.dims.push_back(data_shape.dims[i]);
        }
        for (size_t i = 0; i < indices_shape.dims.size(); ++i) {
            output_shape.dims.push_back(indices_shape.dims[i]);
        }
        for (size_t i = axis + 1; i < data_shape.dims.size(); ++i) {
            output_shape.dims.push_back(data_shape.dims[i]);
        }
        
        output_shapes.push_back(output_shape);
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Gather requires at least 2 inputs and 1 output");
        }
        
        Tensor* data = inputs[0];
        Tensor* indices = inputs[1];
        Tensor* output = outputs[0];
        
        const Shape& data_shape = data->GetShape();
        const Shape& indices_shape = indices->GetShape();
        const Shape& output_shape = output->GetShape();
        
        // 获取axis属性（默认为0）
        int axis = 0;
        auto axis_attr = GetAttribute("axis");
        if (axis_attr.GetType() == AttributeValue::Type::INT) {
            axis = static_cast<int>(axis_attr.GetInt());
        }
        
        // 支持任意axis（通用实现）
        if (axis < 0) {
            axis += static_cast<int>(data_shape.dims.size());
        }
        
        if (axis < 0 || axis >= static_cast<int>(data_shape.dims.size())) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Gather axis out of range");
        }
        
        const float* data_ptr = static_cast<const float*>(data->GetData());
        const int64_t* indices_ptr = static_cast<const int64_t*>(indices->GetData());
        float* output_ptr = static_cast<float*>(output->GetData());
        // indices_shape和output_shape已在上面定义
        
        // 计算每个元素的大小（axis之后的维度）
        size_t element_size = 1;
        for (size_t i = axis + 1; i < data_shape.dims.size(); ++i) {
            element_size *= data_shape.dims[i];
        }
        
        // 计算stride
        size_t stride = element_size * data_shape.dims[axis];
        
        // 执行Gather
        size_t indices_count = indices_shape.GetElementCount();
        for (size_t i = 0; i < indices_count; ++i) {
            int64_t idx = indices_ptr[i];
            if (idx < 0 || idx >= data_shape.dims[axis]) {
                return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                   "Gather index out of range");
            }
            
            const float* src = data_ptr + idx * stride;
            float* dst = output_ptr + i * element_size;
            std::memcpy(dst, src, element_size * sizeof(float));
        }
        
        return Status::Ok();
    }
};

// Slice算子 - 从输入张量中切片
class SliceOperator : public Operator {
public:
    std::string GetName() const override { return "Slice"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Slice requires at least 1 input");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Slice requires at least 1 input");
        }
        
        const Shape& input_shape = inputs[0]->GetShape();
        int rank = static_cast<int>(input_shape.dims.size());
        
        // 获取starts, ends, axes, steps属性
        // 支持从属性或输入tensor获取（ONNX Slice格式）
        std::vector<int64_t> starts, ends, axes, steps;
        
        // 尝试从属性获取
        auto starts_attr = GetAttribute("starts");
        auto ends_attr = GetAttribute("ends");
        auto axes_attr = GetAttribute("axes");
        auto steps_attr = GetAttribute("steps");
        
        if (starts_attr.GetType() == AttributeValue::Type::INTS) {
            starts = starts_attr.GetInts();
        }
        if (ends_attr.GetType() == AttributeValue::Type::INTS) {
            ends = ends_attr.GetInts();
        }
        if (axes_attr.GetType() == AttributeValue::Type::INTS) {
            axes = axes_attr.GetInts();
        }
        if (steps_attr.GetType() == AttributeValue::Type::INTS) {
            steps = steps_attr.GetInts();
        }
        
        // 如果属性中没有，尝试从输入tensor获取（ONNX Slice可以有starts/ends/axes/steps作为输入）
        if (starts.empty() && inputs.size() >= 2) {
            Tensor* starts_tensor = inputs[1];
            if (starts_tensor->GetDataType() == DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(starts_tensor->GetData());
                size_t count = starts_tensor->GetElementCount();
                starts.assign(data, data + count);
            }
        }
        if (ends.empty() && inputs.size() >= 3) {
            Tensor* ends_tensor = inputs[2];
            if (ends_tensor->GetDataType() == DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(ends_tensor->GetData());
                size_t count = ends_tensor->GetElementCount();
                ends.assign(data, data + count);
            }
        }
        
        // 如果没有指定axes，默认对所有维度切片
        if (axes.empty()) {
            for (int i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        }
        
        // 如果没有指定steps，默认为1
        if (steps.empty()) {
            steps.assign(axes.size(), 1);
        }
        
        // 验证参数
        if (starts.size() != axes.size() || ends.size() != axes.size()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Slice: starts, ends, and axes must have the same size");
        }
        
        // 计算输出形状
        Shape output_shape = input_shape;
        for (size_t i = 0; i < axes.size(); ++i) {
            int axis = static_cast<int>(axes[i]);
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank) {
                return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                   "Slice: invalid axis");
            }
            
            int64_t dim_size = input_shape.dims[axis];
            int64_t start = starts[i];
            int64_t end = ends[i];
            int64_t step = steps[i];
            
            // 处理负数索引
            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size;
            
            // 限制范围
            start = std::max(0LL, std::min(start, dim_size));
            end = std::max(0LL, std::min(end, dim_size));
            
            // 计算输出维度
            if (step > 0) {
                output_shape.dims[axis] = (end - start + step - 1) / step;
            } else if (step < 0) {
                output_shape.dims[axis] = (start - end - step - 1) / (-step);
            } else {
                return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                   "Slice: step cannot be zero");
            }
            
            output_shape.dims[axis] = std::max(0LL, output_shape.dims[axis]);
        }
        
        output_shapes.push_back(output_shape);
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.empty() || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Slice requires at least 1 input and 1 output");
        }
        
        Tensor* input = inputs[0];
        Tensor* output = outputs[0];
        
        const Shape& input_shape = input->GetShape();
        const Shape& output_shape = output->GetShape();
        int rank = static_cast<int>(input_shape.dims.size());
        
        // 获取切片参数（与InferOutputShape相同的逻辑）
        std::vector<int64_t> starts, ends, axes, steps;
        
        auto starts_attr = GetAttribute("starts");
        auto ends_attr = GetAttribute("ends");
        auto axes_attr = GetAttribute("axes");
        auto steps_attr = GetAttribute("steps");
        
        if (starts_attr.GetType() == AttributeValue::Type::INTS) {
            starts = starts_attr.GetInts();
        }
        if (ends_attr.GetType() == AttributeValue::Type::INTS) {
            ends = ends_attr.GetInts();
        }
        if (axes_attr.GetType() == AttributeValue::Type::INTS) {
            axes = axes_attr.GetInts();
        }
        if (steps_attr.GetType() == AttributeValue::Type::INTS) {
            steps = steps_attr.GetInts();
        }
        
        // 从输入tensor获取（如果属性中没有）
        if (starts.empty() && inputs.size() >= 2) {
            Tensor* starts_tensor = inputs[1];
            if (starts_tensor->GetDataType() == DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(starts_tensor->GetData());
                size_t count = starts_tensor->GetElementCount();
                starts.assign(data, data + count);
            }
        }
        if (ends.empty() && inputs.size() >= 3) {
            Tensor* ends_tensor = inputs[2];
            if (ends_tensor->GetDataType() == DataType::INT64) {
                const int64_t* data = static_cast<const int64_t*>(ends_tensor->GetData());
                size_t count = ends_tensor->GetElementCount();
                ends.assign(data, data + count);
            }
        }
        
        if (axes.empty()) {
            for (int i = 0; i < rank; ++i) {
                axes.push_back(i);
            }
        }
        if (steps.empty()) {
            steps.assign(axes.size(), 1);
        }
        
        // 如果形状匹配且没有切片，直接拷贝
        if (input_shape.dims == output_shape.dims && 
            std::all_of(steps.begin(), steps.end(), [](int64_t s) { return s == 1; }) &&
            std::all_of(starts.begin(), starts.end(), [](int64_t s) { return s == 0; })) {
            bool all_ends_max = true;
            for (size_t i = 0; i < axes.size(); ++i) {
                if (ends[i] != input_shape.dims[axes[i]]) {
                    all_ends_max = false;
                    break;
                }
            }
            if (all_ends_max) {
                return input->CopyTo(*output);
            }
        }
        
        // 实现完整的切片逻辑
        // 使用Tensor::Slice方法（如果可能）或手动实现
        // 简化实现：对于简单情况使用Tensor::Slice，复杂情况手动实现
        
        // 检查是否可以使用Tensor::Slice（单维度切片，step=1）
        if (axes.size() == 1 && steps[0] == 1) {
            int axis = static_cast<int>(axes[0]);
            if (axis < 0) axis += rank;
            
            // 构建starts和ends向量（只对指定axis切片）
            std::vector<int64_t> slice_starts(rank, 0);
            std::vector<int64_t> slice_ends = input_shape.dims;
            slice_starts[axis] = starts[0];
            slice_ends[axis] = ends[0];
            
            Tensor sliced = input->Slice(slice_starts, slice_ends);
            return sliced.CopyTo(*output);
        }
        
        // 通用实现：手动切片
        // 这里实现一个通用的切片逻辑
        const float* input_data = static_cast<const float*>(input->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // 计算输入和输出的strides
        std::vector<size_t> input_strides(rank);
        std::vector<size_t> output_strides(rank);
        input_strides[rank - 1] = 1;
        output_strides[rank - 1] = 1;
        for (int i = rank - 2; i >= 0; --i) {
            input_strides[i] = input_strides[i + 1] * input_shape.dims[i + 1];
            output_strides[i] = output_strides[i + 1] * output_shape.dims[i + 1];
        }
        
        // 构建切片映射
        std::vector<int64_t> slice_starts(rank, 0);
        std::vector<int64_t> slice_ends = input_shape.dims;
        std::vector<int64_t> slice_steps(rank, 1);
        
        for (size_t i = 0; i < axes.size(); ++i) {
            int axis = static_cast<int>(axes[i]);
            if (axis < 0) axis += rank;
            slice_starts[axis] = starts[i];
            slice_ends[axis] = ends[i];
            slice_steps[axis] = steps[i];
        }
        
        // 递归切片（使用迭代方式）
        std::function<void(const std::vector<int64_t>&, size_t)> slice_recursive;
        slice_recursive = [&](const std::vector<int64_t>& indices, size_t dim) {
            if (dim == rank) {
                // 计算输入和输出索引
                size_t input_idx = 0;
                size_t output_idx = 0;
                for (int i = 0; i < rank; ++i) {
                    input_idx += indices[i] * input_strides[i];
                    output_idx += (indices[i] - slice_starts[i]) / slice_steps[i] * output_strides[i];
                }
                output_data[output_idx] = input_data[input_idx];
                return;
            }
            
            int64_t start = slice_starts[dim];
            int64_t end = slice_ends[dim];
            int64_t step = slice_steps[dim];
            
            if (step > 0) {
                for (int64_t i = start; i < end; i += step) {
                    std::vector<int64_t> new_indices = indices;
                    new_indices.push_back(i);
                    slice_recursive(new_indices, dim + 1);
                }
            } else if (step < 0) {
                for (int64_t i = start; i > end; i += step) {
                    std::vector<int64_t> new_indices = indices;
                    new_indices.push_back(i);
                    slice_recursive(new_indices, dim + 1);
                }
            }
        };
        
        slice_recursive({}, 0);
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Reshape", ReshapeOperator);
REGISTER_OPERATOR("Concat", ConcatOperator);
REGISTER_OPERATOR("Split", SplitOperator);
REGISTER_OPERATOR("Transpose", TransposeOperator);
REGISTER_OPERATOR("Gather", GatherOperator);
REGISTER_OPERATOR("Slice", SliceOperator);

// Embedding算子 - 词嵌入（Transformer模型必需）
// Embedding(input_ids, weight) -> embeddings
// input_ids: [batch_size, seq_len] (INT64)
// weight: [vocab_size, embedding_dim] (FLOAT)
// output: [batch_size, seq_len, embedding_dim] (FLOAT)
class EmbeddingOperator : public Operator {
public:
    std::string GetName() const override { return "Embedding"; }
    
    Status ValidateInputs(const std::vector<Tensor*>& inputs) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Embedding requires 2 inputs (input_ids, weight)");
        }
        if (inputs[0]->GetDataType() != DataType::INT64) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Embedding input_ids must be INT64");
        }
        if (inputs[1]->GetDataType() != DataType::FLOAT32) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Embedding weight must be FLOAT32");
        }
        return Status::Ok();
    }
    
    Status InferOutputShape(const std::vector<Tensor*>& inputs,
                           std::vector<Shape>& output_shapes) const override {
        if (inputs.size() < 2) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Embedding requires 2 inputs");
        }
        
        const Shape& input_ids_shape = inputs[0]->GetShape();
        const Shape& weight_shape = inputs[1]->GetShape();
        
        // 输出形状: [batch_size, seq_len, embedding_dim]
        std::vector<int64_t> output_dims = input_ids_shape.dims;
        if (weight_shape.dims.size() >= 2) {
            output_dims.push_back(weight_shape.dims[1]);  // embedding_dim
        } else {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                               "Invalid weight shape for Embedding");
        }
        
        output_shapes.push_back(Shape(output_dims));
        return Status::Ok();
    }
    
    Status Execute(const std::vector<Tensor*>& inputs,
                  const std::vector<Tensor*>& outputs,
                  ExecutionContext* ctx) override {
        if (inputs.size() < 2 || outputs.empty()) {
            return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT, "Invalid inputs/outputs");
        }
        
        Tensor* input_ids = inputs[0];
        Tensor* weight = inputs[1];
        Tensor* output = outputs[0];
        
        const Shape& input_ids_shape = input_ids->GetShape();
        const Shape& weight_shape = weight->GetShape();
        
        int64_t batch_size = input_ids_shape.dims[0];
        int64_t seq_len = input_ids_shape.dims.size() > 1 ? input_ids_shape.dims[1] : 1;
        int64_t vocab_size = weight_shape.dims[0];
        int64_t embedding_dim = weight_shape.dims[1];
        
        const int64_t* input_ids_data = static_cast<const int64_t*>(input_ids->GetData());
        const float* weight_data = static_cast<const float*>(weight->GetData());
        float* output_data = static_cast<float*>(output->GetData());
        
        // Embedding查找: output[i, j, :] = weight[input_ids[i, j], :]
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t token_id = input_ids_data[b * seq_len + s];
                
                // 边界检查
                if (token_id < 0 || token_id >= vocab_size) {
                    return Status::Error(StatusCode::ERROR_INVALID_ARGUMENT,
                                       "Token ID out of range: " + std::to_string(token_id));
                }
                
                // 复制embedding向量
                const float* embedding = weight_data + token_id * embedding_dim;
                float* output_embedding = output_data + (b * seq_len + s) * embedding_dim;
                std::memcpy(output_embedding, embedding, embedding_dim * sizeof(float));
            }
        }
        
        return Status::Ok();
    }
};

REGISTER_OPERATOR("Embedding", EmbeddingOperator);

} // namespace operators
} // namespace inferunity

