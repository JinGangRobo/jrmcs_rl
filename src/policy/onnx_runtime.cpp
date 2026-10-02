#include "onnx_runtime.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace rmcs_rl {
namespace {

std::string input_name_at(
    Ort::Session& session, std::size_t index, Ort::AllocatorWithDefaultOptions& allocator) {
    const auto name = session.GetInputNameAllocated(index, allocator);
    return std::string{name.get()};
}

std::size_t input_width_at(Ort::Session& session, std::size_t index) {
    const auto type = session.GetInputTypeInfo(index);
    const auto info = type.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::invalid_argument("input tensor element type must be float32");
    const auto shape = info.GetShape();
    if (shape.size() != 2)
        throw std::invalid_argument("input tensor rank must be 2 ([batch, N])");
    if (shape[0] != 1 && shape[0] != -1)
        throw std::invalid_argument("input batch dimension must be 1 or dynamic");
    if (shape[1] <= 0)
        throw std::invalid_argument("input feature dimension must be a concrete positive value");
    return static_cast<std::size_t>(shape[1]);
}

std::size_t validate_output(
    Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator,
    const std::string& output_name) {
    if (session.GetOutputCount() != 1)
        throw std::invalid_argument("model must have exactly one output");
    const auto actual_output = session.GetOutputNameAllocated(0, allocator);
    if (actual_output.get() != output_name)
        throw std::invalid_argument(
            "output tensor name must be '" + output_name + "', got '" + actual_output.get() + "'");
    const auto type = session.GetOutputTypeInfo(0);
    const auto info = type.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
        throw std::invalid_argument("output tensor element type must be float32");
    const auto shape = info.GetShape();
    if (shape.size() != 2)
        throw std::invalid_argument("output tensor rank must be 2 ([batch, N])");
    if (shape[0] != 1 && shape[0] != -1)
        throw std::invalid_argument("output batch dimension must be 1 or dynamic");
    if (shape[1] <= 0)
        throw std::invalid_argument("output feature dimension must be a concrete positive value");
    return static_cast<std::size_t>(shape[1]);
}

} // namespace

OnnxRuntime::OnnxRuntime(
    const std::string& model_path, const std::string& input_name,
    const std::string& history_input_name, const std::string& output_name)
    : input_name_(input_name)
    , history_input_name_(history_input_name)
    , output_name_(output_name) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    session_ = Ort::Session{env_, model_path.c_str(), options};

    output_size_ = validate_output(session_, allocator_, output_name_);

    const std::size_t input_count = session_.GetInputCount();
    if (history_input_name_.empty()) {
        if (input_count != 1)
            throw std::invalid_argument(
                "model must have exactly one input; set 'history_input_name' to deploy a "
                "dual-input (obs + obs_history) model");
        const auto actual = input_name_at(session_, 0, allocator_);
        if (actual != input_name_)
            throw std::invalid_argument(
                "input tensor name must be '" + input_name_ + "', got '" + actual + "'");
        const std::size_t width = input_width_at(session_, 0);
        obs_size_ = width;
        frame_size_ = width;
        history_length_ = 1;
        bindings_.push_back(
            InputBinding{input_name_, {1, static_cast<std::int64_t>(width)}, 0, width});
    } else {
        if (input_name_ == history_input_name_)
            throw std::invalid_argument(
                "'input_name' and 'history_input_name' must name two distinct inputs");
        if (input_count != 2)
            throw std::invalid_argument("dual-input model must have exactly two inputs");

        const auto name_a = input_name_at(session_, 0, allocator_);
        const auto name_b = input_name_at(session_, 1, allocator_);
        const auto width_of = [&](const std::string& want) -> std::size_t {
            if (name_a == want)
                return input_width_at(session_, 0);
            if (name_b == want)
                return input_width_at(session_, 1);
            throw std::invalid_argument(
                "model input '" + want + "' not found; model declares '" + name_a + "' and '"
                + name_b + "'");
        };
        const std::size_t frame_width = width_of(input_name_);
        const std::size_t history_width = width_of(history_input_name_);
        if (history_width <= frame_width || history_width % frame_width != 0)
            throw std::invalid_argument(
                "history input width (" + std::to_string(history_width)
                + ") must be an integer multiple of the frame input width ("
                + std::to_string(frame_width) + ")");
        obs_size_ = history_width;
        frame_size_ = frame_width;
        history_length_ = history_width / frame_width;
        // Whole history goes to the history input; the newest frame is the trailing slice.
        bindings_.push_back(InputBinding{
            history_input_name_, {1, static_cast<std::int64_t>(history_width)}, 0, history_width});
        bindings_.push_back(InputBinding{
            input_name_, {1, static_cast<std::int64_t>(frame_width)}, history_width - frame_width,
            frame_width});
    }

    input_buffer_.resize(obs_size_);
}

std::optional<std::string> OnnxRuntime::metadata(const char* key) const {
    const auto model_metadata = session_.GetModelMetadata();
    const auto value = model_metadata.LookupCustomMetadataMapAllocated(key, allocator_);
    if (!value)
        return std::nullopt;
    return std::string{value.get()};
}

void OnnxRuntime::run(std::span<const float> input, std::span<float> output) {
    if (input.size() != obs_size_ || output.size() != output_size_)
        throw std::invalid_argument("inference buffers do not match model dimensions");
    std::copy(input.begin(), input.end(), input_buffer_.begin());

    std::vector<Ort::Value> tensors;
    std::vector<const char*> input_names;
    tensors.reserve(bindings_.size());
    input_names.reserve(bindings_.size());
    for (const auto& binding : bindings_) {
        tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, input_buffer_.data() + binding.offset, binding.width,
            binding.shape.data(), binding.shape.size()));
        input_names.push_back(binding.name.c_str());
    }

    const char* output_names[] = {output_name_.c_str()};
    const auto outputs = session_.Run(
        Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), tensors.size(), output_names,
        1);
    if (outputs.size() != 1 || !outputs[0].IsTensor()
        || outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != output_size_)
        throw std::runtime_error("inference output does not match model dimensions");
    std::copy_n(outputs[0].GetTensorData<float>(), output_size_, output.begin());
}

} // namespace rmcs_rl
