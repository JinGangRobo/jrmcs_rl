#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace rmcs_rl {

class OnnxRuntime {
public:
    // `history_input_name` may be empty for a single-input model. When it is set the
    // model must expose exactly two inputs: a per-frame input (the narrower one) and a
    // full-history input (the wider one, an integer multiple of the frame width). The
    // history input receives the whole observation; the frame input receives the newest
    // frame, i.e. the trailing slice of the observation.
    OnnxRuntime(
        const std::string& model_path, const std::string& input_name,
        const std::string& history_input_name, const std::string& output_name);

    // Size of the whole observation fed to `run` (frame_size * history_length).
    [[nodiscard]] std::size_t input_size() const { return obs_size_; }
    [[nodiscard]] std::size_t frame_size() const { return frame_size_; }
    [[nodiscard]] std::size_t history_length() const { return history_length_; }
    [[nodiscard]] bool has_history_input() const { return !history_input_name_.empty(); }
    [[nodiscard]] std::size_t output_size() const { return output_size_; }
    [[nodiscard]] std::optional<std::string> metadata(const char* key) const;

    void run(std::span<const float> input, std::span<float> output);

private:
    struct InputBinding {
        std::string name;
        std::array<std::int64_t, 2> shape{};
        std::size_t offset = 0;
        std::size_t width = 0;
    };

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "rmcs_rl"};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Session session_{nullptr};
    std::string input_name_;
    std::string history_input_name_;
    std::string output_name_;
    std::vector<InputBinding> bindings_;
    std::vector<float> input_buffer_;
    std::size_t obs_size_ = 0;
    std::size_t frame_size_ = 0;
    std::size_t history_length_ = 1;
    std::size_t output_size_ = 0;
};

} // namespace rmcs_rl
