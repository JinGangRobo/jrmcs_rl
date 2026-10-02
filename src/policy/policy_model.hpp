#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "onnx_runtime.hpp"

namespace rmcs_rl {

class PolicyModel {
public:
    struct Config {
        std::string path;
        std::string input_name = "obs";
        // Empty for single-input models. Set to the history tensor name (e.g.
        // "obs_history") to deploy a dual-input (obs + obs_history) model.
        std::string history_input_name;
        std::string output_name = "actions";
        bool normalization_from_metadata = true;
        std::optional<double> obs_clip;
        std::optional<double> action_clip;
    };

    struct Info {
        std::string path;
        std::string obs_signature;
        std::string actions_signature;
        std::string version;
        // obs_size is the whole observation (frame_size * history_length), matching the
        // bridge layout contract used for the layout hash.
        std::size_t obs_size = 0;
        std::size_t frame_size = 0;
        std::size_t history_length = 1;
        std::size_t action_size = 0;
        std::uint64_t layout_hash = 0;
        std::uint64_t model_id = 0;
    };

    explicit PolicyModel(const Config& config);

    [[nodiscard]] const Info& info() const { return info_; }
    bool run(std::span<const double> observation, std::span<double> action, std::string& error);

private:
    void load_normalization(const Config& config);

    OnnxRuntime runtime_;
    Info info_;
    std::vector<double> obs_mean_;
    std::vector<double> obs_std_;
    std::optional<double> obs_clip_;
    std::optional<double> action_clip_;
    std::vector<float> obs_buffer_;
    std::vector<float> action_buffer_;
};

} // namespace rmcs_rl
