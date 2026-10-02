#include "policy_model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <rmcs_rl/rl_layout.hpp>

namespace rmcs_rl {
namespace {

double parse_number(const std::string& text, const char* key) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value))
        throw std::invalid_argument(std::string{"non-finite or invalid metadata: "} + key);
    return value;
}

std::vector<double> parse_numbers(const std::string& text, const char* key) {
    std::vector<double> result;
    for (std::size_t begin = 0; begin < text.size();) {
        begin = text.find_first_not_of(", \t", begin);
        if (begin == std::string::npos)
            break;
        const auto end = text.find_first_of(", \t", begin);
        result.push_back(parse_number(text.substr(begin, end - begin), key));
        if (end == std::string::npos)
            break;
        begin = end;
    }
    return result;
}

std::optional<double> clip_metadata(const OnnxRuntime& runtime, const char* key) {
    const auto text = runtime.metadata(key);
    if (!text || text->empty())
        return std::nullopt;
    const double value = parse_number(*text, key);
    if (value <= 0.0)
        throw std::invalid_argument(std::string{key} + " must be positive");
    return value;
}

} // namespace

PolicyModel::PolicyModel(const Config& config)
    : runtime_(config.path, config.input_name, config.history_input_name, config.output_name) {
    info_.path = config.path;
    info_.obs_size = runtime_.input_size();
    info_.frame_size = runtime_.frame_size();
    info_.history_length = runtime_.history_length();
    info_.action_size = runtime_.output_size();
    std::string error;
    if (!model_id_of_file(config.path, info_.model_id, error))
        throw std::runtime_error(error);

    info_.obs_signature = runtime_.metadata("rmcs_obs_layout").value_or("");
    info_.actions_signature = runtime_.metadata("rmcs_actions_layout").value_or("");
    info_.version = runtime_.metadata("policy_version").value_or("");
    if (info_.obs_signature.empty() || info_.actions_signature.empty())
        throw std::invalid_argument(
            "model is missing rmcs_obs_layout / rmcs_actions_layout; "
            "stamp it with tool/stamp_layout_metadata.py before deploying");
    info_.layout_hash = rmcs_rl::layout_hash(
        info_.obs_signature, info_.actions_signature, info_.obs_size, info_.action_size);
    if (const auto declared = runtime_.metadata("policy_layout_hash");
        declared && !declared->empty()) {
        std::size_t consumed = 0;
        const auto hash = std::stoull(*declared, &consumed, 16);
        if (consumed != declared->size() || hash != info_.layout_hash)
            throw std::invalid_argument(
                "policy_layout_hash does not match layout metadata: expected "
                + hex16(info_.layout_hash));
    }

    load_normalization(config);
    obs_buffer_.resize(info_.obs_size);
    action_buffer_.resize(info_.action_size);
}

void PolicyModel::load_normalization(const Config& config) {
    if (config.normalization_from_metadata) {
        const auto mean = runtime_.metadata("rmcs_obs_mean");
        const auto deviation = runtime_.metadata("rmcs_obs_std");
        if (mean.has_value() != deviation.has_value())
            throw std::invalid_argument("rmcs_obs_mean and rmcs_obs_std must appear together");
        if (mean && deviation) {
            obs_mean_ = parse_numbers(*mean, "rmcs_obs_mean");
            obs_std_ = parse_numbers(*deviation, "rmcs_obs_std");
            if (obs_mean_.size() != info_.obs_size || obs_std_.size() != info_.obs_size)
                throw std::invalid_argument("normalization metadata does not match obs_size");
            if (std::any_of(
                    obs_std_.begin(), obs_std_.end(), [](double value) { return value <= 0; }))
                throw std::invalid_argument("rmcs_obs_std must be positive");
        }
        obs_clip_ = clip_metadata(runtime_, "rmcs_obs_clip");
        action_clip_ = clip_metadata(runtime_, "rmcs_action_clip");
    }
    if (config.obs_clip)
        obs_clip_ = config.obs_clip;
    if (config.action_clip)
        action_clip_ = config.action_clip;
    for (const auto clip : {obs_clip_, action_clip_})
        if (clip && (!std::isfinite(*clip) || *clip < 0.0))
            throw std::invalid_argument("clip parameter must be finite and non-negative");
}

bool PolicyModel::run(
    std::span<const double> observation, std::span<double> action, std::string& error) {
    error.clear();
    if (observation.size() != info_.obs_size || action.size() != info_.action_size) {
        error = "observation/action dimensions do not match model";
        return false;
    }
    for (std::size_t i = 0; i < observation.size(); ++i) {
        double value = observation[i];
        if (!obs_mean_.empty())
            value = (value - obs_mean_[i]) / obs_std_[i];
        if (obs_clip_)
            value = std::clamp(value, -*obs_clip_, *obs_clip_);
        obs_buffer_[i] = static_cast<float>(value);
        if (!std::isfinite(obs_buffer_[i])) {
            error = "non-finite observation at index " + std::to_string(i);
            return false;
        }
    }
    try {
        runtime_.run(obs_buffer_, action_buffer_);
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
    if (!std::all_of(action_buffer_.begin(), action_buffer_.end(), [](float value) {
            return std::isfinite(value);
        })) {
        error = "policy output contains non-finite values";
        return false;
    }
    for (std::size_t i = 0; i < action.size(); ++i) {
        const double value = action_buffer_[i];
        action[i] = action_clip_ ? std::clamp(value, -*action_clip_, *action_clip_) : value;
    }
    return true;
}

} // namespace rmcs_rl
