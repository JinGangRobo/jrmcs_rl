#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rmcs_rl/msg/action.hpp>
#include <rmcs_rl/msg/observation.hpp>
#include <rmcs_rl/msg/policy_status.hpp>
#include <rmcs_rl/parameters.hpp>
#include <rmcs_rl/rl_layout.hpp>

#include "policy/policy_model.hpp"

namespace rmcs_rl {

// 单进程可载多个策略：主策略 (rl_base/rl_model_path) + 可选第二策略 (jump_rl_base/jump_model_path)。
// 每条策略各订阅 <base>/obs、发布 <base>/action，彼此独立。
class PolicyServer final : public rclcpp::Node {
public:
    PolicyServer()
        : Node(
              "policy_server",
              rclcpp::NodeOptions{}.automatically_declare_parameters_from_overrides(true)) {
        add_channel_(string_or(*this, "rl_base", "/rl"), string_or(*this, "rl_model_path", ""));

        const auto jump_base = string_or(*this, "jump_rl_base", "");
        const auto jump_model = string_or(*this, "jump_model_path", "");
        if (!jump_base.empty() && !jump_model.empty()) {
            try {
                add_channel_(jump_base, jump_model);
            } catch (const std::exception& error) {
                RCLCPP_ERROR(
                    get_logger(), "jump policy disabled (%s): %s", jump_base.c_str(), error.what());
            }
        }

        if (bool_or(*this, "publish_status", false)) {
            status_publisher_ = create_publisher<msg::PolicyStatus>(
                channels_.front()->base + "/policy_status",
                rclcpp::QoS{rclcpp::KeepLast(1)}.transient_local().best_effort());
            const double rate = std::max(number_or(*this, "status_rate", 2.0), 0.1);
            status_timer_ = create_wall_timer(
                std::chrono::duration<double>(1.0 / rate), [this]() { publish_status(); });
        }
    }

private:
    struct Channel {
        std::string base;
        std::unique_ptr<PolicyModel> model;
        rclcpp::Publisher<msg::Action>::SharedPtr action_publisher;
        rclcpp::Subscription<msg::Observation>::SharedPtr observation_subscription;
        bool layout_mismatch_logged = false;
    };

    void add_channel_(const std::string& base, const std::string& model_path) {
        PolicyModel::Config config;
        config.path = model_path;
        if (config.path.empty())
            throw std::invalid_argument("policy_server: parameter 'rl_model_path' is required");
        if (config.path.front() != '/')
            config.path =
                ament_index_cpp::get_package_share_directory("rmcs_rl") + "/" + config.path;
        config.input_name = string_or(*this, "input_name", "obs");
        config.history_input_name = string_or(*this, "history_input_name", "");
        config.output_name = string_or(*this, "output_name", "actions");
        config.normalization_from_metadata = bool_or(*this, "normalization_from_metadata", true);
        if (const double clip = number_or(*this, "obs_clip", -1.0); clip >= 0.0)
            config.obs_clip = clip;
        if (const double clip = number_or(*this, "action_clip", -1.0); clip >= 0.0)
            config.action_clip = clip;

        auto channel = std::make_unique<Channel>();
        channel->base = base;
        channel->model = std::make_unique<PolicyModel>(config);
        channel->action_publisher = create_publisher<msg::Action>(
            base + "/action", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort());
        channel->observation_subscription = create_subscription<msg::Observation>(
            base + "/obs", rclcpp::QoS{rclcpp::KeepLast(1)}.best_effort(),
            [this, entry = channel.get()](
                msg::Observation::UniquePtr message) { on_observation(*entry, std::move(message)); });

        const auto& info = channel->model->info();
        RCLCPP_INFO(get_logger(), "[%s] policy loaded: %s", base.c_str(), info.path.c_str());
        RCLCPP_INFO(
            get_logger(), "[%s] model_id=%s layout_hash=%s version=%s obs=%zu action=%zu",
            base.c_str(), hex16(info.model_id).c_str(), hex16(info.layout_hash).c_str(),
            info.version.c_str(), info.obs_size, info.action_size);
        RCLCPP_INFO(
            get_logger(), "[%s] input contract: frame=%zu history=%zu (%s)", base.c_str(),
            info.frame_size, info.history_length,
            info.frame_size * info.history_length == info.obs_size ? "ok" : "MISMATCH");
        RCLCPP_INFO(get_logger(), "[%s] obs signature: %s", base.c_str(),
                    info.obs_signature.c_str());
        RCLCPP_INFO(get_logger(), "[%s] action signature: %s", base.c_str(),
                    info.actions_signature.c_str());
        RCLCPP_INFO(get_logger(), "[%s] waiting for obs on %s/obs", base.c_str(), base.c_str());
        channels_.push_back(std::move(channel));
    }

    void on_observation(Channel& channel, msg::Observation::UniquePtr message) {
        const auto& info = channel.model->info();
        if (message->layout_hash != info.layout_hash) {
            if (!channel.layout_mismatch_logged) {
                channel.layout_mismatch_logged = true;
                RCLCPP_ERROR(
                    get_logger(), "[%s] layout_hash mismatch: bridge=%s model=%s; refusing actions",
                    channel.base.c_str(), hex16(message->layout_hash).c_str(),
                    hex16(info.layout_hash).c_str());
            }
            return;
        }

        const auto started = std::chrono::steady_clock::now();
        msg::Action action;
        action.action.resize(info.action_size);
        std::string error;
        if (!channel.model->run(message->obs, action.action, error)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000, "[%s] policy rejected observation: %s",
                channel.base.c_str(), error.c_str());
            return;
        }
        action.header.stamp = get_clock()->now();
        action.obs_seq = message->obs_seq;
        action.layout_hash = info.layout_hash;
        action.model_id = info.model_id;
        channel.action_publisher->publish(action);
        const auto elapsed =
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started);
        record_inference_time(elapsed.count());
    }

    void record_inference_time(double microseconds) {
        inference_window_[inference_window_cursor_] = microseconds;
        inference_window_cursor_ = (inference_window_cursor_ + 1) % inference_window_.size();
        inference_window_count_ = std::min(inference_window_count_ + 1, inference_window_.size());
    }

    double percentile(double fraction) const {
        if (inference_window_count_ == 0)
            return 0.0;
        std::vector<double> samples(
            inference_window_.begin(), inference_window_.begin() + inference_window_count_);
        std::sort(samples.begin(), samples.end());
        const auto index =
            static_cast<std::size_t>(fraction * static_cast<double>(samples.size() - 1) + 0.5);
        return samples[std::min(index, samples.size() - 1)];
    }

    void publish_status() {
        if (channels_.empty())
            return;
        const auto& info = channels_.front()->model->info();
        msg::PolicyStatus status;
        status.header.stamp = get_clock()->now();
        status.model_name = info.path;
        status.model_id = info.model_id;
        status.policy_version = info.version;
        status.layout_hash = info.layout_hash;
        status.obs_size = static_cast<std::uint32_t>(info.obs_size);
        status.action_size = static_cast<std::uint32_t>(info.action_size);
        status.inference_p50_us = percentile(0.50);
        status.inference_p99_us = percentile(0.99);
        status_publisher_->publish(status);
    }

    std::vector<std::unique_ptr<Channel>> channels_;
    rclcpp::Publisher<msg::PolicyStatus>::SharedPtr status_publisher_;
    rclcpp::TimerBase::SharedPtr status_timer_;
    std::array<double, 256> inference_window_{};
    std::size_t inference_window_cursor_ = 0;
    std::size_t inference_window_count_ = 0;
};

} // namespace rmcs_rl

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<rmcs_rl::PolicyServer>();
        rclcpp::spin(node);
    } catch (const std::exception& error) {
        fprintf(stderr, "[Fatal] policy_server startup failed: %s\n", error.what());
        fflush(stderr);
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
