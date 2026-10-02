#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <rmcs_rl/rl_bridge/action_channel.hpp>
#include <rmcs_rl/rl_bridge/observation_history.hpp>
#include <rmcs_rl/rl_bridge/observation_timeline.hpp>

#include "policy/policy_model.hpp"

namespace rmcs_rl {
namespace {

TEST(ObservationHistory, PreservesFrameSizeOrderAndReset) {
    ObservationHistory history{2, 3};
    EXPECT_EQ(history.append(std::array{1.0, 2.0}), (std::vector{1., 2., 1., 2., 1., 2.}));
    EXPECT_EQ(history.append(std::array{3.0, 4.0}), (std::vector{1., 2., 1., 2., 3., 4.}));
    EXPECT_EQ(history.append(std::array{5.0, 6.0}), (std::vector{1., 2., 3., 4., 5., 6.}));
    history.reset();
    EXPECT_EQ(history.append(std::array{7.0, 8.0}), (std::vector{7., 8., 7., 8., 7., 8.}));
    EXPECT_THROW(history.append(std::array{1.0}), std::invalid_argument);
}

TEST(ObservationTimeline, RejectsPreResetAndOlderActions) {
    using namespace std::chrono_literals;
    const ObservationTimeline::Clock::time_point start{};
    ObservationTimeline timeline;
    EXPECT_TRUE(timeline.due(start, 20ms));
    EXPECT_TRUE(std::isnan(timeline.age(0, start)));
    const auto first = timeline.publish(start);
    EXPECT_FALSE(timeline.due(start + 19ms, 20ms));
    EXPECT_TRUE(timeline.due(start + 20ms, 20ms));
    const auto second = timeline.publish(start + 20ms);
    EXPECT_DOUBLE_EQ(timeline.age(first, start + 30ms), 0.03);
    EXPECT_DOUBLE_EQ(timeline.age(second, start + 30ms), 0.01);
    timeline.reset();
    EXPECT_TRUE(std::isnan(timeline.age(second, start + 31ms)));
    const auto third = timeline.publish(start + 31ms);
    EXPECT_GT(third, second);
    EXPECT_TRUE(std::isnan(timeline.age(second, start + 32ms)));
    EXPECT_DOUBLE_EQ(timeline.age(third, start + 32ms), 0.001);
    timeline.publish(start + 51ms);
    EXPECT_TRUE(std::isnan(timeline.age(second, start + 52ms)));
}

TEST(ActionChannel, ConcurrentSnapshotsNeverMixFrames) {
    ActionChannel channel;
    channel.resize(6);
    ActionSnapshot snapshot;
    EXPECT_FALSE(channel.try_read(snapshot));
    std::atomic<bool> finished = false;
    std::jthread writer{[&] {
        msg::Action message;
        message.action.resize(6);
        for (std::uint64_t sequence = 1; sequence <= 10000; ++sequence) {
            message.obs_seq = message.model_id = message.layout_hash = sequence;
            std::fill(message.action.begin(), message.action.end(), static_cast<double>(sequence));
            channel.store(message);
        }
        finished = true;
    }};
    do {
        if (!channel.try_read(snapshot))
            continue;
        EXPECT_EQ(snapshot.model_id, snapshot.obs_seq);
        EXPECT_EQ(snapshot.layout_hash, snapshot.obs_seq);
        for (const double action : snapshot.action)
            EXPECT_EQ(action, static_cast<double>(snapshot.obs_seq));
    } while (!finished);
    writer.join();
    ASSERT_TRUE(channel.try_read(snapshot));
    EXPECT_EQ(snapshot.obs_seq, 10000u);
}

TEST(ActionChannel, KeepsPreviousSnapshotWhenWriterOwnsLock) {
    ActionChannel channel;
    channel.resize(1);

    msg::Action message;
    message.action = {1.0};
    message.obs_seq = message.layout_hash = message.model_id = 1;
    channel.store(message);

    ActionSnapshot snapshot;
    ASSERT_TRUE(channel.try_read(snapshot));
    EXPECT_EQ(snapshot.action, (std::vector{1.0}));

    // A second read after the first successful read is allowed to retain the
    // caller's snapshot if a concurrent store briefly owns the mutex. The
    // channel's public API does not expose the mutex, so this loop exercises
    // the same boundary without adding a production-only test hook.
    std::atomic<bool> finished = false;
    std::jthread writer{[&] {
        for (std::uint64_t sequence = 2; sequence <= 10000; ++sequence) {
            message.obs_seq = message.layout_hash = message.model_id = sequence;
            message.action[0] = static_cast<double>(sequence);
            channel.store(message);
        }
        finished = true;
    }};

    while (!finished) {
        ASSERT_TRUE(channel.try_read(snapshot));
        ASSERT_EQ(snapshot.action.size(), 1u);
        EXPECT_EQ(snapshot.obs_seq, snapshot.model_id);
        EXPECT_EQ(snapshot.action.front(), static_cast<double>(snapshot.obs_seq));
    }
    writer.join();
}

TEST(OnnxRuntime, DualInputFeedsHistoryAndNewestFrame) {
    OnnxRuntime runtime{RMCS_RL_DUAL_FIXTURE, "obs", "obs_history", "actions"};
    ASSERT_TRUE(runtime.has_history_input());
    EXPECT_EQ(runtime.input_size(), 125u);
    EXPECT_EQ(runtime.frame_size(), 25u);
    EXPECT_EQ(runtime.history_length(), 5u);
    EXPECT_EQ(runtime.output_size(), 150u);

    std::vector<float> input(runtime.input_size());
    for (std::size_t i = 0; i < input.size(); ++i)
        input[i] = static_cast<float>(i);
    std::vector<float> output(runtime.output_size());
    ASSERT_NO_THROW(runtime.run(input, output));
    // Fixture is Concat(newest_frame, full_history); the frame is the trailing slice.
    EXPECT_FLOAT_EQ(output.front(), 100.0f);
    EXPECT_FLOAT_EQ(output[24], 124.0f);
    EXPECT_FLOAT_EQ(output[25], 0.0f);
    EXPECT_FLOAT_EQ(output.back(), 124.0f);
}

TEST(OnnxRuntime, RejectsHistoryInputForSingleInputModel) {
    EXPECT_THROW(
        (OnnxRuntime{RMCS_RL_IDENTITY_FIXTURE, "obs", "obs_history", "actions"}),
        std::invalid_argument);
}

TEST(PolicyModel, NormalizesClipsAndRejectsInvalidFrames) {
    PolicyModel::Config config;
    config.path = RMCS_RL_IDENTITY_FIXTURE;
    PolicyModel model{config};
    ASSERT_EQ(model.info().obs_size, 2u);
    ASSERT_EQ(model.info().action_size, 2u);
    std::array<double, 2> action{};
    std::string error;
    ASSERT_TRUE(model.run(std::array{3.0, 6.0}, action, error)) << error;
    EXPECT_EQ(action, (std::array{1.0, 1.0}));
    ASSERT_TRUE(model.run(std::array{9.0, -18.0}, action, error)) << error;
    EXPECT_EQ(action, (std::array{1.5, -1.5}));
    EXPECT_FALSE(model.run(std::array{1.0}, action, error));
    EXPECT_FALSE(
        model.run(std::array{1.0, std::numeric_limits<double>::quiet_NaN()}, action, error));
    ASSERT_TRUE(model.run(std::array{1.0, 2.0}, action, error)) << error;
    EXPECT_EQ(action, (std::array{0.0, 0.0}));
}

TEST(PolicyModel, ParameterClipsOverrideMetadata) {
    PolicyModel::Config config;
    config.path = RMCS_RL_IDENTITY_FIXTURE;
    config.normalization_from_metadata = false;
    config.obs_clip = 4.0;
    config.action_clip = 3.0;
    PolicyModel model{config};
    std::array<double, 2> action{};
    std::string error;
    ASSERT_TRUE(model.run(std::array{6.0, -6.0}, action, error)) << error;
    EXPECT_EQ(action, (std::array{3.0, -3.0}));
}

} // namespace
} // namespace rmcs_rl
#include <algorithm>
