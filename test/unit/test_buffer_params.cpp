#include <gtest/gtest.h>

#include <mrl/cuda_depth_buffer/buffer_params.hpp>

namespace {

TEST(BufferParamsTest, UnspecifiedFieldsKeepTheirDefault) {
  const mrl::BufferParams defaults;
  const mrl::BufferParams parsed = mrl::BufferParams::fromYaml("num_frames: 7");

  EXPECT_EQ(parsed.num_frames, 7u);
  EXPECT_EQ(parsed.min_pool_size, defaults.min_pool_size);
  EXPECT_FLOAT_EQ(parsed.depth_scale_factor, defaults.depth_scale_factor);
  EXPECT_FLOAT_EQ(parsed.pose_covariance_confidence,
                  defaults.pose_covariance_confidence);
}

TEST(BufferParamsTest, RoundTripsThroughYaml) {
  mrl::BufferParams params;
  params.num_frames = 12;
  params.use_keyframing = true;
  params.min_pool_size = 6;
  params.depth_scale_factor = 0.001f;
  params.invalid_is_farplane = true;
  params.robot_radius = 0.35f;
  params.keyframe_dist_threshold = 0.75f;
  params.keyframe_angle_threshold = 0.2f;
  params.keyframe_time_threshold = 2.5f;
  params.pose_covariance_confidence = 2.0f;

  const mrl::BufferParams parsed = mrl::BufferParams::fromYaml(params.toYaml());

  EXPECT_EQ(parsed.num_frames, params.num_frames);
  EXPECT_EQ(parsed.use_keyframing, params.use_keyframing);
  EXPECT_EQ(parsed.min_pool_size, params.min_pool_size);
  EXPECT_FLOAT_EQ(parsed.depth_scale_factor, params.depth_scale_factor);
  EXPECT_EQ(parsed.invalid_is_farplane, params.invalid_is_farplane);
  EXPECT_FLOAT_EQ(parsed.robot_radius, params.robot_radius);
  EXPECT_FLOAT_EQ(parsed.keyframe_dist_threshold,
                  params.keyframe_dist_threshold);
  EXPECT_FLOAT_EQ(parsed.keyframe_angle_threshold,
                  params.keyframe_angle_threshold);
  EXPECT_FLOAT_EQ(parsed.keyframe_time_threshold,
                  params.keyframe_time_threshold);
  EXPECT_FLOAT_EQ(parsed.pose_covariance_confidence,
                  params.pose_covariance_confidence);
}

TEST(BufferParamsTest, RejectsMalformedYaml) {
  EXPECT_THROW(mrl::BufferParams::fromYamlFile("/nonexistent/params.yaml"),
               std::runtime_error);
}

}  // namespace
