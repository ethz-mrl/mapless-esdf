#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>
#include <mrl/cuda_depth_buffer/buffer_params.hpp>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace mrl {

void QueryResults::resize(const uint32_t n) {
  num_queries = n;
  optimistic_distance.assign(n, std::numeric_limits<float>::quiet_NaN());
  optimistic_gradient.assign(3 * n, std::numeric_limits<float>::quiet_NaN());
  optimistic_closest.assign(3 * n, std::numeric_limits<float>::quiet_NaN());
  // Left as NaN unless the query asks for conservative handling.
  conservative_distance.assign(n, std::numeric_limits<float>::quiet_NaN());
  conservative_gradient.assign(3 * n, std::numeric_limits<float>::quiet_NaN());
  conservative_closest.assign(3 * n, std::numeric_limits<float>::quiet_NaN());
}

const char* toString(const PoolCompensation mode) {
  switch (mode) {
    case PoolCompensation::kNone:
      return "none";
    case PoolCompensation::kExact:
      return "exact";
    case PoolCompensation::kApproximate:
      return "approximate";
  }
  return "exact";
}

PoolCompensation poolCompensationFromString(const std::string& name) {
  if (name == "none") return PoolCompensation::kNone;
  if (name == "exact") return PoolCompensation::kExact;
  if (name == "approximate") return PoolCompensation::kApproximate;
  throw std::invalid_argument(
      "pool_compensation must be none, exact or approximate, not '" + name +
      "'");
}

namespace {

template <typename T>
void readIfPresent(const YAML::Node& node, const char* key, T& value) {
  if (node[key]) value = node[key].as<T>();
}

BufferParams fromNode(const YAML::Node& node) {
  BufferParams params;
  readIfPresent(node, "num_frames", params.num_frames);
  readIfPresent(node, "use_keyframing", params.use_keyframing);
  readIfPresent(node, "min_pool_size", params.min_pool_size);
  if (node["pool_compensation"]) {
    params.pool_compensation =
        poolCompensationFromString(node["pool_compensation"].as<std::string>());
  } else if (node["pixel_extent"]) {
    // The pre-enum switch: true was the exact patch, false the point sample.
    params.pool_compensation = node["pixel_extent"].as<bool>()
                                   ? PoolCompensation::kExact
                                   : PoolCompensation::kNone;
  }
  readIfPresent(node, "depth_scale_factor", params.depth_scale_factor);
  readIfPresent(node, "invalid_is_farplane", params.invalid_is_farplane);
  readIfPresent(node, "robot_radius", params.robot_radius);
  readIfPresent(node,
                "keyframe_dist_threshold",
                params.keyframe_dist_threshold);
  readIfPresent(node,
                "keyframe_angle_threshold",
                params.keyframe_angle_threshold);
  readIfPresent(node,
                "keyframe_time_threshold",
                params.keyframe_time_threshold);
  readIfPresent(node,
                "pose_covariance_confidence",
                params.pose_covariance_confidence);
  return params;
}

}  // namespace

BufferParams BufferParams::fromYamlFile(const std::string& filename) {
  try {
    return fromNode(YAML::LoadFile(filename));
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("cannot parse buffer params from " + filename +
                             ": " + e.what());
  }
}

BufferParams BufferParams::fromYaml(const std::string& yaml) {
  try {
    return fromNode(YAML::Load(yaml));
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(std::string("cannot parse buffer params: ") +
                             e.what());
  }
}

std::string BufferParams::toYaml() const {
  std::ostringstream out;
  out << "num_frames: " << num_frames << "\n"
      << "use_keyframing: " << (use_keyframing ? "true" : "false") << "\n"
      << "min_pool_size: " << min_pool_size << "\n"
      << "pool_compensation: " << toString(pool_compensation) << "\n"
      << "depth_scale_factor: " << depth_scale_factor << "\n"
      << "invalid_is_farplane: " << (invalid_is_farplane ? "true" : "false")
      << "\n"
      << "robot_radius: " << robot_radius << "\n"
      << "keyframe_dist_threshold: " << keyframe_dist_threshold << "\n"
      << "keyframe_angle_threshold: " << keyframe_angle_threshold << "\n"
      << "keyframe_time_threshold: " << keyframe_time_threshold << "\n"
      << "pose_covariance_confidence: " << pose_covariance_confidence << "\n";
  return out.str();
}

std::ostream& operator<<(std::ostream& os, const BufferParams& params) {
  return os << params.toYaml();
}

}  // namespace mrl
