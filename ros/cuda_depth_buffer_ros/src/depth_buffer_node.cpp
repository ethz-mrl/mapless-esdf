#include "cuda_depth_buffer_ros/depth_buffer_node.hpp"

#include <tf2_ros/create_timer_ros.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <rclcpp/create_timer.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace cuda_depth_buffer_ros {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

/// Depth in metres as float32, whatever the image says it is.
///
/// 32FC1 is passed through by pointer; 16UC1 has to be widened, and the scale
/// that turns it into metres is the buffer's depth_scale_factor, not something
/// applied here, so the raw counts go in.
bool depthAsFloat(const sensor_msgs::msg::Image& image,
                  const float** data,
                  std::vector<float>& storage) {
  const size_t pixels = static_cast<size_t>(image.width) * image.height;
  if (image.encoding == "32FC1") {
    if (image.data.size() < pixels * sizeof(float)) return false;
    *data = reinterpret_cast<const float*>(image.data.data());
    return true;
  }
  if (image.encoding == "16UC1") {
    if (image.data.size() < pixels * sizeof(uint16_t)) return false;
    const uint16_t* raw = reinterpret_cast<const uint16_t*>(image.data.data());
    storage.resize(pixels);
    std::copy_n(raw, pixels, storage.begin());
    *data = storage.data();
    return true;
  }
  return false;
}

/// x, y, z, qx, qy, qz, qw, the order the buffer takes.
std::vector<float> flatten(const geometry_msgs::msg::Pose& pose) {
  return {static_cast<float>(pose.position.x),
          static_cast<float>(pose.position.y),
          static_cast<float>(pose.position.z),
          static_cast<float>(pose.orientation.x),
          static_cast<float>(pose.orientation.y),
          static_cast<float>(pose.orientation.z),
          static_cast<float>(pose.orientation.w)};
}

int64_t stampToNanoseconds(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

/// Matplotlib's "seismic" evaluated at t in [0, 1]: blue, white, red.
///
/// RViz colours a scalar by interpolating between two colours, which cannot put
/// white in the middle, and the middle is where the answer changes sign. So the
/// colour is computed here and shipped as an rgb field; the raw distance rides
/// along in `intensity` for anyone who wants to map it differently.
void seismic(float t, uint8_t& red, uint8_t& green, uint8_t& blue) {
  struct Knot {
    float t, r, g, b;
  };
  static const Knot knots[5] = {{0.00f, 0.0f, 0.0f, 0.3f},
                                {0.25f, 0.0f, 0.0f, 1.0f},
                                {0.50f, 1.0f, 1.0f, 1.0f},
                                {0.75f, 1.0f, 0.0f, 0.0f},
                                {1.00f, 0.5f, 0.0f, 0.0f}};
  t = std::min(1.0f, std::max(0.0f, t));
  int i = 0;
  while (i < 3 && t > knots[i + 1].t) ++i;
  const float span = knots[i + 1].t - knots[i].t;
  const float u = span > 0.0f ? (t - knots[i].t) / span : 0.0f;
  const auto lerp = [u](float a, float b) {
    return static_cast<uint8_t>(255.0f * (a + u * (b - a)) + 0.5f);
  };
  red = lerp(knots[i].r, knots[i + 1].r);
  green = lerp(knots[i].g, knots[i + 1].g);
  blue = lerp(knots[i].b, knots[i + 1].b);
}

}  // namespace

DepthBufferNode::DepthBufferNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("cuda_depth_buffer", options) {
  // Every knob of the library is a parameter, so a deployment is a yaml file
  // and nothing here needs recompiling.
  params_.num_frames = static_cast<size_t>(
      declare_parameter<int>("num_frames",
                             static_cast<int>(params_.num_frames)));
  params_.use_keyframing =
      declare_parameter<bool>("use_keyframing", params_.use_keyframing);
  params_.min_pool_size = static_cast<uint16_t>(
      declare_parameter<int>("min_pool_size", params_.min_pool_size));
  params_.depth_scale_factor =
      static_cast<float>(declare_parameter<double>("depth_scale_factor",
                                                   params_.depth_scale_factor));
  params_.invalid_is_farplane =
      declare_parameter<bool>("invalid_is_farplane",
                              params_.invalid_is_farplane);
  params_.robot_radius =
      static_cast<float>(declare_parameter<double>("robot_radius", 0.0));
  const std::vector<double> robot_center =
      declare_parameter<std::vector<double>>(
          "robot_center_in_camera",
          std::vector<double>{0.0, 0.0, 0.0});
  if (robot_center.size() == 3) {
    for (int i = 0; i < 3; ++i) {
      params_.robot_center_in_camera[i] = static_cast<float>(robot_center[i]);
    }
  }
  params_.keyframe_dist_threshold = static_cast<float>(
      declare_parameter<double>("keyframe_dist_threshold",
                                params_.keyframe_dist_threshold));
  params_.keyframe_angle_threshold = static_cast<float>(
      declare_parameter<double>("keyframe_angle_threshold",
                                params_.keyframe_angle_threshold));
  params_.keyframe_time_threshold = static_cast<float>(
      declare_parameter<double>("keyframe_time_threshold",
                                params_.keyframe_time_threshold));
  params_.pose_covariance_confidence = static_cast<float>(
      declare_parameter<double>("pose_covariance_confidence",
                                params_.pose_covariance_confidence));

  reference_frame_ = declare_parameter<std::string>("reference_frame", "odom");
  pose_source_ = declare_parameter<std::string>("pose_source", "topic");
  tf_timeout_s_ = declare_parameter<double>("tf_timeout", 0.05);
  truncation_radius_ =
      static_cast<float>(declare_parameter<double>("truncation_radius", 3.0));
  use_camera_info_ = declare_parameter<bool>("use_camera_info", true);

  // The image geometry can also come from parameters, for a sensor that
  // publishes no CameraInfo or a bag that lost it.
  intrinsics_.fx = static_cast<float>(declare_parameter<double>("fx", 0.0));
  intrinsics_.fy = static_cast<float>(declare_parameter<double>("fy", 0.0));
  intrinsics_.cx = static_cast<float>(declare_parameter<double>("cx", 0.0));
  intrinsics_.cy = static_cast<float>(declare_parameter<double>("cy", 0.0));
  intrinsics_.width = static_cast<uint16_t>(declare_parameter<int>("width", 0));
  intrinsics_.height =
      static_cast<uint16_t>(declare_parameter<int>("height", 0));
  intrinsics_.z_near =
      static_cast<float>(declare_parameter<double>("z_near", 0.2));
  intrinsics_.z_far =
      static_cast<float>(declare_parameter<double>("z_far", 20.0));
  intrinsics_known_ = intrinsics_.fx > 0.0f && intrinsics_.fy > 0.0f &&
                      intrinsics_.width > 0 && intrinsics_.height > 0;

  slice_height_ = declare_parameter<double>("slice.height", 1.0);
  slice_extent_ = declare_parameter<double>("slice.extent", 10.0);
  slice_resolution_ = declare_parameter<int>("slice.resolution", 128);
  slice_conservative_ = declare_parameter<bool>("slice.conservative", true);
  slice_colorize_ = declare_parameter<bool>("slice.colorize", true);
  const double slice_rate = declare_parameter<double>("slice.rate", 2.0);

  // Inputs mutate node state and are serialized against each other. Queries do
  // not, and the buffer lets them run concurrently with an insert and with one
  // another, which is the whole point of several planning threads.
  input_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  query_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  rclcpp::SubscriptionOptions input_options;
  input_options.callback_group = input_group_;
  const auto sensor_qos = rclcpp::SensorDataQoS();

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "~/depth",
      sensor_qos,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr image) {
        onImage(image);
      },
      input_options);

  if (use_camera_info_) {
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "~/camera_info",
        sensor_qos,
        [this](const sensor_msgs::msg::CameraInfo::ConstSharedPtr info) {
          onCameraInfo(info);
        },
        input_options);
  }

  if (pose_source_ == "tf") {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_buffer_->setCreateTimerInterface(
        std::make_shared<tf2_ros::CreateTimerROS>(get_node_base_interface(),
                                                  get_node_timers_interface()));
    tf_listener_ =
        std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
  } else {
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "~/pose",
        sensor_qos,
        [this](const geometry_msgs::msg::PoseStamped::ConstSharedPtr pose) {
          onPose(pose);
        },
        input_options);
    pose_covariance_sub_ =
        create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "~/pose_with_covariance",
            sensor_qos,
            [this](const geometry_msgs::msg::PoseWithCovarianceStamped::
                       ConstSharedPtr pose) { onPoseWithCovariance(pose); },
            input_options);
  }

  // A backend that closes a loop revises many poses at once, and the buffer
  // takes them by timestamp, newest or oldest alike.
  path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "~/pose_updates",
      rclcpp::QoS(1),
      [this](const nav_msgs::msg::Path::ConstSharedPtr path) {
        onPathUpdate(path);
      },
      input_options);

  slice_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/esdf_slice",
                                                               rclcpp::QoS(1));

  query_service_ = create_service<cuda_depth_buffer_msgs::srv::QueryDistances>(
      "~/query",
      [this](
          const std::shared_ptr<
              cuda_depth_buffer_msgs::srv::QueryDistances::Request> request,
          std::shared_ptr<cuda_depth_buffer_msgs::srv::QueryDistances::Response>
              response) { onQuery(request, response); },
      rclcpp::ServicesQoS(),
      query_group_);

  if (slice_rate > 0.0) {
    // On the node clock, not the wall clock, so that with use_sim_time the
    // slice follows the bag: pause the playback and the slice stops with it
    // instead of recomputing the same answer forever.
    slice_timer_ = rclcpp::create_timer(
        this,
        get_clock(),
        rclcpp::Duration::from_seconds(1.0 / slice_rate),
        [this]() { publishSlice(); },
        query_group_);
  }

  if (intrinsics_known_) ensureBuffer(intrinsics_.width, intrinsics_.height);
  RCLCPP_INFO(
      get_logger(),
      "reference frame '%s', poses from %s, %zu frames, truncation %.2f m",
      reference_frame_.c_str(),
      pose_source_.c_str(),
      params_.num_frames,
      static_cast<double>(truncation_radius_));
}

mrl::CudaDepthBuffer* DepthBufferNode::buffer() {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  return buffer_.get();
}

bool DepthBufferNode::ensureBuffer(uint16_t width, uint16_t height) {
  if (buffer_) return true;
  if (!intrinsics_known_ || width == 0 || height == 0) return false;
  intrinsics_.width = width;
  intrinsics_.height = height;
  try {
    buffer_ = std::make_unique<mrl::CudaDepthBuffer>(intrinsics_, params_);
    buffer_->warmup();
  } catch (const std::exception& error) {
    RCLCPP_ERROR(get_logger(),
                 "could not create the depth buffer: %s",
                 error.what());
    return false;
  }
  RCLCPP_INFO(get_logger(),
              "buffer ready: %ux%u, fx %.1f, fy %.1f, %.2f to %.2f m",
              width,
              height,
              static_cast<double>(intrinsics_.fx),
              static_cast<double>(intrinsics_.fy),
              static_cast<double>(intrinsics_.z_near),
              static_cast<double>(intrinsics_.z_far));
  return true;
}

void DepthBufferNode::onCameraInfo(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info) {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  if (buffer_) return;  // the geometry is fixed for the life of the buffer
  intrinsics_.fx = static_cast<float>(info->k[0]);
  intrinsics_.fy = static_cast<float>(info->k[4]);
  intrinsics_.cx = static_cast<float>(info->k[2]);
  intrinsics_.cy = static_cast<float>(info->k[5]);
  intrinsics_.width = static_cast<uint16_t>(info->width);
  intrinsics_.height = static_cast<uint16_t>(info->height);
  intrinsics_known_ = intrinsics_.fx > 0.0f && intrinsics_.fy > 0.0f &&
                      intrinsics_.width > 0 && intrinsics_.height > 0;
  if (intrinsics_known_) ensureBuffer(intrinsics_.width, intrinsics_.height);
}

void DepthBufferNode::onPose(
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose) {
  const std::lock_guard<std::mutex> lock(state_mutex_);
  last_pose_ = *pose;
  have_last_pose_ = true;
}

void DepthBufferNode::onPoseWithCovariance(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr& pose) {
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    last_pose_.header = pose->header;
    last_pose_.pose = pose->pose.pose;
    have_last_pose_ = true;
  }
  // A covariance for a frame already inserted revises it; for one not inserted
  // yet there is nothing to attach it to, and the next image carries the pose.
  if (mrl::CudaDepthBuffer* depth_buffer = buffer()) {
    depth_buffer->updatePoseAndCovarianceByTimestamp(
        stampToNanoseconds(pose->header.stamp),
        nullptr,
        pose->pose.covariance.data());
  }
}

void DepthBufferNode::onPathUpdate(
    const nav_msgs::msg::Path::ConstSharedPtr& path) {
  mrl::CudaDepthBuffer* depth_buffer = buffer();
  if (!depth_buffer) return;
  size_t revised = 0;
  for (const geometry_msgs::msg::PoseStamped& pose : path->poses) {
    const std::vector<float> flat = flatten(pose.pose);
    if (depth_buffer->updatePoseAndCovarianceByTimestamp(
            stampToNanoseconds(pose.header.stamp),
            flat.data(),
            nullptr)) {
      ++revised;
    }
  }
  RCLCPP_DEBUG(get_logger(),
               "revised %zu of %zu poses",
               revised,
               path->poses.size());
}

std::vector<float> DepthBufferNode::lookupPose(
    const rclcpp::Time& stamp,
    const std::string& camera_frame) {
  if (pose_source_ == "tf") {
    if (!tf_buffer_) return {};
    try {
      const geometry_msgs::msg::TransformStamped transform =
          tf_buffer_->lookupTransform(
              reference_frame_,
              camera_frame,
              stamp,
              rclcpp::Duration::from_seconds(tf_timeout_s_));
      geometry_msgs::msg::Pose pose;
      pose.position.x = transform.transform.translation.x;
      pose.position.y = transform.transform.translation.y;
      pose.position.z = transform.transform.translation.z;
      pose.orientation = transform.transform.rotation;
      return flatten(pose);
    } catch (const tf2::TransformException& error) {
      RCLCPP_WARN_THROTTLE(get_logger(),
                           *get_clock(),
                           2000,
                           "no transform %s -> %s: %s",
                           reference_frame_.c_str(),
                           camera_frame.c_str(),
                           error.what());
      return {};
    }
  }
  const std::lock_guard<std::mutex> lock(state_mutex_);
  if (!have_last_pose_) return {};
  return flatten(last_pose_.pose);
}

void DepthBufferNode::onImage(
    const sensor_msgs::msg::Image::ConstSharedPtr& image) {
  {
    const std::lock_guard<std::mutex> lock(state_mutex_);
    if (!ensureBuffer(static_cast<uint16_t>(image->width),
                      static_cast<uint16_t>(image->height))) {
      if (!warned_about_geometry_) {
        warned_about_geometry_ = true;
        RCLCPP_WARN(get_logger(),
                    "dropping images until the geometry is known: publish a "
                    "CameraInfo or set the fx, fy, cx, cy parameters");
      }
      return;
    }
  }

  // The buffer's geometry is fixed, so a stream that changes resolution has to
  // restart the node rather than silently read past the end of a row.
  if (image->width != intrinsics_.width ||
      image->height != intrinsics_.height) {
    RCLCPP_ERROR_THROTTLE(get_logger(),
                          *get_clock(),
                          5000,
                          "image is %ux%u but the buffer is %ux%u",
                          image->width,
                          image->height,
                          intrinsics_.width,
                          intrinsics_.height);
    return;
  }

  const float* depth = nullptr;
  std::vector<float> widened;
  if (!depthAsFloat(*image, &depth, widened)) {
    RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        5000,
        "unsupported depth encoding '%s', want 32FC1 or 16UC1",
        image->encoding.c_str());
    return;
  }

  const rclcpp::Time stamp(image->header.stamp);
  const std::vector<float> pose = lookupPose(stamp, image->header.frame_id);
  if (pose.empty()) return;

  mrl::CudaDepthBuffer* depth_buffer = buffer();
  if (!depth_buffer) return;
  const int64_t stamp_ns = stampToNanoseconds(image->header.stamp);
  const bool keyframe = !params_.use_keyframing ||
                        depth_buffer->isKeyframe(pose.data(), stamp_ns);
  depth_buffer->setPosedDepthImage(depth, pose.data(), stamp_ns, keyframe);
}

void DepthBufferNode::onQuery(
    const std::shared_ptr<cuda_depth_buffer_msgs::srv::QueryDistances::Request>
        request,
    std::shared_ptr<cuda_depth_buffer_msgs::srv::QueryDistances::Response>
        response) {
  mrl::CudaDepthBuffer* depth_buffer = buffer();
  if (!depth_buffer) {
    response->success = false;
    response->message = "no depth image has been received yet";
    return;
  }
  if (request->points.empty() || request->points.size() % 3 != 0) {
    response->success = false;
    response->message =
        "points must be xyz interleaved, so a multiple of three";
    return;
  }

  const uint32_t count = static_cast<uint32_t>(request->points.size() / 3);
  const float truncation = request->truncation_radius > 0.0f
                               ? request->truncation_radius
                               : truncation_radius_;
  mrl::QueryResults results;
  try {
    depth_buffer->query(request->points.data(),
                        count,
                        request->conservative,
                        results,
                        truncation);
  } catch (const std::exception& error) {
    response->success = false;
    response->message = error.what();
    return;
  }

  response->success = true;
  response->optimistic_distance = results.optimistic_distance;
  response->optimistic_gradient = results.optimistic_gradient;
  response->optimistic_closest = results.optimistic_closest;
  response->conservative_distance = results.conservative_distance;
  response->conservative_gradient = results.conservative_gradient;
  response->conservative_closest = results.conservative_closest;
}

void DepthBufferNode::publishSlice() {
  // On demand: computing and shipping a slice nobody is looking at would take
  // the GPU away from the queries that are actually steering something.
  if (slice_pub_->get_subscription_count() == 0) return;
  mrl::CudaDepthBuffer* depth_buffer = buffer();
  if (!depth_buffer) return;

  // Built once: extent, resolution and height are all fixed after construction.
  if (slice_points_.empty()) {
    const int resolution = std::max(2, slice_resolution_);
    const float step =
        static_cast<float>(2.0 * slice_extent_ / (resolution - 1));
    slice_points_.resize(static_cast<size_t>(resolution) * resolution * 3);
    for (int row = 0; row < resolution; ++row) {
      for (int col = 0; col < resolution; ++col) {
        const size_t i = (static_cast<size_t>(row) * resolution + col) * 3;
        slice_points_[i + 0] = static_cast<float>(-slice_extent_) + col * step;
        slice_points_[i + 1] = static_cast<float>(-slice_extent_) + row * step;
        slice_points_[i + 2] = static_cast<float>(slice_height_);
      }
    }
  }
  const std::vector<float>& points = slice_points_;

  mrl::QueryResults results;
  try {
    depth_buffer->query(points.data(),
                        static_cast<uint32_t>(points.size() / 3),
                        slice_conservative_,
                        results,
                        truncation_radius_);
  } catch (const std::exception& error) {
    RCLCPP_ERROR_THROTTLE(get_logger(),
                          *get_clock(),
                          5000,
                          "slice query failed: %s",
                          error.what());
    return;
  }
  const std::vector<float>& distance = slice_conservative_
                                           ? results.conservative_distance
                                           : results.optimistic_distance;

  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.frame_id = reference_frame_;
  cloud.header.stamp = now();
  cloud.height = 1;
  cloud.width = static_cast<uint32_t>(distance.size());
  cloud.is_dense = false;
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  if (slice_colorize_) {
    modifier.setPointCloud2Fields(5,
                                  "x",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "y",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "z",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "intensity",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "rgb",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32);
  } else {
    modifier.setPointCloud2Fields(4,
                                  "x",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "y",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "z",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32,
                                  "intensity",
                                  1,
                                  sensor_msgs::msg::PointField::FLOAT32);
  }
  modifier.resize(distance.size());
  sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
  sensor_msgs::PointCloud2Iterator<float> out_intensity(cloud, "intensity");
  for (size_t i = 0; i < distance.size();
       ++i, ++out_x, ++out_y, ++out_z, ++out_intensity) {
    *out_x = points[3 * i + 0];
    *out_y = points[3 * i + 1];
    *out_z = points[3 * i + 2];
    *out_intensity = distance[i];
  }
  if (slice_colorize_) {
    // Symmetric about zero and scaled to the truncation radius, so a colour is
    // a magnitude and white is always the boundary of unknown space.
    sensor_msgs::PointCloud2Iterator<uint8_t> out_rgb(cloud, "rgb");
    for (size_t i = 0; i < distance.size(); ++i, ++out_rgb) {
      const float d = std::isfinite(distance[i]) ? distance[i] : 0.0f;
      const float t =
          0.5f *
          (1.0f + std::min(1.0f, std::max(-1.0f, d / truncation_radius_)));
      // PointCloud2Iterator<uint8_t> on an rgb field walks b, g, r, a.
      seismic(t, out_rgb[2], out_rgb[1], out_rgb[0]);
      out_rgb[3] = 255;
    }
  }
  slice_pub_->publish(std::move(cloud));
}

}  // namespace cuda_depth_buffer_ros

RCLCPP_COMPONENTS_REGISTER_NODE(cuda_depth_buffer_ros::DepthBufferNode)
