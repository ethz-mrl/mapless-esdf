#ifndef CUDA_DEPTH_BUFFER_ROS_DEPTH_BUFFER_NODE_HPP
#define CUDA_DEPTH_BUFFER_ROS_DEPTH_BUFFER_NODE_HPP

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <cuda_depth_buffer_msgs/srv/query_distances.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <memory>
#include <mrl/cuda_depth_buffer/buffer_params.hpp>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <mutex>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <vector>

namespace cuda_depth_buffer_ros {

/// Answers signed distance queries from a ring of posed depth images.
///
/// A component, so it can be loaded into the same process as the camera driver
/// and take its images without a copy or a serialization.
///
/// The buffer is fixed at construction and needs the image geometry to build,
/// which is not known until either the parameters give it or the first
/// CameraInfo arrives. Until then images are dropped and said so, once.
class DepthBufferNode : public rclcpp::Node {
 public:
  explicit DepthBufferNode(const rclcpp::NodeOptions& options);

 private:
  void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info);
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr& image);
  void onPose(const geometry_msgs::msg::PoseStamped::ConstSharedPtr& pose);
  void onPoseWithCovariance(
      const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr&
          pose);
  void onPathUpdate(const nav_msgs::msg::Path::ConstSharedPtr& path);
  void onQuery(
      const std::shared_ptr<
          cuda_depth_buffer_msgs::srv::QueryDistances::Request> request,
      std::shared_ptr<cuda_depth_buffer_msgs::srv::QueryDistances::Response>
          response);
  void publishSlice();

  /// Builds the buffer once the image geometry is known. Returns false while it
  /// still is not.
  bool ensureBuffer(uint16_t width, uint16_t height);

  /// The buffer, or null while there is none yet. Queries run in a reentrant
  /// group and the inputs that construct it do not, so the pointer itself is
  /// read under the lock; what it points at is thread safe and is deliberately
  /// used outside it.
  mrl::CudaDepthBuffer* buffer();

  /// Pose of the camera at `stamp`, from tf or from the last pose message,
  /// as x, y, z, qx, qy, qz, qw. Empty when it is not available.
  std::vector<float> lookupPose(const rclcpp::Time& stamp,
                                const std::string& camera_frame);

  mrl::BufferParams params_;
  mrl::CameraIntrinsics intrinsics_;
  std::unique_ptr<mrl::CudaDepthBuffer> buffer_;
  /// Guards the members above and the pose cache, not the buffer itself, which
  /// is thread safe and is deliberately left to overlap queries with inserts.
  std::mutex state_mutex_;
  bool intrinsics_known_ = false;
  bool warned_about_geometry_ = false;

  std::string reference_frame_;
  std::string pose_source_;
  double tf_timeout_s_ = 0.05;
  float truncation_radius_ = 3.0f;
  bool use_camera_info_ = true;

  geometry_msgs::msg::PoseStamped last_pose_;
  bool have_last_pose_ = false;

  double slice_height_ = 1.0;
  double slice_extent_ = 10.0;
  int slice_resolution_ = 128;
  bool slice_conservative_ = true;
  bool slice_colorize_ = true;
  /// The query grid, built on the first publish and reused after it. Only ever
  /// touched from the slice timer.
  std::vector<float> slice_points_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      pose_covariance_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr slice_pub_;
  rclcpp::Service<cuda_depth_buffer_msgs::srv::QueryDistances>::SharedPtr
      query_service_;
  rclcpp::TimerBase::SharedPtr slice_timer_;
  rclcpp::CallbackGroup::SharedPtr input_group_;
  rclcpp::CallbackGroup::SharedPtr query_group_;
};

}  // namespace cuda_depth_buffer_ros

#endif  // CUDA_DEPTH_BUFFER_ROS_DEPTH_BUFFER_NODE_HPP
