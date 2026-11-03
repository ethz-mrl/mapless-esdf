// Standalone entry point. The component is the interesting way to run this,
// composed into the camera driver's process so the depth image never leaves it,
// but a node of its own is easier to start with.

#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "cuda_depth_buffer_ros/depth_buffer_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  auto node = std::make_shared<cuda_depth_buffer_ros::DepthBufferNode>(options);
  // Several threads, because queries are meant to overlap each other and the
  // insert of the next image.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
