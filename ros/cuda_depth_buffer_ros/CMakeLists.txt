cmake_minimum_required(VERSION 3.16)
project(cuda_depth_buffer_ros)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(cuda_depth_buffer REQUIRED)
find_package(cuda_depth_buffer_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

set(dependencies
  cuda_depth_buffer_msgs
  geometry_msgs
  nav_msgs
  rclcpp
  rclcpp_components
  sensor_msgs
  tf2_ros
)

add_library(depth_buffer_component SHARED src/depth_buffer_node.cpp)
target_include_directories(depth_buffer_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(depth_buffer_component PRIVATE -Wall -Wextra -Wpedantic)
# Imported targets rather than ament_target_dependencies, which uses the plain
# target_link_libraries signature and cannot be mixed with the keyword one.
target_link_libraries(depth_buffer_component PUBLIC
  mrl::cuda_depth_buffer
  rclcpp::rclcpp
  rclcpp_components::component
  tf2_ros::tf2_ros
  ${cuda_depth_buffer_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${sensor_msgs_TARGETS})
rclcpp_components_register_node(depth_buffer_component
  PLUGIN "cuda_depth_buffer_ros::DepthBufferNode"
  EXECUTABLE depth_buffer_component_node)

add_executable(depth_buffer_node src/depth_buffer_node_main.cpp)
target_link_libraries(depth_buffer_node PRIVATE depth_buffer_component)
target_compile_options(depth_buffer_node PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS depth_buffer_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS depth_buffer_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY test DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(depth_buffer_component)
ament_export_dependencies(${dependencies} cuda_depth_buffer)
ament_package()
