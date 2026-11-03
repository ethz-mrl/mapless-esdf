cmake_minimum_required(VERSION 3.16)
project(cuda_depth_buffer_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  srv/QueryDistances.srv
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
