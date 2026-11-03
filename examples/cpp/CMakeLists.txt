find_package(Threads REQUIRED)

function(cuda_depth_buffer_add_example name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE mrl::cuda_depth_buffer Threads::Threads)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
endfunction()

cuda_depth_buffer_add_example(concurrent_queries concurrent_queries.cpp)
