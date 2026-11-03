add_executable(query_benchmark query_benchmark.cpp)
target_link_libraries(query_benchmark PRIVATE mrl::cuda_depth_buffer)
target_compile_options(query_benchmark PRIVATE -Wall -Wextra -Wpedantic)

# The exact pixel patch against the point sample minus a footprint margin:
# accuracy of both against the finest answer the sensor gives, per pool.
add_executable(pixel_extent_study pixel_extent_study.cpp)
target_link_libraries(pixel_extent_study PRIVATE mrl::cuda_depth_buffer)
target_compile_options(pixel_extent_study PRIVATE -Wall -Wextra -Wpedantic)
