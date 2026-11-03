find_package(GTest REQUIRED)

# Tests that touch the device carry the "gpu" label. Everything else is host
# arithmetic and runs anywhere, which is most of what is worth checking: the
# scene library, the bound soundness proofs and the pose covariance parity.
#   ctest -LE gpu

# One binary per test group, registered with ctest.
function(cuda_depth_buffer_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    mrl::cuda_depth_buffer GTest::gtest GTest::gtest_main)
  target_compile_options(${name} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
  find_package(Threads REQUIRED)
  target_link_libraries(${name} PRIVATE Threads::Threads)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

cuda_depth_buffer_add_test(test_buffer_params unit/test_buffer_params.cpp)
cuda_depth_buffer_add_test(test_scene unit/test_scene.cpp)

# Compiled as CUDA so the test can call the same __host__ __device__ math the
# query kernel runs, and needs the internal headers for it.
foreach(cuda_test test_pose_compensation_parity test_bounds)
  cuda_depth_buffer_add_test(${cuda_test} unit/${cuda_test}.cpp)
  set_source_files_properties(unit/${cuda_test}.cpp PROPERTIES LANGUAGE CUDA)
  target_include_directories(${cuda_test} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_options(${cuda_test} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEPTH_BUFFER_CUDA_OPTIONS}>)
endforeach()
cuda_depth_buffer_add_test(test_depth_buffer system/test_depth_buffer.cpp)
cuda_depth_buffer_add_test(test_concurrency system/test_concurrency.cpp)
set_tests_properties(test_depth_buffer test_concurrency PROPERTIES LABELS gpu)
