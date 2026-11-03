find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core src/bindings.cpp)
target_link_libraries(_core PRIVATE mrl::cuda_depth_buffer)
target_compile_options(_core PRIVATE -Wall -Wextra)

# Build the package next to its __init__.py so the build tree is importable as
# is, which is what the Python tests below run against.
set(cuda_depth_buffer_python_dir ${CMAKE_CURRENT_BINARY_DIR}/cuda_depth_buffer)
set_target_properties(_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${cuda_depth_buffer_python_dir})
add_custom_command(TARGET _core POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${CMAKE_CURRENT_SOURCE_DIR}/cuda_depth_buffer/__init__.py
          ${cuda_depth_buffer_python_dir}/__init__.py)

# Tagged so that a wheel installs the package alone; the C++ library, its
# headers and its CMake config belong to an ordinary install, not to
# site-packages.
install(TARGETS _core LIBRARY DESTINATION cuda_depth_buffer COMPONENT python)
install(DIRECTORY cuda_depth_buffer/ DESTINATION cuda_depth_buffer
        COMPONENT python FILES_MATCHING PATTERN "*.py")

if(CUDA_DEPTH_BUFFER_BUILD_TESTS)
  add_test(NAME test_python
           COMMAND ${Python_EXECUTABLE} -m pytest -q
                   ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  set_tests_properties(test_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
    LABELS gpu)
endif()
