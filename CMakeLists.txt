cmake_minimum_required(VERSION 3.20)

# "native" compiles for the GPU in this machine, which is what a developer wants
# and the only thing nvcc can work out on its own. It needs CMake 3.24 and a
# device to look at, so a machine without one, a CI runner or a docker build,
# falls back to a list. Override for cross builds, e.g.
# -DCMAKE_CUDA_ARCHITECTURES=87 for a Jetson Orin.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CUDA_DEPTH_BUFFER_FALLBACK_ARCHITECTURES 75 80 86 87 89 120)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    execute_process(COMMAND nvidia-smi -L
                    RESULT_VARIABLE CUDA_DEPTH_BUFFER_GPU_PRESENT
                    OUTPUT_QUIET ERROR_QUIET)
  endif()
  if(CUDA_DEPTH_BUFFER_GPU_PRESENT EQUAL 0)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # An architecture this nvcc has never heard of is fatal, not ignored
    # ("nvcc fatal : Unsupported gpu architecture 'compute_120'"), and it is
    # the compiler test inside project() that dies, so the list has to be
    # filtered before it gets there. A Jetson is exactly this case: no
    # nvidia-smi to detect a device with, and a CUDA too old for Blackwell.
    find_program(CUDA_DEPTH_BUFFER_NVCC
                 NAMES nvcc
                 HINTS ENV CUDA_PATH ENV CUDA_HOME ENV CUDACXX
                 PATHS /usr/local/cuda/bin /usr/lib/nvidia-cuda-toolkit/bin)
    set(_cdb_nvcc_arch_text "")
    if(CUDA_DEPTH_BUFFER_NVCC)
      # --list-gpu-arch is CUDA 11.5 and newer; --help lists them further back.
      execute_process(COMMAND "${CUDA_DEPTH_BUFFER_NVCC}" --list-gpu-arch
                      OUTPUT_VARIABLE _cdb_nvcc_arch_text
                      RESULT_VARIABLE _cdb_nvcc_rc ERROR_QUIET)
      if(NOT _cdb_nvcc_rc EQUAL 0)
        execute_process(COMMAND "${CUDA_DEPTH_BUFFER_NVCC}" --help
                        OUTPUT_VARIABLE _cdb_nvcc_arch_text ERROR_QUIET)
      endif()
    endif()
    if(_cdb_nvcc_arch_text)
      set(_cdb_arches "")
      foreach(_cdb_arch IN LISTS CUDA_DEPTH_BUFFER_FALLBACK_ARCHITECTURES)
        if(_cdb_nvcc_arch_text MATCHES "compute_${_cdb_arch}")
          list(APPEND _cdb_arches ${_cdb_arch})
        endif()
      endforeach()
      if(_cdb_arches)
        message(STATUS "no GPU visible, building for ${_cdb_arches} "
                       "(of ${CUDA_DEPTH_BUFFER_FALLBACK_ARCHITECTURES}, "
                       "the rest is beyond this nvcc)")
        set(CMAKE_CUDA_ARCHITECTURES ${_cdb_arches})
      else()
        message(WARNING "this nvcc supports none of "
                        "${CUDA_DEPTH_BUFFER_FALLBACK_ARCHITECTURES}; "
                        "pass -DCMAKE_CUDA_ARCHITECTURES=<arch>")
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_DEPTH_BUFFER_FALLBACK_ARCHITECTURES})
      endif()
    else()
      message(STATUS "no GPU visible and no nvcc to ask, building for "
                     "${CUDA_DEPTH_BUFFER_FALLBACK_ARCHITECTURES}")
      set(CMAKE_CUDA_ARCHITECTURES ${CUDA_DEPTH_BUFFER_FALLBACK_ARCHITECTURES})
    endif()
  endif()
endif()

project(cuda_depth_buffer
    VERSION 0.1.0
    DESCRIPTION "Mapless sensor-space ESDF queries over a buffer of posed depth images"
    HOMEPAGE_URL "https://github.com/ethz-mrl/mapless-esdf"
    LANGUAGES CXX CUDA
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# On by default for whoever is developing this project, off for whoever is only
# consuming it: a planner adding this repository as a subdirectory should not be
# made to have GTest, nor to compile examples it will never run.
# PROJECT_IS_TOP_LEVEL is set by project() from CMake 3.21; this project asks
# for 3.20, so it is derived rather than relied on.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(_cdb_developing ON)
else()
  set(_cdb_developing OFF)
endif()
option(CUDA_DEPTH_BUFFER_BUILD_TESTS "Build the unit and system tests"
       ${_cdb_developing})
option(CUDA_DEPTH_BUFFER_BUILD_EXAMPLES "Build the C++ examples"
       ${_cdb_developing})
option(CUDA_DEPTH_BUFFER_BUILD_BENCHMARK "Build the benchmark driver" OFF)
option(CUDA_DEPTH_BUFFER_BUILD_PYTHON "Build the Python extension module" OFF)
option(CUDA_DEPTH_BUFFER_ENABLE_PROFILING
       "Record per stage timings in the profiling registry" ON)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

# Device compilation options, pinned on the library target rather than the
# global flags so that every translation unit is built with the same numerics.
# -use_fast_math is part of the contract, not a tuning knob: the occlusion pass
# recomputes pixel ray directions with the expression the integration path uses,
# and both must lower identically for the results to agree bit for bit.
# --default-stream per-thread keeps the implicit stream private to the caller.
set(CUDA_DEPTH_BUFFER_CUDA_OPTIONS
  -O3
  -use_fast_math
  --expt-relaxed-constexpr
  --expt-extended-lambda
  --default-stream per-thread
)

add_library(${PROJECT_NAME}
  src/buffer_params.cpp
  src/depth_buffer.cu
  src/pose_covariance.cpp
  src/scene.cpp
  src/profiling.cpp
)
add_library(mrl::cuda_depth_buffer ALIAS ${PROJECT_NAME})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_DEPTH_BUFFER_CUDA_OPTIONS}>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
)
if(CUDA_DEPTH_BUFFER_ENABLE_PROFILING)
  target_compile_definitions(${PROJECT_NAME} PUBLIC CUDA_DEPTH_BUFFER_ENABLE_PROFILING)
endif()

target_include_directories(${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC CUDA::cudart Eigen3::Eigen
  PRIVATE yaml-cpp
)

############################## tests ##########################################
if(CUDA_DEPTH_BUFFER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

############################## examples #######################################
if(CUDA_DEPTH_BUFFER_BUILD_EXAMPLES)
  add_subdirectory(examples/cpp)
endif()

############################## benchmark ######################################
if(CUDA_DEPTH_BUFFER_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

if(CUDA_DEPTH_BUFFER_BUILD_PYTHON)
  add_subdirectory(python)
endif()

############################## install ########################################
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE mrl::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
configure_package_config_file(
  cmake/${PROJECT_NAME}-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  COMPATIBILITY SameMinorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
