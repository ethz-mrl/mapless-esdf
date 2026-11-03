// Python bindings.
//
// Arrays come in as numpy by default and go out as numpy always. Anything
// exposing __cuda_array_interface__ (torch cuda tensors, cupy) or DLPack is
// taken by its device pointer instead, so a GPU pipeline never has to round
// trip through the host to ask a question. The library reads those pointers
// directly; see CudaDepthBuffer::query.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mrl/cuda_depth_buffer/buffer_params.hpp>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <mrl/cuda_depth_buffer/profiling.hpp>
#include <mrl/cuda_depth_buffer/scene.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

/// A float32 buffer to read from, either a borrowed pointer into the caller's
/// array or a converted copy that has to stay alive for the call.
struct FloatInput {
  const float* data = nullptr;
  size_t count = 0;
  py::array_t<float> owned;          ///< empty when the pointer is borrowed
  std::shared_ptr<void> keep_alive;  ///< holds a DLPack capsule, if any
};

/// Number of elements of a __cuda_array_interface__ or DLPack shape.
size_t elementCount(const py::sequence& shape) {
  size_t count = 1;
  for (const py::handle& dim : shape) count *= dim.cast<size_t>();
  return count;
}

/// What a device array has to be for its memory to be read where it lies.
/// Anything else has to be converted by its owner, which is a real copy and
/// should say so rather than happening silently.
[[noreturn]] void rejectDeviceArray(const char* reason) {
  throw std::invalid_argument(
      std::string("CUDA array is ") + reason +
      "; pass a contiguous float32 array (torch: .contiguous().float(), "
      "cupy: cupy.ascontiguousarray(a, dtype=cupy.float32)), or move it to "
      "the host yourself");
}

/// Device pointer of an object exposing __cuda_array_interface__, or nullptr if
/// it exposes no such interface.
const float* cudaArrayInterfacePointer(const py::object& source,
                                       size_t& count) {
  if (!py::hasattr(source, "__cuda_array_interface__")) return nullptr;
  const py::dict interface =
      source.attr("__cuda_array_interface__").cast<py::dict>();
  if (interface.contains("mask") && !interface["mask"].is_none()) {
    rejectDeviceArray("masked");
  }
  if (interface.contains("strides") && !interface["strides"].is_none()) {
    rejectDeviceArray("not contiguous");
  }
  if (interface["typestr"].cast<std::string>() != "<f4") {
    rejectDeviceArray("not float32");
  }
  const py::tuple pointer = interface["data"].cast<py::tuple>();
  count = elementCount(interface["shape"].cast<py::sequence>());
  return reinterpret_cast<const float*>(pointer[0].cast<uintptr_t>());
}

// Minimal DLPack ABI. Vendored rather than depended on: the struct layout is
// the stable part of the standard, and the only producer that ships the header
// is the one framework we do not want to require.
extern "C" {
struct DLDevice {
  int32_t device_type;
  int32_t device_id;
};
struct DLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};
struct DLTensor {
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};
struct DLManagedTensor {
  DLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(DLManagedTensor* self);
};
}
constexpr int32_t kDLCPU = 1;
constexpr int32_t kDLCUDA = 2;
constexpr int32_t kDLCUDAHost = 3;
constexpr int32_t kDLCUDAManaged = 13;
constexpr uint8_t kDLFloat = 2;

/// Owns the DLPack capsule for as long as the pointer taken out of it is used.
/// The protocol hands ownership to the consumer, so the deleter is ours to
/// call.
class DLPackTensor {
 public:
  explicit DLPackTensor(const py::object& source) {
    if (!py::hasattr(source, "__dlpack__")) return;
    py::object capsule = source.attr("__dlpack__")();
    if (!PyCapsule_CheckExact(capsule.ptr())) return;
    if (!PyCapsule_IsValid(capsule.ptr(), "dltensor")) return;
    managed_ = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(), "dltensor"));
    // Renaming marks the capsule consumed, so the producer's own deleter for an
    // unconsumed capsule does not also run.
    PyCapsule_SetName(capsule.ptr(), "used_dltensor");
  }

  ~DLPackTensor() {
    if (managed_ && managed_->deleter) managed_->deleter(managed_);
  }

  DLPackTensor(const DLPackTensor&) = delete;
  DLPackTensor& operator=(const DLPackTensor&) = delete;

  /// Pointer to the payload, or nullptr when the caller should fall back to
  /// converting through numpy.
  ///
  /// A host tensor that does not fit falls back silently: numpy copies it and
  /// nothing is lost. A device tensor that does not fit raises instead, because
  /// the fallback would be a device to host copy the caller did not ask for and
  /// would only notice in the timings.
  const float* pointer(size_t& count) const {
    if (!managed_) return nullptr;
    const DLTensor& tensor = managed_->dl_tensor;
    const int32_t device_type = tensor.device.device_type;
    const bool on_device =
        device_type == kDLCUDA || device_type == kDLCUDAManaged;
    if (!on_device && device_type != kDLCPU && device_type != kDLCUDAHost) {
      return nullptr;
    }
    const auto reject = [on_device](const char* reason) -> const float* {
      if (on_device) rejectDeviceArray(reason);
      return nullptr;
    };
    if (tensor.dtype.code != kDLFloat || tensor.dtype.bits != 32 ||
        tensor.dtype.lanes != 1) {
      return reject("not float32");
    }
    count = 1;
    for (int32_t i = 0; i < tensor.ndim; ++i) {
      count *= static_cast<size_t>(tensor.shape[i]);
    }
    if (tensor.strides != nullptr) {
      int64_t packed = 1;
      for (int32_t i = tensor.ndim - 1; i >= 0; --i) {
        if (tensor.strides[i] != packed) return reject("not contiguous");
        packed *= tensor.shape[i];
      }
    }
    return reinterpret_cast<const float*>(
        static_cast<const char*>(tensor.data) + tensor.byte_offset);
  }

 private:
  DLManagedTensor* managed_ = nullptr;
};

/// Contiguous float32 view of `source`, without copying where possible. The
/// pointer may be device memory; the library infers that.
FloatInput asFloatInput(const py::object& source) {
  FloatInput input;
  size_t count = 0;
  if (const float* device = cudaArrayInterfacePointer(source, count)) {
    input.data = device;
    input.count = count;
    return input;
  }
  if (!py::isinstance<py::array>(source) && py::hasattr(source, "__dlpack__")) {
    auto tensor = std::make_shared<DLPackTensor>(source);
    if (const float* data = tensor->pointer(count)) {
      input.data = data;
      input.count = count;
      input.keep_alive = std::move(tensor);
      return input;
    }
  }
  input.owned =
      py::array_t<float, py::array::c_style | py::array::forcecast>(source);
  input.data = input.owned.data();
  input.count = static_cast<size_t>(input.owned.size());
  return input;
}

/// Checked here rather than in the kernel so that the error names the argument.
void expectCount(const FloatInput& input, const char* name, size_t expected) {
  if (input.count != expected) {
    throw std::invalid_argument(std::string(name) + " must hold " +
                                std::to_string(expected) + " values, got " +
                                std::to_string(input.count));
  }
}

FloatInput asFloatInput(const py::object& source,
                        const char* name,
                        size_t expected) {
  FloatInput input = asFloatInput(source);
  expectCount(input, name, expected);
  return input;
}

py::array_t<float> toArray(const std::vector<float>& values,
                           size_t rows,
                           size_t cols) {
  py::array_t<float> array =
      cols == 1 ? py::array_t<float>(
                      py::array::ShapeContainer{static_cast<py::ssize_t>(rows)})
                : py::array_t<float>(py::array::ShapeContainer{
                      static_cast<py::ssize_t>(rows),
                      static_cast<py::ssize_t>(cols)});
  std::copy(values.begin(), values.end(), array.mutable_data());
  return array;
}

/// QueryResults as numpy, distances (N,) and points and gradients (N, 3).
py::dict resultsToPython(const mrl::QueryResults& results, size_t num_queries) {
  py::dict out;
  out["optimistic_distance"] =
      toArray(results.optimistic_distance, num_queries, 1);
  out["optimistic_gradient"] =
      toArray(results.optimistic_gradient, num_queries, 3);
  out["optimistic_closest"] =
      toArray(results.optimistic_closest, num_queries, 3);
  out["conservative_distance"] =
      toArray(results.conservative_distance, num_queries, 1);
  out["conservative_gradient"] =
      toArray(results.conservative_gradient, num_queries, 3);
  out["conservative_closest"] =
      toArray(results.conservative_closest, num_queries, 3);
  return out;
}

std::vector<double> asCovariance(const py::object& source) {
  const py::array_t<double, py::array::c_style | py::array::forcecast> array(
      source);
  if (array.size() != 36) {
    throw std::invalid_argument("covariance must be 6x6");
  }
  return std::vector<double>(array.data(), array.data() + 36);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "Sensor space ESDF queries over a ring buffer of posed depth images";

  py::class_<mrl::CameraIntrinsics>(m, "CameraIntrinsics")
      .def(py::init<>())
      .def(py::init([](float fx,
                       float fy,
                       float cx,
                       float cy,
                       uint16_t width,
                       uint16_t height,
                       float z_near,
                       float z_far) {
             mrl::CameraIntrinsics k;
             k.fx = fx;
             k.fy = fy;
             k.cx = cx;
             k.cy = cy;
             k.width = width;
             k.height = height;
             k.z_near = z_near;
             k.z_far = z_far;
             return k;
           }),
           py::arg("fx"),
           py::arg("fy"),
           py::arg("cx"),
           py::arg("cy"),
           py::arg("width"),
           py::arg("height"),
           py::arg("z_near") = 0.1f,
           py::arg("z_far") = 20.0f)
      .def_readwrite("fx", &mrl::CameraIntrinsics::fx)
      .def_readwrite("fy", &mrl::CameraIntrinsics::fy)
      .def_readwrite("cx", &mrl::CameraIntrinsics::cx)
      .def_readwrite("cy", &mrl::CameraIntrinsics::cy)
      .def_readwrite("width", &mrl::CameraIntrinsics::width)
      .def_readwrite("height", &mrl::CameraIntrinsics::height)
      .def_readwrite("z_near", &mrl::CameraIntrinsics::z_near)
      .def_readwrite("z_far", &mrl::CameraIntrinsics::z_far)
      .def("__repr__", [](const mrl::CameraIntrinsics& k) {
        return "CameraIntrinsics(fx=" + std::to_string(k.fx) +
               ", fy=" + std::to_string(k.fy) + ", cx=" + std::to_string(k.cx) +
               ", cy=" + std::to_string(k.cy) +
               ", width=" + std::to_string(k.width) +
               ", height=" + std::to_string(k.height) + ")";
      });

  py::class_<mrl::BufferParams>(m, "BufferParams")
      .def(py::init<>())
      .def_readwrite("num_frames", &mrl::BufferParams::num_frames)
      .def_readwrite("use_keyframing", &mrl::BufferParams::use_keyframing)
      .def_readwrite("min_pool_size", &mrl::BufferParams::min_pool_size)
      .def_readwrite("depth_scale_factor",
                     &mrl::BufferParams::depth_scale_factor)
      .def_readwrite("invalid_is_farplane",
                     &mrl::BufferParams::invalid_is_farplane)
      .def_readwrite("robot_radius", &mrl::BufferParams::robot_radius)
      .def_readwrite("keyframe_dist_threshold",
                     &mrl::BufferParams::keyframe_dist_threshold)
      .def_readwrite("keyframe_angle_threshold",
                     &mrl::BufferParams::keyframe_angle_threshold)
      .def_readwrite("keyframe_time_threshold",
                     &mrl::BufferParams::keyframe_time_threshold)
      .def_readwrite("pose_covariance_confidence",
                     &mrl::BufferParams::pose_covariance_confidence)
      .def_property(
          "robot_center_in_camera",
          [](const mrl::BufferParams& p) {
            return std::vector<float>(p.robot_center_in_camera,
                                      p.robot_center_in_camera + 3);
          },
          [](mrl::BufferParams& p, const std::vector<float>& center) {
            if (center.size() != 3) {
              throw std::invalid_argument("robot_center_in_camera must be xyz");
            }
            std::copy(center.begin(), center.end(), p.robot_center_in_camera);
          })
      .def_static("from_yaml_file",
                  &mrl::BufferParams::fromYamlFile,
                  py::arg("filename"))
      .def_static("from_yaml", &mrl::BufferParams::fromYaml, py::arg("yaml"))
      .def("to_yaml", &mrl::BufferParams::toYaml)
      .def("__repr__", [](const mrl::BufferParams& p) {
        return "BufferParams(num_frames=" + std::to_string(p.num_frames) + ")";
      });

  py::class_<mrl::CudaDepthBuffer>(m, "CudaDepthBuffer")
      .def(py::init<const mrl::CameraIntrinsics&, const mrl::BufferParams&>(),
           py::arg("intrinsics"),
           py::arg("params"))
      .def_property_readonly("intrinsics", &mrl::CudaDepthBuffer::intrinsics)
      .def_property_readonly("params", &mrl::CudaDepthBuffer::params)
      .def(
          "set_posed_depth_image",
          [](mrl::CudaDepthBuffer& self,
             const py::object& depth,
             const py::object& pose,
             int64_t stamp_ns,
             bool is_keyframe) {
            const mrl::CameraIntrinsics k = self.intrinsics();
            const mrl::BufferParams p = self.params();
            const size_t pool = p.min_pool_size == 0 ? 1 : p.min_pool_size;
            const size_t pixels =
                static_cast<size_t>(k.width) * k.height * pool * pool;
            const FloatInput depth_input = asFloatInput(depth, "depth", pixels);
            const FloatInput pose_input = asFloatInput(pose, "pose", 7);
            py::gil_scoped_release unlocked;
            self.setPosedDepthImage(depth_input.data,
                                    pose_input.data,
                                    stamp_ns,
                                    is_keyframe);
          },
          py::arg("depth"),
          py::arg("pose"),
          py::arg("stamp_ns"),
          py::arg("is_keyframe") = true,
          "Insert a depth image with the pose of its camera. `depth` may be a\n"
          "numpy array or any CUDA array; `pose` is x, y, z, qx, qy, qz, qw.")
      .def(
          "is_keyframe",
          [](const mrl::CudaDepthBuffer& self,
             const py::object& pose,
             int64_t stamp_ns) {
            const FloatInput pose_input = asFloatInput(pose, "pose", 7);
            return self.isKeyframe(pose_input.data, stamp_ns);
          },
          py::arg("pose"),
          py::arg("stamp_ns"))
      .def(
          "query",
          [](mrl::CudaDepthBuffer& self,
             const py::object& points,
             bool conservative,
             float truncation_radius,
             float suppression_radius,
             bool verbose) {
            const FloatInput input = asFloatInput(points);
            if (input.count % 3 != 0) {
              throw std::invalid_argument(
                  "queries must be xyz interleaved, so a multiple of three");
            }
            const size_t num_queries = input.count / 3;
            mrl::QueryResults results;
            {
              py::gil_scoped_release unlocked;
              self.query(input.data,
                         static_cast<uint32_t>(num_queries),
                         conservative,
                         results,
                         truncation_radius,
                         verbose,
                         suppression_radius,
                         nullptr);
            }
            return resultsToPython(results, num_queries);
          },
          py::arg("points"),
          py::arg("conservative") = true,
          py::arg("truncation_radius") = std::numeric_limits<float>::infinity(),
          py::arg("suppression_radius") = 0.0f,
          py::arg("verbose") = false,
          "Signed distance, gradient and closest point of every query point,\n"
          "in the same reference frame as the poses. Returns a dict of numpy\n"
          "arrays; the conservative entries are NaN when conservative=False.")
      .def(
          "update_pose_and_covariance_by_timestamp",
          [](mrl::CudaDepthBuffer& self,
             int64_t stamp_ns,
             const py::object& pose,
             const py::object& covariance) {
            std::vector<double> flat;
            const double* covariance_data = nullptr;
            if (!covariance.is_none()) {
              flat = asCovariance(covariance);
              covariance_data = flat.data();
            }
            if (pose.is_none()) {
              return self.updatePoseAndCovarianceByTimestamp(stamp_ns,
                                                             nullptr,
                                                             covariance_data);
            }
            const FloatInput pose_input = asFloatInput(pose, "pose", 7);
            return self.updatePoseAndCovarianceByTimestamp(stamp_ns,
                                                           pose_input.data,
                                                           covariance_data);
          },
          py::arg("stamp_ns"),
          py::arg("pose") = py::none(),
          py::arg("covariance") = py::none(),
          "Revise the pose and/or covariance of a buffered frame. Returns\n"
          "False when no frame carries that timestamp.")
      .def(
          "covariance_by_timestamp",
          [](const mrl::CudaDepthBuffer& self, int64_t stamp_ns) -> py::object {
            std::vector<double> covariance(36);
            if (!self.getCovarianceByTimestamp(stamp_ns, covariance.data())) {
              return py::none();
            }
            py::array_t<double> array(py::array::ShapeContainer{6, 6});
            std::copy(covariance.begin(),
                      covariance.end(),
                      array.mutable_data());
            return array;
          },
          py::arg("stamp_ns"))
      .def("set_depth_scale_factor",
           &mrl::CudaDepthBuffer::setDepthScaleFactor,
           py::arg("depth_scale_factor"))
      .def("set_robot_radius",
           &mrl::CudaDepthBuffer::setRobotRadius,
           py::arg("robot_radius"))
      .def("set_pose_covariance_confidence",
           &mrl::CudaDepthBuffer::setPoseCovarianceConfidence,
           py::arg("confidence"))
      .def("warmup", &mrl::CudaDepthBuffer::warmup);

  // Scene generation, so an example, a notebook or a test can build something
  // to look at without a recorded dataset.
  py::class_<mrl::Sphere>(m, "Sphere")
      .def(py::init([](std::array<float, 3> center, float radius) {
             return mrl::Sphere{center, radius};
           }),
           py::arg("center"),
           py::arg("radius"))
      .def_readwrite("center", &mrl::Sphere::center)
      .def_readwrite("radius", &mrl::Sphere::radius);

  py::class_<mrl::Cylinder>(m, "Cylinder")
      .def(py::init([](std::array<float, 3> base,
                       std::array<float, 3> axis,
                       float radius,
                       float height) {
             return mrl::Cylinder{base, axis, radius, height};
           }),
           py::arg("base"),
           py::arg("axis"),
           py::arg("radius"),
           py::arg("height"))
      .def_readwrite("base", &mrl::Cylinder::base)
      .def_readwrite("axis", &mrl::Cylinder::axis)
      .def_readwrite("radius", &mrl::Cylinder::radius)
      .def_readwrite("height", &mrl::Cylinder::height);

  py::class_<mrl::Plane>(m, "Plane")
      .def(
          py::init([](std::array<float, 3> point, std::array<float, 3> normal) {
            return mrl::Plane{point, normal};
          }),
          py::arg("point"),
          py::arg("normal"))
      .def_readwrite("point", &mrl::Plane::point)
      .def_readwrite("normal", &mrl::Plane::normal);

  py::class_<mrl::AnalyticScene>(m, "AnalyticScene")
      .def(py::init<>())
      .def_readwrite("spheres", &mrl::AnalyticScene::spheres)
      .def_readwrite("cylinders", &mrl::AnalyticScene::cylinders)
      .def_readwrite("planes", &mrl::AnalyticScene::planes)
      .def_static("forest",
                  &mrl::AnalyticScene::forest,
                  py::arg("seed"),
                  py::arg("num_cylinders") = 24,
                  py::arg("num_spheres") = 16,
                  py::arg("extent") = 12.0f,
                  py::arg("with_ground") = true,
                  "Cylinders standing on the ground with spheres scattered\n"
                  "between them, reproducible from the seed.")
      .def(
          "render",
          [](const mrl::AnalyticScene& scene,
             const mrl::CameraIntrinsics& k,
             const std::array<float, 7>& pose) {
            const std::vector<float> depth = scene.render(k, pose);
            py::array_t<float> array(
                py::array::ShapeContainer{static_cast<py::ssize_t>(k.height),
                                          static_cast<py::ssize_t>(k.width)});
            std::copy(depth.begin(), depth.end(), array.mutable_data());
            return array;
          },
          py::arg("intrinsics"),
          py::arg("pose"),
          "Depth image of the scene from `pose`, shaped (height, width).")
      .def(
          "distance_to_surface",
          [](const mrl::AnalyticScene& scene,
             const std::array<float, 3>& point) {
            return scene.distanceToSurface(point);
          },
          py::arg("point"),
          "Distance to the nearest surface of the geometry. This is the "
          "scene,\n"
          "not the answer a query should give: the buffer only knows what a\n"
          "camera saw, and only at pixel resolution.");

  m.def("look_at",
        &mrl::lookAt,
        py::arg("eye"),
        py::arg("target"),
        py::arg("up") = std::array<float, 3>{{0.0f, 0.0f, 1.0f}},
        "Camera pose looking from `eye` at `target`, optical convention.");
  m.def("orbit_poses",
        &mrl::orbitPoses,
        py::arg("count"),
        py::arg("radius"),
        py::arg("height"),
        py::arg("target") = std::array<float, 3>{{0.0f, 0.0f, 0.0f}},
        "`count` poses on a circle around `target`, all looking at it.");
  m.def(
      "brute_force_distances",
      [](const mrl::CameraIntrinsics& k,
         const std::vector<mrl::PosedDepthImage>& frames,
         const std::array<float, 3>& query,
         float truncation_radius) {
        const mrl::BruteForceDistances d =
            mrl::bruteForceDistances(k, frames, query, truncation_radius);
        py::dict out;
        out["optimistic"] = d.optimistic;
        out["conservative"] = d.conservative;
        return out;
      },
      py::arg("intrinsics"),
      py::arg("frames"),
      py::arg("query"),
      py::arg("truncation_radius"),
      "Both channels evaluated exhaustively. O(frames * pixels) per query, so\n"
      "this is the definition written out, not a fallback.");

  py::class_<mrl::PosedDepthImage>(m, "PosedDepthImage")
      .def(py::init([](const py::object& depth,
                       std::array<float, 7> pose,
                       int64_t stamp_ns) {
             const py::array_t<float, py::array::c_style | py::array::forcecast>
                 array(depth);
             mrl::PosedDepthImage frame;
             frame.depth.assign(array.data(), array.data() + array.size());
             frame.pose = pose;
             frame.stamp_ns = stamp_ns;
             return frame;
           }),
           py::arg("depth"),
           py::arg("pose"),
           py::arg("stamp_ns") = 0)
      .def_readwrite("pose", &mrl::PosedDepthImage::pose)
      .def_readwrite("stamp_ns", &mrl::PosedDepthImage::stamp_ns)
      .def_property_readonly("depth", [](const mrl::PosedDepthImage& frame) {
        return py::array_t<float>(
            py::array::ShapeContainer{
                static_cast<py::ssize_t>(frame.depth.size())},
            frame.depth.data());
      });

  py::module_ profiling = m.def_submodule("profiling");
  profiling.def("report", &mrl::profiling::Timing::report);
  profiling.def("reset", &mrl::profiling::Timing::reset);
}
