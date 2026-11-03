#include <algorithm>
#include <cmath>
#include <limits>
#include <mrl/cuda_depth_buffer/scene.hpp>
#include <random>

namespace mrl {
namespace {

using Vec3 = std::array<float, 3>;

Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}
Vec3 operator+(const Vec3& a, const Vec3& b) {
  return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}
Vec3 operator*(const Vec3& a, float s) {
  return {{a[0] * s, a[1] * s, a[2] * s}};
}
float dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {{a[1] * b[2] - a[2] * b[1],
           a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]}};
}
float norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
Vec3 normalized(const Vec3& a) {
  const float n = norm(a);
  return n > 0.0f ? a * (1.0f / n) : Vec3{{0.0f, 0.0f, 1.0f}};
}

/// Rotates by the pose's quaternion, or by its inverse.
Vec3 rotate(const Pose& pose, const Vec3& p, bool inverse) {
  const float sign = inverse ? -1.0f : 1.0f;
  const float qx = sign * pose[3];
  const float qy = sign * pose[4];
  const float qz = sign * pose[5];
  const float qw = pose[6];
  const float cx = 2.0f * (qy * p[2] - qz * p[1]);
  const float cy = 2.0f * (qz * p[0] - qx * p[2]);
  const float cz = 2.0f * (qx * p[1] - qy * p[0]);
  return {{p[0] + qw * cx + qy * cz - qz * cy,
           p[1] + qw * cy + qz * cx - qx * cz,
           p[2] + qw * cz + qx * cy - qy * cx}};
}

constexpr float kNoHit = std::numeric_limits<float>::infinity();

/// Ray origin + t * direction against a sphere. `direction` is deliberately not
/// normalized: it is the pixel ray (px, py, 1) in camera coordinates, so t
/// comes out as the depth the sensor would report.
float intersectSphere(const Vec3& origin,
                      const Vec3& direction,
                      const Sphere& sphere) {
  const Vec3 oc = origin - sphere.center;
  const float a = dot(direction, direction);
  const float b = 2.0f * dot(direction, oc);
  const float c = dot(oc, oc) - sphere.radius * sphere.radius;
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) return kNoHit;
  const float root = std::sqrt(discriminant);
  const float t0 = (-b - root) / (2.0f * a);
  const float t1 = (-b + root) / (2.0f * a);
  if (t0 > 0.0f) return t0;
  if (t1 > 0.0f) return t1;
  return kNoHit;
}

float intersectCylinder(const Vec3& origin,
                        const Vec3& direction,
                        const Cylinder& cylinder) {
  const Vec3 axis = normalized(cylinder.axis);
  const Vec3 w = origin - cylinder.base;
  const Vec3 d_perp = direction - axis * dot(direction, axis);
  const Vec3 w_perp = w - axis * dot(w, axis);

  float best = kNoHit;
  const float a = dot(d_perp, d_perp);
  if (a > 1e-12f) {  // ray not parallel to the axis
    const float b = 2.0f * dot(d_perp, w_perp);
    const float c = dot(w_perp, w_perp) - cylinder.radius * cylinder.radius;
    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant >= 0.0f) {
      const float root = std::sqrt(discriminant);
      for (const float t :
           {(-b - root) / (2.0f * a), (-b + root) / (2.0f * a)}) {
        if (t <= 0.0f || t >= best) continue;
        const float along = dot(w + direction * t, axis);
        if (along >= 0.0f && along <= cylinder.height) best = t;
      }
    }
  }

  // End caps, so that looking down the axis of a trunk still hits something.
  const float denominator = dot(direction, axis);
  if (std::fabs(denominator) > 1e-12f) {
    for (const float cap : {0.0f, cylinder.height}) {
      const float t = (cap - dot(w, axis)) / denominator;
      if (t <= 0.0f || t >= best) continue;
      const Vec3 hit = w + direction * t;
      const Vec3 radial = hit - axis * dot(hit, axis);
      if (dot(radial, radial) <= cylinder.radius * cylinder.radius) best = t;
    }
  }
  return best;
}

float intersectPlane(const Vec3& origin,
                     const Vec3& direction,
                     const Plane& plane) {
  const float denominator = dot(direction, plane.normal);
  if (std::fabs(denominator) < 1e-12f) return kNoHit;
  const float t = dot(plane.point - origin, plane.normal) / denominator;
  return t > 0.0f ? t : kNoHit;
}

float distanceToCylinder(const Vec3& p, const Cylinder& cylinder) {
  const Vec3 axis = normalized(cylinder.axis);
  const Vec3 w = p - cylinder.base;
  const float along = dot(w, axis);
  const float radial = norm(w - axis * along);
  const float clamped = std::min(std::max(along, 0.0f), cylinder.height);
  // Outside the slab the closest point sits on a cap rim or face.
  const float axial_gap = std::fabs(along - clamped);
  const float radial_gap = radial - cylinder.radius;
  if (axial_gap <= 0.0f) return std::fabs(radial_gap);
  if (radial_gap <= 0.0f) return axial_gap;
  return std::sqrt(axial_gap * axial_gap + radial_gap * radial_gap);
}

}  // namespace

AnalyticScene AnalyticScene::forest(const uint32_t seed,
                                    const int num_cylinders,
                                    const int num_spheres,
                                    const float extent,
                                    const bool with_ground) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  const auto place = [&] { return (unit(rng) - 0.5f) * extent; };

  AnalyticScene scene;
  if (with_ground) {
    scene.planes.push_back(Plane{{{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 1.0f}}});
  }
  scene.cylinders.reserve(num_cylinders);
  for (int i = 0; i < num_cylinders; ++i) {
    Cylinder trunk;
    trunk.base = {{place(), place(), 0.0f}};
    trunk.axis = {{0.0f, 0.0f, 1.0f}};
    trunk.radius = 0.08f + 0.22f * unit(rng);
    trunk.height = 2.0f + 3.0f * unit(rng);
    scene.cylinders.push_back(trunk);
  }
  scene.spheres.reserve(num_spheres);
  for (int i = 0; i < num_spheres; ++i) {
    Sphere bush;
    bush.radius = 0.15f + 0.45f * unit(rng);
    bush.center = {{place(), place(), bush.radius + 1.5f * unit(rng)}};
    scene.spheres.push_back(bush);
  }
  return scene;
}

std::vector<float> AnalyticScene::render(const CameraIntrinsics& intrinsics,
                                         const Pose& pose) const {
  const Vec3 origin{{pose[0], pose[1], pose[2]}};
  std::vector<float> depth(static_cast<size_t>(intrinsics.width) *
                           intrinsics.height);

  for (uint16_t v = 0; v < intrinsics.height; ++v) {
    const float py = (static_cast<float>(v) - intrinsics.cy) / intrinsics.fy;
    for (uint16_t u = 0; u < intrinsics.width; ++u) {
      const float px = (static_cast<float>(u) - intrinsics.cx) / intrinsics.fx;
      // Unnormalized on purpose: t along this ray is the depth.
      const Vec3 direction = rotate(pose, Vec3{{px, py, 1.0f}}, false);

      float best = kNoHit;
      for (const Sphere& sphere : spheres) {
        best = std::min(best, intersectSphere(origin, direction, sphere));
      }
      for (const Cylinder& cylinder : cylinders) {
        best = std::min(best, intersectCylinder(origin, direction, cylinder));
      }
      for (const Plane& plane : planes) {
        best = std::min(best, intersectPlane(origin, direction, plane));
      }
      depth[static_cast<size_t>(v) * intrinsics.width + u] =
          std::min(best, intrinsics.z_far);
    }
  }
  return depth;
}

PosedDepthImage AnalyticScene::renderPosed(const CameraIntrinsics& intrinsics,
                                           const Pose& pose,
                                           const int64_t stamp_ns) const {
  PosedDepthImage image;
  image.depth = render(intrinsics, pose);
  image.pose = pose;
  image.stamp_ns = stamp_ns;
  return image;
}

float AnalyticScene::distanceToSurface(
    const std::array<float, 3>& point) const {
  float best = std::numeric_limits<float>::infinity();
  for (const Sphere& sphere : spheres) {
    best =
        std::min(best, std::fabs(norm(point - sphere.center) - sphere.radius));
  }
  for (const Cylinder& cylinder : cylinders) {
    best = std::min(best, distanceToCylinder(point, cylinder));
  }
  for (const Plane& plane : planes) {
    best = std::min(best, std::fabs(dot(point - plane.point, plane.normal)));
  }
  return best;
}

Pose lookAt(const std::array<float, 3>& eye,
            const std::array<float, 3>& target,
            const std::array<float, 3>& up) {
  // Optical frame: z towards the target, x to the right, y down.
  const Vec3 forward = normalized(target - eye);
  Vec3 right = cross(forward, up);
  if (norm(right) < 1e-6f) {  // looking along `up`, pick any perpendicular
    right = cross(forward, Vec3{{1.0f, 0.0f, 0.0f}});
  }
  right = normalized(right);
  const Vec3 down = cross(forward, right);

  // Rotation matrix with columns (right, down, forward), to a quaternion.
  const float m[9] = {right[0],
                      down[0],
                      forward[0],
                      right[1],
                      down[1],
                      forward[1],
                      right[2],
                      down[2],
                      forward[2]};
  const float trace = m[0] + m[4] + m[8];
  Pose pose{{eye[0], eye[1], eye[2], 0.0f, 0.0f, 0.0f, 1.0f}};
  if (trace > 0.0f) {
    const float s = 0.5f / std::sqrt(trace + 1.0f);
    pose[6] = 0.25f / s;
    pose[3] = (m[7] - m[5]) * s;
    pose[4] = (m[2] - m[6]) * s;
    pose[5] = (m[3] - m[1]) * s;
  } else if (m[0] > m[4] && m[0] > m[8]) {
    const float s = 2.0f * std::sqrt(1.0f + m[0] - m[4] - m[8]);
    pose[6] = (m[7] - m[5]) / s;
    pose[3] = 0.25f * s;
    pose[4] = (m[1] + m[3]) / s;
    pose[5] = (m[2] + m[6]) / s;
  } else if (m[4] > m[8]) {
    const float s = 2.0f * std::sqrt(1.0f + m[4] - m[0] - m[8]);
    pose[6] = (m[2] - m[6]) / s;
    pose[3] = (m[1] + m[3]) / s;
    pose[4] = 0.25f * s;
    pose[5] = (m[5] + m[7]) / s;
  } else {
    const float s = 2.0f * std::sqrt(1.0f + m[8] - m[0] - m[4]);
    pose[6] = (m[3] - m[1]) / s;
    pose[3] = (m[2] + m[6]) / s;
    pose[4] = (m[5] + m[7]) / s;
    pose[5] = 0.25f * s;
  }
  const float n = std::sqrt(pose[3] * pose[3] + pose[4] * pose[4] +
                            pose[5] * pose[5] + pose[6] * pose[6]);
  for (int i = 3; i < 7; ++i) pose[i] /= n;
  return pose;
}

std::vector<Pose> orbitPoses(const int count,
                             const float radius,
                             const float height,
                             const std::array<float, 3>& target) {
  std::vector<Pose> poses;
  poses.reserve(static_cast<size_t>(std::max(count, 0)));
  for (int i = 0; i < count; ++i) {
    const float angle = 2.0f * 3.14159265358979323846f * static_cast<float>(i) /
                        static_cast<float>(std::max(count, 1));
    const std::array<float, 3> eye{{target[0] + radius * std::cos(angle),
                                    target[1] + radius * std::sin(angle),
                                    height}};
    poses.push_back(lookAt(eye, target));
  }
  return poses;
}

namespace {

/// The invariant the buffer stores depths under, replicated so the oracle sees
/// exactly what the query sees.
float normalizeDepth(float depth, const CameraIntrinsics& k) {
  if (std::isnan(depth) || depth <= 0.0f) return k.z_near;
  return std::min(std::max(depth, k.z_near), k.z_far);
}

struct FrameDistances {
  double optimistic = 0.0;
  double conservative = 0.0;
  bool query_is_free = false;
};

FrameDistances frameDistances(const CameraIntrinsics& k,
                              const PosedDepthImage& frame,
                              const Vec3& query_world,
                              const double truncation_radius,
                              const PoolCompensation mode) {
  const bool pixel_extent = mode == PoolCompensation::kExact;
  const Vec3 relative =
      query_world - Vec3{{frame.pose[0], frame.pose[1], frame.pose[2]}};
  const Vec3 q = rotate(frame.pose, relative, true);

  double surface = truncation_radius;
  double surface_depth = 0.0;  // depth of the pixel that set `surface`
  double occlusion = truncation_radius;

  // The near clipping plane bounds unknown space, so it contributes candidates
  // whenever the query can reach it.
  const bool near_active = q[2] < k.z_near + truncation_radius;

  const auto depthAt = [&](int u, int v) {
    return normalizeDepth(frame.depth[static_cast<size_t>(v) * k.width + u], k);
  };

  for (int v = 0; v < k.height; ++v) {
    const double py = (static_cast<double>(v) - k.cy) / k.fy;
    for (int u = 0; u < k.width; ++u) {
      const double px = (static_cast<double>(u) - k.cx) / k.fx;
      const double depth = depthAt(u, v);

      // --- observed surface -------------------------------------------------
      double sx = px * depth - q[0];
      double sy = py * depth - q[1];
      if (pixel_extent) {
        // The pixel is the patch half a pixel either side of its ray at its
        // depth: nearest point = query clamped into the patch.
        const double hx = 0.5 / k.fx * depth, hy = 0.5 / k.fy * depth;
        sx = std::min(std::max(static_cast<double>(q[0]), px * depth - hx),
                      px * depth + hx) -
             q[0];
        sy = std::min(std::max(static_cast<double>(q[1]), py * depth - hy),
                      py * depth + hy) -
             q[1];
      }
      const double sz = depth - q[2];
      const double candidate = std::sqrt(sx * sx + sy * sy + sz * sz);
      if (candidate < surface) {
        surface = candidate;
        surface_depth = depth;
      }

      // --- boundary of unknown space ---------------------------------------
      const double ray_norm = std::sqrt(px * px + py * py + 1.0);
      const Vec3 dir{{static_cast<float>(px / ray_norm),
                      static_cast<float>(py / ray_norm),
                      static_cast<float>(1.0 / ray_norm)}};

      const bool border =
          (u == 0 || v == 0 || u == k.width - 1 || v == k.height - 1);
      const double neighbour_u = border ? k.z_near : depthAt(u + 1, v);
      const double neighbour_v = border ? k.z_near : depthAt(u, v + 1);

      double along = q[0] * dir[0] + q[1] * dir[1] + q[2] * dir[2];
      along = std::max(along, static_cast<double>(k.z_near));
      const double fx_ = along * dir[0];
      const double fy_ = along * dir[1];
      const double fz = along * dir[2];

      const bool inside_u = fz > std::min(depth, neighbour_u) &&
                            fz < std::max(depth, neighbour_u);
      const bool inside_v = fz > std::min(depth, neighbour_v) &&
                            fz < std::max(depth, neighbour_v);
      if (inside_u || inside_v) {
        const double dx = fx_ - q[0];
        const double dy = fy_ - q[1];
        const double dz = fz - q[2];
        occlusion = std::min(occlusion, std::sqrt(dx * dx + dy * dy + dz * dz));
      }

      if (near_active) {
        // Where this pixel's ray crosses the near plane.
        const double nx = k.z_near * px - q[0];
        const double ny = k.z_near * py - q[1];
        const double nz = k.z_near - q[2];
        occlusion = std::min(occlusion, std::sqrt(nx * nx + ny * ny + nz * nz));
      }
    }
  }

  // PoolCompensation::kApproximate, as the buffer does it: the raw nearest
  // sample competes with the occlusion candidates, and a winning surface
  // sample then loses half the pooled footprint at its depth.
  double surface_compensated = surface;
  if (mode == PoolCompensation::kApproximate && surface < truncation_radius) {
    surface_compensated = std::max(0.0, surface - 0.5 * surface_depth / k.fx);
  }
  FrameDistances out;
  out.optimistic = std::min(surface_compensated, truncation_radius);
  out.conservative =
      std::min(surface <= occlusion ? surface_compensated : occlusion,
               truncation_radius);

  // Sign: positive when the query sits in front of what this frame observed.
  out.query_is_free = false;
  if (q[2] >= k.z_near && q[2] <= k.z_far) {
    const int u = static_cast<int>(std::lround(k.fx * q[0] / q[2] + k.cx));
    const int v = static_cast<int>(std::lround(k.fy * q[1] / q[2] + k.cy));
    if (u >= 0 && u < k.width && v >= 0 && v < k.height) {
      out.query_is_free = q[2] < depthAt(u, v);
    }
  }
  return out;
}

}  // namespace

BruteForceDistances bruteForceDistances(
    const CameraIntrinsics& intrinsics,
    const std::vector<PosedDepthImage>& frames,
    const std::array<float, 3>& query,
    const float truncation_radius,
    const bool pixel_extent) {
  return bruteForceDistances(
      intrinsics,
      frames,
      query,
      truncation_radius,
      pixel_extent ? PoolCompensation::kExact : PoolCompensation::kNone);
}

BruteForceDistances bruteForceDistances(
    const CameraIntrinsics& intrinsics,
    const std::vector<PosedDepthImage>& frames,
    const std::array<float, 3>& query,
    const float truncation_radius,
    const PoolCompensation mode) {
  double optimistic = truncation_radius;
  double conservative = -static_cast<double>(truncation_radius);
  for (const PosedDepthImage& frame : frames) {
    const FrameDistances d =
        frameDistances(intrinsics, frame, query, truncation_radius, mode);
    optimistic = std::min(optimistic, d.optimistic);
    // Max pooling of the signed distances: free space observed anywhere wins.
    const double signed_distance =
        d.query_is_free ? d.conservative : -d.conservative;
    conservative = std::max(conservative, signed_distance);
  }
  BruteForceDistances out;
  out.optimistic = static_cast<float>(optimistic);
  out.conservative = static_cast<float>(conservative);
  return out;
}

float bruteForceDistance(const CameraIntrinsics& intrinsics,
                         const std::vector<PosedDepthImage>& frames,
                         const std::array<float, 3>& query,
                         const float truncation_radius,
                         const bool pixel_extent) {
  return bruteForceDistances(intrinsics,
                             frames,
                             query,
                             truncation_radius,
                             pixel_extent)
      .optimistic;
}

float bruteForceDistance(const CameraIntrinsics& intrinsics,
                         const std::vector<PosedDepthImage>& frames,
                         const std::array<float, 3>& query,
                         const float truncation_radius,
                         const PoolCompensation mode) {
  return bruteForceDistances(intrinsics, frames, query, truncation_radius, mode)
      .optimistic;
}

}  // namespace mrl
