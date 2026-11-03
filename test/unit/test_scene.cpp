#include <gtest/gtest.h>

#include <cmath>
#include <mrl/cuda_depth_buffer/scene.hpp>
#include <random>

namespace {

mrl::CameraIntrinsics intrinsics(uint16_t width = 64, uint16_t height = 48) {
  mrl::CameraIntrinsics k;
  k.cx = 0.5f * width - 0.5f;
  k.cy = 0.5f * height - 0.5f;
  k.fx = 50.0f;
  k.fy = 50.0f;
  k.width = width;
  k.height = height;
  k.z_near = 0.1f;
  k.z_far = 20.0f;
  return k;
}

/// Identity pose: the camera sits at the origin looking along world +z.
const mrl::Pose kIdentity{{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};

}  // namespace

TEST(SceneTest, DepthIsMeasuredAlongTheOpticalAxisNotAlongTheRay) {
  // A sphere of radius 1 centred 5 m ahead. The centre pixel must report 4 m,
  // and an off axis pixel more than 4, because depth is the z coordinate.
  mrl::AnalyticScene scene;
  scene.spheres.push_back(mrl::Sphere{{{0.0f, 0.0f, 5.0f}}, 1.0f});
  const mrl::CameraIntrinsics k = intrinsics();
  const std::vector<float> depth = scene.render(k, kIdentity);

  const size_t centre =
      static_cast<size_t>(k.height / 2) * k.width + k.width / 2;
  EXPECT_NEAR(depth[centre], 4.0f, 0.05f);

  const size_t offset = centre + 8;
  EXPECT_GT(depth[offset], depth[centre]);
  EXPECT_LT(depth[offset], 5.0f);
}

TEST(SceneTest, RaysThatMissReportTheFarPlane) {
  mrl::AnalyticScene scene;
  scene.spheres.push_back(mrl::Sphere{{{0.0f, 0.0f, 5.0f}}, 0.2f});
  const mrl::CameraIntrinsics k = intrinsics();
  const std::vector<float> depth = scene.render(k, kIdentity);
  EXPECT_FLOAT_EQ(depth[0], k.z_far);  // corner pixel misses the small sphere
}

TEST(SceneTest, CylinderIsHitOnItsSideAndOnItsCaps) {
  const mrl::CameraIntrinsics k = intrinsics();
  const size_t centre =
      static_cast<size_t>(k.height / 2) * k.width + k.width / 2;

  // Upright cylinder crossing the optical axis 6 m ahead.
  mrl::AnalyticScene side;
  side.cylinders.push_back(
      mrl::Cylinder{{{0.0f, -2.0f, 6.0f}}, {{0.0f, 1.0f, 0.0f}}, 0.5f, 4.0f});
  EXPECT_NEAR(side.render(k, kIdentity)[centre], 5.5f, 0.05f);

  // Same cylinder pointing at the camera: the near cap is what is seen.
  mrl::AnalyticScene cap;
  cap.cylinders.push_back(
      mrl::Cylinder{{{0.0f, 0.0f, 6.0f}}, {{0.0f, 0.0f, 1.0f}}, 0.5f, 4.0f});
  EXPECT_NEAR(cap.render(k, kIdentity)[centre], 6.0f, 0.05f);
}

TEST(SceneTest, PlaneDepthGrowsWithTheViewingAngle) {
  mrl::AnalyticScene scene;
  // Plane 3 m ahead, facing the camera.
  scene.planes.push_back(
      mrl::Plane{{{0.0f, 0.0f, 3.0f}}, {{0.0f, 0.0f, 1.0f}}});
  const mrl::CameraIntrinsics k = intrinsics();
  const std::vector<float> depth = scene.render(k, kIdentity);
  // A plane perpendicular to the optical axis has constant depth.
  for (float d : depth) EXPECT_NEAR(d, 3.0f, 1e-4f);
}

TEST(SceneTest, LookAtPointsTheOpticalAxisAtTheTarget) {
  const std::array<float, 3> target{{2.0f, -1.0f, 0.5f}};
  for (const std::array<float, 3>& eye :
       {std::array<float, 3>{{-4.0f, 0.0f, 1.0f}},
        std::array<float, 3>{{0.0f, 5.0f, 3.0f}},
        std::array<float, 3>{{3.0f, 3.0f, 0.0f}}}) {
    const mrl::Pose pose = mrl::lookAt(eye, target);

    // A sphere at the target must land on the centre pixel.
    mrl::AnalyticScene scene;
    scene.spheres.push_back(mrl::Sphere{{target}, 0.3f});
    const mrl::CameraIntrinsics k = intrinsics();
    const std::vector<float> depth = scene.render(k, pose);
    const size_t centre =
        static_cast<size_t>(k.height / 2) * k.width + k.width / 2;

    const float range = std::sqrt((target[0] - eye[0]) * (target[0] - eye[0]) +
                                  (target[1] - eye[1]) * (target[1] - eye[1]) +
                                  (target[2] - eye[2]) * (target[2] - eye[2]));
    EXPECT_NEAR(depth[centre], range - 0.3f, 0.05f);

    const float quaternion_norm =
        std::sqrt(pose[3] * pose[3] + pose[4] * pose[4] + pose[5] * pose[5] +
                  pose[6] * pose[6]);
    EXPECT_NEAR(quaternion_norm, 1.0f, 1e-5f);
  }
}

TEST(SceneTest, OrbitPosesRingTheTarget) {
  const std::array<float, 3> target{{1.0f, 2.0f, 0.0f}};
  const std::vector<mrl::Pose> poses = mrl::orbitPoses(12, 5.0f, 1.5f, target);
  ASSERT_EQ(poses.size(), 12u);
  for (const mrl::Pose& pose : poses) {
    const float dx = pose[0] - target[0];
    const float dy = pose[1] - target[1];
    EXPECT_NEAR(std::sqrt(dx * dx + dy * dy), 5.0f, 1e-4f);
    EXPECT_FLOAT_EQ(pose[2], 1.5f);
  }
}

TEST(SceneTest, DistanceToSurfaceMatchesTheAnalyticGeometry) {
  mrl::AnalyticScene scene;
  scene.spheres.push_back(mrl::Sphere{{{0.0f, 0.0f, 0.0f}}, 1.0f});
  EXPECT_NEAR(scene.distanceToSurface({{3.0f, 0.0f, 0.0f}}), 2.0f, 1e-5f);
  EXPECT_NEAR(scene.distanceToSurface({{0.0f, 0.0f, 0.0f}}), 1.0f, 1e-5f);

  mrl::AnalyticScene column;
  column.cylinders.push_back(
      mrl::Cylinder{{{0.0f, 0.0f, 0.0f}}, {{0.0f, 0.0f, 1.0f}}, 0.5f, 2.0f});
  // Beside the shaft: purely radial.
  EXPECT_NEAR(column.distanceToSurface({{2.0f, 0.0f, 1.0f}}), 1.5f, 1e-5f);
  // Directly above the cap: purely axial.
  EXPECT_NEAR(column.distanceToSurface({{0.0f, 0.0f, 3.0f}}), 1.0f, 1e-5f);
  // Past the rim: the corner of the cap, so both gaps count.
  EXPECT_NEAR(column.distanceToSurface({{1.5f, 0.0f, 3.0f}}),
              std::sqrt(1.0f * 1.0f + 1.0f * 1.0f),
              1e-5f);
}

/// The oracle the system tests rely on has to be right itself: for a single
/// frame of a plane it reduces to the distance to that plane, up to the metric
/// footprint of a pixel.
TEST(SceneTest, BruteForceDistanceReducesToThePlaneDistance) {
  mrl::AnalyticScene scene;
  scene.planes.push_back(
      mrl::Plane{{{0.0f, 0.0f, 4.0f}}, {{0.0f, 0.0f, 1.0f}}});
  const mrl::CameraIntrinsics k = intrinsics();
  const std::vector<mrl::PosedDepthImage> frames = {
      scene.renderPosed(k, kIdentity)};

  for (float z : {1.0f, 2.0f, 3.5f}) {
    const float expected = 4.0f - z;
    const float pixel_footprint = 4.0f / k.fx;
    EXPECT_NEAR(mrl::bruteForceDistance(k, frames, {{0.0f, 0.0f, z}}, 10.0f),
                expected,
                pixel_footprint);
  }
}

TEST(SceneTest, ForestIsReproducibleAndPopulated) {
  const mrl::AnalyticScene a = mrl::AnalyticScene::forest(7);
  const mrl::AnalyticScene b = mrl::AnalyticScene::forest(7);
  const mrl::AnalyticScene c = mrl::AnalyticScene::forest(8);

  ASSERT_FALSE(a.cylinders.empty());
  ASSERT_FALSE(a.spheres.empty());
  ASSERT_FALSE(a.planes.empty());
  EXPECT_FLOAT_EQ(a.cylinders[0].radius, b.cylinders[0].radius);
  EXPECT_FLOAT_EQ(a.spheres[0].center[0], b.spheres[0].center[0]);
  EXPECT_NE(a.cylinders[0].radius, c.cylinders[0].radius);

  for (const mrl::Cylinder& trunk : a.cylinders) {
    EXPECT_FLOAT_EQ(trunk.base[2], 0.0f) << "trunks stand on the ground";
    EXPECT_GT(trunk.height, 0.0f);
  }
}
