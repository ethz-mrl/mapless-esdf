#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <mrl/cuda_depth_buffer/scene.hpp>
#include <random>
#include <vector>

// The depth buffer is FP32-only.

namespace {

/// One optimistic-or-conservative query of a single point, unpacked into the
/// four scalars most of these tests care about.
struct SingleResult {
  float distance = 0.0f;
  float x = 0.0f, y = 0.0f, z = 0.0f;
  float gx = 0.0f, gy = 0.0f, gz = 0.0f;
};

SingleResult querySingle(
    mrl::CudaDepthBuffer& buffer,
    float qx,
    float qy,
    float qz,
    bool conservative,
    float truncation_radius = std::numeric_limits<float>::infinity(),
    mrl::DebugDistances* debug = nullptr) {
  const std::vector<float> q = {qx, qy, qz};
  mrl::QueryResults r;
  buffer.query(q.data(),
               1,
               conservative,
               r,
               truncation_radius,
               false,
               0.0f,
               debug);
  const auto& d =
      conservative ? r.conservative_distance : r.optimistic_distance;
  const auto& p = conservative ? r.conservative_closest : r.optimistic_closest;
  const auto& g =
      conservative ? r.conservative_gradient : r.optimistic_gradient;
  return {d[0], p[0], p[1], p[2], g[0], g[1], g[2]};
}

}  // namespace

class CudaDepthBufferTest : public ::testing::Test {
 protected:
  const size_t BUFFER_CAPACITY = 20;
  const size_t IMAGE_ROWS = 100;
  const size_t IMAGE_COLS = 100;
  const size_t IMAGE_SIZE = IMAGE_ROWS * IMAGE_COLS;

  // Camera Intrinsics
  float FX = 50.0f;
  float FY = 50.0f;
  // On a pixel centre, so an on-axis query has an exactly on-axis nearest ray.
  float CX = static_cast<float>(IMAGE_COLS) / 2.0f;
  float CY = static_cast<float>(IMAGE_ROWS) / 2.0f;
  float MIN_DEPTH = 0.1f;
  float MAX_DEPTH = 10.0f;

  mrl::CameraIntrinsics intrinsics() const {
    mrl::CameraIntrinsics k;
    k.cx = CX;
    k.cy = CY;
    k.fx = FX;
    k.fy = FY;
    k.width = static_cast<uint16_t>(IMAGE_COLS);
    k.height = static_cast<uint16_t>(IMAGE_ROWS);
    k.z_near = MIN_DEPTH;
    k.z_far = MAX_DEPTH;
    return k;
  }

  mrl::BufferParams params() const {
    mrl::BufferParams p;
    p.num_frames = BUFFER_CAPACITY;
    // The geometry tests are about distances, not about uncertainty; the tests
    // below that care set a confidence explicitly. They are written against the
    // exact pixel patch; the tests about the compensation modes set their own.
    p.pose_covariance_confidence = 0.0f;
    p.pool_compensation = mrl::PoolCompensation::kExact;
    return p;
  }

  mrl::CudaDepthBuffer* depth_buffer_ = nullptr;
  std::vector<float> d_depth_image;
  std::vector<float> d_pose;

  mrl::QueryResults results;

  void SetUp() override {
    // Initialize the depth buffer
    depth_buffer_ = new mrl::CudaDepthBuffer(intrinsics(), params());

    // Allocate host-side buffers
    d_depth_image.resize(IMAGE_SIZE);
    d_pose.resize(7);  // tx, ty, tz, qx, qy, qz, qw
  }

  void TearDown() override { delete depth_buffer_; }

  void fillPose(float tx,
                float ty,
                float tz,
                float qx,
                float qy,
                float qz,
                float qw) {
    d_pose[0] = tx;
    d_pose[1] = ty;
    d_pose[2] = tz;
    d_pose[3] = qx;
    d_pose[4] = qy;
    d_pose[5] = qz;
    d_pose[6] = qw;
  }

  void fillDepth(float val) {
    std::fill(d_depth_image.begin(), d_depth_image.end(), val);
  }

  void insertFrame(int64_t stamp_ns, bool is_keyframe = true) {
    depth_buffer_->setPosedDepthImage(d_depth_image.data(),
                                      d_pose.data(),
                                      stamp_ns,
                                      is_keyframe);
  }

  // Helper to push the same depth image and pose for all frames
  void pushAllFrames(float depth_val) {
    fillDepth(depth_val);
    for (size_t i = 0; i < BUFFER_CAPACITY; ++i) {
      depth_buffer_->setPosedDepthImage(d_depth_image.data(),
                                        d_pose.data(),
                                        static_cast<int64_t>(i));
    }
  }

  /// Optimistic query of a single point, results kept in `results`.
  void callGetClosestPoint(float qx, float qy, float qz) {
    const std::vector<float> q = {qx, qy, qz};
    depth_buffer_->query(q.data(), 1, false, results);
  }

  void callGetClosestPointMultiple(const std::vector<float>& queries) {
    depth_buffer_->query(queries.data(),
                         static_cast<uint32_t>(queries.size() / 3),
                         false,
                         results);
  }

  float dist(size_t i = 0) const { return results.optimistic_distance[i]; }
  float px(size_t i = 0) const { return results.optimistic_closest[3 * i + 0]; }
  float py(size_t i = 0) const { return results.optimistic_closest[3 * i + 1]; }
  float pz(size_t i = 0) const { return results.optimistic_closest[3 * i + 2]; }

  // Helper to check results against ground truth
  void checkResult(float gt_x,
                   float gt_y,
                   float gt_z,
                   float gt_dist,
                   float depth) {
    // tolerance in x and y is two unprojected pixels at gt_z
    float tol = (2.f * depth) / FX;
    tol = std::max(tol, static_cast<float>(1e-5));
    EXPECT_NEAR(px(), gt_x, tol);
    EXPECT_NEAR(py(), gt_y, tol);
    EXPECT_NEAR(pz(), gt_z, tol);
    EXPECT_NEAR(dist(), gt_dist, tol);
  }

  void checkResults(const std::vector<float>& gt_points,
                    const std::vector<float>& gt_dist,
                    float depth) {
    float tol = depth / FX;
    tol = std::max(tol, static_cast<float>(1e-5));
    for (size_t i = 0; i < gt_dist.size(); ++i) {
      EXPECT_NEAR(px(i), gt_points[3 * i + 0], tol) << "query " << i;
      EXPECT_NEAR(py(i), gt_points[3 * i + 1], tol) << "query " << i;
      EXPECT_NEAR(pz(i), gt_points[3 * i + 2], tol) << "query " << i;
      EXPECT_NEAR(dist(i), gt_dist[i], tol) << "query " << i;
    }
  }

  void quaternionRotate(float qx,
                        float qy,
                        float qz,
                        float qw,
                        float px,
                        float py,
                        float pz,
                        float& rx,
                        float& ry,
                        float& rz) {
    // most efficient quaternion rotation as derived here:
    // https://math.stackexchange.com/questions/2583669/quaternion-rotation-formula-algebra
    float two = static_cast<float>(2.0);
    float cx = two * (qy * pz - qz * py);
    float cy = two * (qz * px - qx * pz);
    float cz = two * (qx * py - qy * px);

    rx = px + qw * cx + qy * cz - qz * cy;
    ry = py + qw * cy + qz * cx - qx * cz;
    rz = pz + qw * cz + qx * cy - qy * cx;
  }

  std::array<double, 36> makeDiagonalCovariance(double pos_var,
                                                double rot_var) {
    std::array<double, 36> covariance{};
    covariance[0] = pos_var;
    covariance[7] = pos_var;
    covariance[14] = pos_var;
    covariance[21] = rot_var;
    covariance[28] = rot_var;
    covariance[35] = rot_var;
    return covariance;
  }

  void expectCovarianceEquals(int64_t stamp_ns,
                              const std::array<double, 36>& expected) {
    std::array<double, 36> actual{};
    ASSERT_TRUE(
        depth_buffer_->getCovarianceByTimestamp(stamp_ns, actual.data()));
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_DOUBLE_EQ(actual[i], expected[i]);
    }
  }
};

/**
 * @test Invalid depth carries no range information and must land on a clipping
 * plane, never at the camera origin: the near plane by default (an obstacle as
 * close as the sensor can see) and the far plane when invalid_is_farplane is
 * set.
 */
TEST_F(CudaDepthBufferTest, InvalidDepthMapsToNearPlane) {
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  pushAllFrames(0.0f);

  callGetClosestPoint(0.0f, 0.0f, 1.0f);
  EXPECT_NEAR(pz(), MIN_DEPTH, 1e-5f);
  EXPECT_NEAR(dist(), 1.0f - MIN_DEPTH, 1e-4f);
}

TEST_F(CudaDepthBufferTest, InvalidDepthMapsToFarPlaneWhenRequested) {
  mrl::CameraIntrinsics buffer_k;
  buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
  buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
  buffer_k.cx = CX;
  buffer_k.cy = CY;
  buffer_k.fx = FX;
  buffer_k.fy = FY;
  buffer_k.z_near = MIN_DEPTH;
  buffer_k.z_far = MAX_DEPTH;
  mrl::BufferParams buffer_p;
  buffer_p.pool_compensation =
      mrl::PoolCompensation::kExact;  // written against the exact patch
  buffer_p.num_frames = 1;
  buffer_p.invalid_is_farplane = true;
  mrl::CudaDepthBuffer buffer(buffer_k, buffer_p);
  fillDepth(0.0f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  buffer.setPosedDepthImage(d_depth_image.data(), d_pose.data(), 1, true);

  const SingleResult single1 = querySingle(buffer, 0.0f, 0.0f, 1.0f, false);
  EXPECT_NEAR(single1.z, MAX_DEPTH, 1e-4f);
  EXPECT_NEAR(single1.distance, MAX_DEPTH - 1.0f, 1e-4f);
}

/**
 * @test NaN is as uninformative as a zero and must take the same path.
 */
TEST_F(CudaDepthBufferTest, NanDepthIsTreatedAsInvalid) {
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  pushAllFrames(std::numeric_limits<float>::quiet_NaN());

  callGetClosestPoint(0.0f, 0.0f, 1.0f);
  EXPECT_NEAR(pz(), MIN_DEPTH, 1e-5f);
}

/**
 * @test Depth outside the clipping range is clipped to the nearer plane, so a
 * single hole cannot pull the field towards the camera and a sky pixel reads as
 * a surface at the far plane. The min pool must apply the same rule as the
 * unpooled path.
 */
TEST_F(CudaDepthBufferTest, ClippingIsIndependentOfMinPooling) {
  for (uint16_t pool : {uint16_t{1}, uint16_t{2}}) {
    mrl::CameraIntrinsics buffer_k;
    buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
    buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
    buffer_k.cx = CX;
    buffer_k.cy = CY;
    buffer_k.fx = FX;
    buffer_k.fy = FY;
    buffer_k.z_near = MIN_DEPTH;
    buffer_k.z_far = MAX_DEPTH;
    mrl::BufferParams buffer_p;
    buffer_p.pool_compensation =
        mrl::PoolCompensation::kExact;  // written against the exact patch
    buffer_p.num_frames = 1;
    buffer_p.min_pool_size = pool;
    mrl::CudaDepthBuffer buffer(buffer_k, buffer_p);
    fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);

    // beyond the far plane, including a non-finite reading
    fillDepth(MAX_DEPTH + 5.0f);
    d_depth_image[0] = std::numeric_limits<float>::infinity();
    buffer.setPosedDepthImage(d_depth_image.data(), d_pose.data(), 1, true);
    const SingleResult single2 = querySingle(buffer, 0.0f, 0.0f, 1.0f, false);
    EXPECT_NEAR(single2.z, MAX_DEPTH, 1e-4f) << "pool = " << pool;
    EXPECT_NEAR(single2.distance, MAX_DEPTH - 1.0f, 1e-4f) << "pool = " << pool;

    // below the near plane, but still a measurement
    fillDepth(MIN_DEPTH * 0.5f);
    buffer.setPosedDepthImage(d_depth_image.data(), d_pose.data(), 2, true);
    const SingleResult single3 = querySingle(buffer, 0.0f, 0.0f, 1.0f, false);
    EXPECT_NEAR(single3.z, MIN_DEPTH, 1e-4f) << "pool = " << pool;
  }
}

/**
 * @test An invalid pixel contributes a point on its own ray at the near plane,
 * not at the camera origin. The origin is not on any pixel ray and would make
 * every hole report a distance of |q| regardless of where the hole is in the
 * image.
 */
TEST_F(CudaDepthBufferTest, InvalidPixelStaysOnItsOwnRay) {
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  fillDepth(3.0f);
  d_depth_image[0] = 0.0f;  // corner pixel (u = 0, v = 0)
  for (size_t i = 0; i < BUFFER_CAPACITY; ++i) {
    depth_buffer_->setPosedDepthImage(d_depth_image.data(),
                                      d_pose.data(),
                                      static_cast<int64_t>(i));
  }

  // Pixels are patches (PoolCompensation::kExact): the hole's closest point is
  // the query clamped into the corner pixel's patch [-1/2, 1/2) x [-1/2, 1/2)
  // at MIN_DEPTH - still on that pixel, never at the camera origin.
  const float expected_x = std::clamp(0.0f,
                                      (-0.5f - CX) * MIN_DEPTH / FX,
                                      (0.5f - CX) * MIN_DEPTH / FX);
  const float expected_y = std::clamp(0.0f,
                                      (-0.5f - CY) * MIN_DEPTH / FY,
                                      (0.5f - CY) * MIN_DEPTH / FY);
  callGetClosestPoint(0.0f, 0.0f, 1.0f);
  EXPECT_NEAR(px(), expected_x, 1e-4f);
  EXPECT_NEAR(py(), expected_y, 1e-4f);
  EXPECT_NEAR(pz(), MIN_DEPTH, 1e-4f);
  EXPECT_NEAR(dist(),
              std::sqrt(expected_x * expected_x + expected_y * expected_y +
                        (1.0f - MIN_DEPTH) * (1.0f - MIN_DEPTH)),
              1e-4f);
}

/**
 * @test With invalid_is_farplane the same hole yields the far plane instead, so
 * the observed surface in front of it wins.
 */
TEST_F(CudaDepthBufferTest, InvalidPixelYieldsToSurfaceWhenMappedToFarPlane) {
  mrl::CameraIntrinsics buffer_k;
  buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
  buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
  buffer_k.cx = CX;
  buffer_k.cy = CY;
  buffer_k.fx = FX;
  buffer_k.fy = FY;
  buffer_k.z_near = MIN_DEPTH;
  buffer_k.z_far = MAX_DEPTH;
  mrl::BufferParams buffer_p;
  buffer_p.pool_compensation =
      mrl::PoolCompensation::kExact;  // written against the exact patch
  buffer_p.num_frames = 1;
  buffer_p.invalid_is_farplane = true;
  mrl::CudaDepthBuffer buffer(buffer_k, buffer_p);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  fillDepth(3.0f);
  d_depth_image[0] = 0.0f;
  buffer.setPosedDepthImage(d_depth_image.data(), d_pose.data(), 1, true);

  const SingleResult single4 = querySingle(buffer, 0.0f, 0.0f, 1.0f, false);
  EXPECT_NEAR(single4.z, 3.0f, 1e-4f);
  EXPECT_NEAR(single4.distance, 2.0f, 1e-3f);
}

/**
 * @test Test 2: All ones depth, identity pose.
 * This creates a point cloud plane at z=1.0 in the camera (and world) frame.
 * We test queries against this plane. The closest point will be the
 * unprojection of the pixel closest to the query's (x,y) coordinates.
 */
TEST_F(CudaDepthBufferTest, AllOnesIdentityPose) {
  // tol is unprojected pixel size at depth 1.0
  float depth = 1.0f;

  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  pushAllFrames(depth);

  // Test 1: Query (0, 0, 0)
  // The closest point on the infinite plane z=1 is (0,0,1) and the sign should
  // be negative, as it's behind the obstacle
  callGetClosestPoint(0.0f, 0.0f, 0.0f);
  float gt_x = 0.0f;
  float gt_y = 0.0f;
  float gt_z = 1.0f;
  float gt_dist = std::sqrt(gt_x * gt_x + gt_y * gt_y + gt_z * gt_z);

  // The result point is in the "current" frame, which is also identity.
  checkResult(gt_x, gt_y, gt_z, gt_dist, depth);
}

/**
 * @test Test 3: All pixels have same random depth and same random pose. Then we
 * use a cropped plane to find the ground truth closest point.
 */
TEST_F(CudaDepthBufferTest, AllRandomDepthRandomPose) {
  srand(static_cast<unsigned int>(time(nullptr)));
  // tol is unprojected pixel size at depth
  float depth = static_cast<float>(rand()) / RAND_MAX + 0.5f;
  //   float depth = 1.34019;

  // sample a random translation
  float tx = static_cast<float>(rand()) / RAND_MAX - 0.5f;
  float ty = static_cast<float>(rand()) / RAND_MAX - 0.5f;
  float tz = static_cast<float>(rand()) / RAND_MAX - 0.5f;

  // Sample a random quaternion according to
  // https://en.wikipedia.org/wiki/Quaternion#Uniform_random_rotation
  float u1 = static_cast<float>(rand()) / RAND_MAX;
  float u2 = static_cast<float>(rand()) / RAND_MAX;
  float u3 = static_cast<float>(rand()) / RAND_MAX;
  float qx = static_cast<float>(std::sqrt(1 - u1) * std::sin(2 * M_PI * u2));
  float qy = static_cast<float>(std::sqrt(1 - u1) * std::cos(2 * M_PI * u2));
  float qz = static_cast<float>(std::sqrt(u1) * std::sin(2 * M_PI * u3));
  float qw = static_cast<float>(std::sqrt(u1) * std::cos(2 * M_PI * u3));

  // sample a random query point in world frame
  float query_x = 10 * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
  float query_y = 10 * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
  float query_z = 10 * (static_cast<float>(rand()) / RAND_MAX - 0.5f);

  // transform query point to camera frame for ground truth calculation
  float p_rel_x = query_x - tx;
  float p_rel_y = query_y - ty;
  float p_rel_z = query_z - tz;
  float p_cam_x, p_cam_y, p_cam_z;
  quaternionRotate(-qx,
                   -qy,
                   -qz,
                   qw,
                   p_rel_x,
                   p_rel_y,
                   p_rel_z,
                   p_cam_x,
                   p_cam_y,
                   p_cam_z);

  fillPose(tx, ty, tz, qx, qy, qz, qw);
  pushAllFrames(depth);
  callGetClosestPoint(query_x, query_y, query_z);

  // project onto depth plane along z vector
  float p_cam_z_closest = depth;

  // project point onto image bounds
  float max_x = depth * ((static_cast<float>(IMAGE_COLS) - 1.0f) - CX) / FX;
  float min_x = depth * -CX / FX;
  float max_y = depth * ((static_cast<float>(IMAGE_ROWS) - 1.0f) - CY) / FY;
  float min_y = depth * -CY / FY;
  float p_cam_x_closest = std::min(std::max(p_cam_x, min_x), max_x);
  float p_cam_y_closest = std::min(std::max(p_cam_y, min_y), max_y);

  float gt_dist = std::sqrt(std::pow(p_cam_x_closest - p_cam_x, 2) +
                            std::pow(p_cam_y_closest - p_cam_y, 2) +
                            std::pow(p_cam_z_closest - p_cam_z, 2));

  // Results come back in the reference frame, so the expected point has to be
  // rotated out of the camera frame the plane was reasoned about in.
  float gt_x, gt_y, gt_z;
  quaternionRotate(qx,
                   qy,
                   qz,
                   qw,
                   p_cam_x_closest,
                   p_cam_y_closest,
                   p_cam_z_closest,
                   gt_x,
                   gt_y,
                   gt_z);
  checkResult(gt_x + tx, gt_y + ty, gt_z + tz, gt_dist, depth);
}

/**
 * @test All ones depth, moving pose.
 * Pose for frame i is (0, 0, i+1). All frames have depth 1.
 * World plane for frame i is at z = (1.0) + (i+1) = i+2.
 * Query is (0,0,0). The closest plane is from frame i=0 (z=2).
 */
TEST_F(CudaDepthBufferTest, AllOnesMovingPose) {
  float depth = 1.0f;
  fillDepth(depth);
  for (size_t i = 0; i < BUFFER_CAPACITY; ++i) {
    // Pose is (0, 0, i+1, 0, 0, 0, 1)
    fillPose(0.0f, 0.0f, static_cast<float>(i) + 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    depth_buffer_->setPosedDepthImage(d_depth_image.data(),
                                      d_pose.data(),
                                      static_cast<int64_t>(i));
  }

  // Frame i sits at z = i + 1 and sees a plane one meter ahead, so the planes
  // are at z = 2 .. BUFFER_CAPACITY + 1. The query sits at z = 1.
  callGetClosestPoint(0.0f, 0.0f, 1.0f);
  checkResult(0.0f, 0.0f, 2.0f, 1.0f, depth);
}

/**
 * @test Test 4: All ones depth, moving pose.
 * Pose for frame i is (0, 0, i+1). All frames have depth 1.
 * World plane for frame i is at z = (1.0) + (i+1) = i+2.
 * Queries are (0,0,0) and (0,0,-1) in world frame.
 * The closest plane is from frame i=0 (z=2).
 */
TEST_F(CudaDepthBufferTest, MultipleQueries) {
  float depth = 1.0f;
  fillDepth(depth);
  for (size_t i = 0; i < BUFFER_CAPACITY; ++i) {
    // Pose is (0, 0, i+1, 0, 0, 0, 1)
    fillPose(0.0f, 0.0f, static_cast<float>(i), 0.0f, 0.0f, 0.0f, 1.0f);
    depth_buffer_->setPosedDepthImage(d_depth_image.data(),
                                      d_pose.data(),
                                      static_cast<int64_t>(i));
  }

  // Frame i sits at z = i and sees a plane one meter in front of it, so the
  // reference frame planes are at z = 1 .. BUFFER_CAPACITY. Both queries are
  // closest to the plane of the first frame.
  callGetClosestPointMultiple({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f});
  checkResults({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}, depth);
}

TEST_F(CudaDepthBufferTest, SuppressionLogic) {
  float depth = 1.0f;
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  pushAllFrames(depth);

  // Define queries
  // Bunch 1: Q0 kept, Q1, Q2 suppressed
  // Q0: (0, 0, 0.5)
  // Q1: (0.01, 0, 0.5) -> dist to Q0 = 0.01
  // Q2: (0.02, 0, 0.5) -> dist to Q0 = 0.02

  // Bunch 2: Q3 kept, Q4 suppressed
  // Q3: (0.2, 0, 0.5)  -> dist to Q0 = 0.2 > 0.05
  // Q4: (0.21, 0, 0.5) -> dist to Q3 = 0.01

  // Bunch 3: Q5 kept, Q6, Q7, Q8 suppressed (Dense bunch)
  // Q5: (0.5, 0, 0.5)
  // Q6: (0.501, 0, 0.5)
  // Q7: (0.502, 0, 0.5)
  // Q8: (0.503, 0, 0.5)

  std::vector<float> queries;
  for (float x :
       {0.0f, 0.01f, 0.02f, 0.2f, 0.21f, 0.5f, 0.501f, 0.502f, 0.503f}) {
    queries.insert(queries.end(), {x, 0.0f, 0.5f});
  }

  const float suppression_radius = 0.05f;
  const float truncation_radius = 10.0f;

  depth_buffer_->query(queries.data(),
                       static_cast<uint32_t>(queries.size() / 3),
                       false,
                       results,
                       truncation_radius,
                       false,
                       suppression_radius);

  auto checkValid = [&](int idx, float expected_dist) {
    EXPECT_FALSE(std::isnan(px(idx))) << "Query " << idx << " should be valid";
    EXPECT_NEAR(dist(idx), expected_dist, 1e-4)
        << "Query " << idx << " dist mismatch";
  };
  auto checkSuppressed = [&](int idx) {
    EXPECT_TRUE(std::isnan(px(idx)))
        << "Query " << idx << " should be suppressed (NaN)";
    EXPECT_NEAR(dist(idx), truncation_radius, 1e-4)
        << "Query " << idx << " should have truncation dist";
  };

  // Q0 Kept
  checkValid(0, 0.5f);
  // Q1, Q2 Suppressed
  checkSuppressed(1);
  checkSuppressed(2);

  // Q3 Kept
  checkValid(3, 0.5f);
  // Q4 Suppressed
  checkSuppressed(4);

  // Q5 Kept
  checkValid(5, 0.5f);
  // Q6, Q7, Q8 Suppressed
  checkSuppressed(6);
  checkSuppressed(7);
  checkSuppressed(8);
}

TEST_F(CudaDepthBufferTest, PoseAndCovarianceUpdateByTimestamp) {
  constexpr int64_t kHistoricalStampNs = 101;
  constexpr int64_t kCurrentStampNs = 202;
  fillDepth(1.0f);

  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  insertFrame(kHistoricalStampNs);

  std::array<double, 36> covariance = makeDiagonalCovariance(0.25, 0.75);
  std::array<double, 36> covariance_out{};
  EXPECT_FALSE(depth_buffer_->getCovarianceByTimestamp(kHistoricalStampNs,
                                                       covariance_out.data()));

  fillPose(0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  insertFrame(kCurrentStampNs);

  callGetClosestPoint(0.0f, 0.0f, 1.5f);
  checkResult(0.0f, 0.0f, 1.0f, 0.5f, 1.0f);

  fillPose(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  EXPECT_TRUE(
      depth_buffer_->updatePoseAndCovarianceByTimestamp(kHistoricalStampNs,
                                                        d_pose.data(),
                                                        covariance.data()));
  expectCovarianceEquals(kHistoricalStampNs, covariance);

  // The revised pose moves that frame's plane from z = 1 to z = 6.
  callGetClosestPoint(0.0f, 0.0f, 1.5f);
  checkResult(0.0f, 0.0f, 6.0f, 4.5f, 1.0f);
}

TEST_F(CudaDepthBufferTest, UnmatchedTimestampDoesNotModifyState) {
  constexpr int64_t kHistoricalStampNs = 303;
  constexpr int64_t kCurrentStampNs = 404;
  fillDepth(1.0f);

  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  insertFrame(kHistoricalStampNs);
  fillPose(0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  insertFrame(kCurrentStampNs);

  callGetClosestPoint(0.0f, 0.0f, 1.5f);
  checkResult(0.0f, 0.0f, 1.0f, 0.5f, 1.0f);

  std::array<double, 36> covariance = makeDiagonalCovariance(1.0, 2.0);
  fillPose(0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  EXPECT_FALSE(
      depth_buffer_->updatePoseAndCovarianceByTimestamp(999,
                                                        d_pose.data(),
                                                        covariance.data()));

  std::array<double, 36> covariance_out{};
  EXPECT_FALSE(depth_buffer_->getCovarianceByTimestamp(kHistoricalStampNs,
                                                       covariance_out.data()));

  callGetClosestPoint(0.0f, 0.0f, 1.5f);
  checkResult(0.0f, 0.0f, 1.0f, 0.5f, 1.0f);
}

TEST_F(CudaDepthBufferTest, ReusedSlotClearsCovarianceState) {
  constexpr int64_t kOriginalStampNs = 505;
  fillDepth(1.0f);

  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  insertFrame(kOriginalStampNs);

  std::array<double, 36> covariance = makeDiagonalCovariance(3.0, 4.0);
  EXPECT_TRUE(
      depth_buffer_->updatePoseAndCovarianceByTimestamp(kOriginalStampNs,
                                                        d_pose.data(),
                                                        covariance.data()));
  expectCovarianceEquals(kOriginalStampNs, covariance);

  int64_t newest_stamp_ns = -1;
  for (size_t i = 0; i < BUFFER_CAPACITY; ++i) {
    newest_stamp_ns = 600 + static_cast<int64_t>(i);
    fillPose(0.0f, 0.0f, static_cast<float>(i + 1), 0.0f, 0.0f, 0.0f, 1.0f);
    insertFrame(newest_stamp_ns);
  }

  std::array<double, 36> covariance_out{};
  EXPECT_FALSE(depth_buffer_->getCovarianceByTimestamp(kOriginalStampNs,
                                                       covariance_out.data()));
  EXPECT_FALSE(depth_buffer_->getCovarianceByTimestamp(newest_stamp_ns,
                                                       covariance_out.data()));
}

TEST_F(CudaDepthBufferTest,
       CrossCompensationReducesDistanceAndPopulatesDebugChannel) {
  mrl::CameraIntrinsics compensation_buffer_k;
  compensation_buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
  compensation_buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
  compensation_buffer_k.cx = CX;
  compensation_buffer_k.cy = CY;
  compensation_buffer_k.fx = FX;
  compensation_buffer_k.fy = FY;
  compensation_buffer_k.z_near = MIN_DEPTH;
  compensation_buffer_k.z_far = MAX_DEPTH;
  mrl::BufferParams compensation_buffer_p;
  compensation_buffer_p.pool_compensation =
      mrl::PoolCompensation::kExact;  // written against the exact patch
  compensation_buffer_p.num_frames = 1;
  compensation_buffer_p.pose_covariance_confidence = 1.0f;
  mrl::CudaDepthBuffer compensation_buffer(compensation_buffer_k,
                                           compensation_buffer_p);

  fillDepth(1.0f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  compensation_buffer.setPosedDepthImage(d_depth_image.data(),
                                         d_pose.data(),
                                         111,
                                         true);

  const std::array<double, 36> covariance = makeDiagonalCovariance(0.25, 0.0);
  ASSERT_TRUE(compensation_buffer.updatePoseAndCovarianceByTimestamp(
      111,
      d_pose.data(),
      covariance.data()));

  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  mrl::DebugDistances debug_distances;
  const SingleResult single5 =
      querySingle(compensation_buffer,
                  qx,
                  qy,
                  qz,
                  false,
                  std::numeric_limits<float>::infinity(),
                  &debug_distances);

  EXPECT_NEAR(single5.x, 0.0f, 1e-5f);
  EXPECT_NEAR(single5.y, 0.0f, 1e-5f);
  EXPECT_NEAR(single5.z, 1.0f, 1e-5f);
  EXPECT_NEAR(single5.distance, 0.5f, 1e-5f);
  ASSERT_EQ(debug_distances.pose_compensation.size(), 1u);
  EXPECT_NEAR(debug_distances.pose_compensation[0], 0.5f, 1e-5f);
  EXPECT_NEAR(debug_distances.merged[0], 0.5f, 1e-5f);
  EXPECT_NEAR(debug_distances.occupied[0], 0.5f, 1e-5f);
}

TEST_F(CudaDepthBufferTest, OverwrittenSlotClearsPoseCompensationCache) {
  mrl::CameraIntrinsics compensation_buffer_k;
  compensation_buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
  compensation_buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
  compensation_buffer_k.cx = CX;
  compensation_buffer_k.cy = CY;
  compensation_buffer_k.fx = FX;
  compensation_buffer_k.fy = FY;
  compensation_buffer_k.z_near = MIN_DEPTH;
  compensation_buffer_k.z_far = MAX_DEPTH;
  mrl::BufferParams compensation_buffer_p;
  compensation_buffer_p.pool_compensation =
      mrl::PoolCompensation::kExact;  // written against the exact patch
  compensation_buffer_p.num_frames = 1;
  compensation_buffer_p.pose_covariance_confidence = 1.0f;
  mrl::CudaDepthBuffer compensation_buffer(compensation_buffer_k,
                                           compensation_buffer_p);

  fillDepth(1.0f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  compensation_buffer.setPosedDepthImage(d_depth_image.data(),
                                         d_pose.data(),
                                         222,
                                         true);

  const std::array<double, 36> covariance = makeDiagonalCovariance(0.25, 0.0);
  ASSERT_TRUE(compensation_buffer.updatePoseAndCovarianceByTimestamp(
      222,
      d_pose.data(),
      covariance.data()));

  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  const SingleResult single6 =
      querySingle(compensation_buffer, qx, qy, qz, false);
  EXPECT_NEAR(single6.distance, 0.5f, 1e-5f);

  compensation_buffer.setPosedDepthImage(d_depth_image.data(),
                                         d_pose.data(),
                                         333,
                                         true);
  const SingleResult single7 =
      querySingle(compensation_buffer, qx, qy, qz, false);
  EXPECT_NEAR(single7.distance, 1.0f, 1e-5f);

  std::array<double, 36> covariance_out{};
  EXPECT_FALSE(
      compensation_buffer.getCovarianceByTimestamp(222, covariance_out.data()));
  EXPECT_FALSE(
      compensation_buffer.getCovarianceByTimestamp(333, covariance_out.data()));
}

TEST_F(CudaDepthBufferTest,
       ExpandedSearchTruncationReclampsFinalDistanceToGlobalBound) {
  mrl::CameraIntrinsics compensation_buffer_k;
  compensation_buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
  compensation_buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
  compensation_buffer_k.cx = CX;
  compensation_buffer_k.cy = CY;
  compensation_buffer_k.fx = FX;
  compensation_buffer_k.fy = FY;
  compensation_buffer_k.z_near = MIN_DEPTH;
  compensation_buffer_k.z_far = MAX_DEPTH;
  mrl::BufferParams compensation_buffer_p;
  compensation_buffer_p.pool_compensation =
      mrl::PoolCompensation::kExact;  // written against the exact patch
  compensation_buffer_p.num_frames = 1;
  compensation_buffer_p.pose_covariance_confidence = 1.0f;
  mrl::CudaDepthBuffer compensation_buffer(compensation_buffer_k,
                                           compensation_buffer_p);

  fillDepth(1.5f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  compensation_buffer.setPosedDepthImage(d_depth_image.data(),
                                         d_pose.data(),
                                         444,
                                         true);

  const std::array<double, 36> covariance = makeDiagonalCovariance(0.25, 0.0);
  ASSERT_TRUE(compensation_buffer.updatePoseAndCovarianceByTimestamp(
      444,
      d_pose.data(),
      covariance.data()));

  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  mrl::DebugDistances debug_distances;
  const SingleResult single8 = querySingle(compensation_buffer,
                                           qx,
                                           qy,
                                           qz,
                                           false,
                                           1.0f,
                                           &debug_distances);

  EXPECT_TRUE(std::isnan(single8.x));
  EXPECT_TRUE(std::isnan(single8.y));
  EXPECT_TRUE(std::isnan(single8.z));
  EXPECT_NEAR(single8.distance, 1.0f, 1e-5f);
  ASSERT_EQ(debug_distances.pose_compensation.size(), 1u);
  EXPECT_NEAR(debug_distances.pose_compensation[0], 0.5f, 1e-5f);
  EXPECT_NEAR(debug_distances.merged[0], 1.0f, 1e-5f);
  EXPECT_NEAR(debug_distances.unknown[0], 1.0f, 1e-5f);
}

TEST_F(CudaDepthBufferTest,
       PerFrameCovarianceChangesEffectiveSearchTruncation) {
  mrl::CameraIntrinsics compensation_buffer_k;
  compensation_buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
  compensation_buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
  compensation_buffer_k.cx = CX;
  compensation_buffer_k.cy = CY;
  compensation_buffer_k.fx = FX;
  compensation_buffer_k.fy = FY;
  compensation_buffer_k.z_near = MIN_DEPTH;
  compensation_buffer_k.z_far = MAX_DEPTH;
  mrl::BufferParams compensation_buffer_p;
  compensation_buffer_p.pool_compensation =
      mrl::PoolCompensation::kExact;  // written against the exact patch
  compensation_buffer_p.num_frames = 2;
  compensation_buffer_p.pose_covariance_confidence = 1.0f;
  mrl::CudaDepthBuffer compensation_buffer(compensation_buffer_k,
                                           compensation_buffer_p);

  fillDepth(1.3f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  compensation_buffer.setPosedDepthImage(d_depth_image.data(),
                                         d_pose.data(),
                                         555,
                                         true);
  compensation_buffer.setPosedDepthImage(d_depth_image.data(),
                                         d_pose.data(),
                                         666,
                                         true);

  const std::array<double, 36> high_covariance =
      makeDiagonalCovariance(0.25, 0.0);
  const std::array<double, 36> low_covariance =
      makeDiagonalCovariance(0.01, 0.0);
  ASSERT_TRUE(compensation_buffer.updatePoseAndCovarianceByTimestamp(
      555,
      d_pose.data(),
      high_covariance.data()));
  ASSERT_TRUE(compensation_buffer.updatePoseAndCovarianceByTimestamp(
      666,
      d_pose.data(),
      low_covariance.data()));

  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  mrl::DebugDistances debug_distances;
  const SingleResult single9 = querySingle(compensation_buffer,
                                           qx,
                                           qy,
                                           qz,
                                           false,
                                           1.0f,
                                           &debug_distances);

  ASSERT_EQ(debug_distances.pose_compensation.size(), 2u);
  std::array<std::pair<float, float>, 2> compensation_and_merged = {{
      {debug_distances.pose_compensation[0], debug_distances.merged[0]},
      {debug_distances.pose_compensation[1], debug_distances.merged[1]},
  }};
  std::sort(
      compensation_and_merged.begin(),
      compensation_and_merged.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  EXPECT_NEAR(compensation_and_merged[0].first, 0.1f, 1e-5f);
  EXPECT_NEAR(compensation_and_merged[0].second, 1.0f, 1e-5f);
  EXPECT_NEAR(compensation_and_merged[1].first, 0.5f, 1e-5f);
  EXPECT_NEAR(compensation_and_merged[1].second, 0.8f, 1e-5f);
  EXPECT_NEAR(single9.x, 0.0f, 1e-5f);
  EXPECT_NEAR(single9.y, 0.0f, 1e-5f);
  EXPECT_NEAR(single9.z, 1.3f, 1e-5f);
  EXPECT_NEAR(single9.distance, 0.8f, 1e-5f);
}

TEST_F(CudaDepthBufferTest,
       ConservativeRobotDistanceUsesExpandedSearchTruncation) {
  mrl::CameraIntrinsics compensation_buffer_k;
  compensation_buffer_k.height = static_cast<uint16_t>(IMAGE_ROWS);
  compensation_buffer_k.width = static_cast<uint16_t>(IMAGE_COLS);
  compensation_buffer_k.cx = CX;
  compensation_buffer_k.cy = CY;
  compensation_buffer_k.fx = FX;
  compensation_buffer_k.fy = FY;
  compensation_buffer_k.z_near = MIN_DEPTH;
  compensation_buffer_k.z_far = MAX_DEPTH;
  mrl::BufferParams compensation_buffer_p;
  compensation_buffer_p.pool_compensation =
      mrl::PoolCompensation::kExact;  // written against the exact patch
  compensation_buffer_p.num_frames = 1;
  compensation_buffer_p.robot_radius = 2.0f;
  compensation_buffer_p.pose_covariance_confidence = 1.0f;
  mrl::CudaDepthBuffer compensation_buffer(compensation_buffer_k,
                                           compensation_buffer_p);

  fillDepth(1.0f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  compensation_buffer.setPosedDepthImage(d_depth_image.data(),
                                         d_pose.data(),
                                         777,
                                         true);

  const std::array<double, 36> covariance = makeDiagonalCovariance(0.25, 0.0);
  ASSERT_TRUE(compensation_buffer.updatePoseAndCovarianceByTimestamp(
      777,
      d_pose.data(),
      covariance.data()));

  float qx = 0.0f;
  float qy = 0.0f;
  float qz = 0.0f;
  mrl::DebugDistances debug_distances;
  const SingleResult single10 = querySingle(compensation_buffer,
                                            qx,
                                            qy,
                                            qz,
                                            true,
                                            1.0f,
                                            &debug_distances);
  EXPECT_NEAR(single10.distance, 1.0f, 1e-5f);
  ASSERT_EQ(debug_distances.robot.size(), 1u);
  EXPECT_NEAR(debug_distances.robot[0], 1.0f, 1e-5f);
  EXPECT_NEAR(debug_distances.merged[0], 1.0f, 1e-5f);
  EXPECT_NEAR(debug_distances.pose_compensation[0], 0.5f, 1e-5f);
}

/**
 * @test A covariance is what enables the margin: there is no separate switch.
 * Supplying one inflates that frame's obstacles by c standard deviations of its
 * pose uncertainty, and a confidence of zero turns it back off.
 */
TEST_F(CudaDepthBufferTest, ACovarianceInflatesObstacles) {
  mrl::BufferParams p = params();
  p.num_frames = 1;
  p.pose_covariance_confidence = 2.0f;
  mrl::CudaDepthBuffer buffer(intrinsics(), p);

  fillDepth(2.0f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  buffer.setPosedDepthImage(d_depth_image.data(), d_pose.data(), 1, true);

  // Without a covariance there is nothing to inflate by.
  EXPECT_NEAR(querySingle(buffer, 0.0f, 0.0f, 0.0f, false).distance,
              2.0f,
              1e-4f);

  // 0.09 m^2 of position variance, no rotation, queried at the camera centre:
  // the lever arm vanishes, so the margin is c * sqrt(0.09) = 0.6 m.
  const std::array<double, 36> covariance = makeDiagonalCovariance(0.09, 0.0);
  ASSERT_TRUE(buffer.updatePoseAndCovarianceByTimestamp(1,
                                                        d_pose.data(),
                                                        covariance.data()));
  EXPECT_NEAR(querySingle(buffer, 0.0f, 0.0f, 0.0f, false).distance,
              1.4f,
              1e-4f);

  // A larger lever arm sees more of the rotational uncertainty; here there is
  // none, so moving the query along the ray leaves the margin unchanged.
  EXPECT_NEAR(querySingle(buffer, 0.0f, 0.0f, 1.0f, false).distance,
              0.4f,
              1e-4f);

  buffer.setPoseCovarianceConfidence(0.0f);
  EXPECT_NEAR(querySingle(buffer, 0.0f, 0.0f, 0.0f, false).distance,
              2.0f,
              1e-4f);
}

/**
 * @test Rotational uncertainty grows with the lever arm, which is the whole
 * reason the margin is per query rather than per frame.
 */
TEST_F(CudaDepthBufferTest, MarginGrowsWithTheLeverArm) {
  mrl::BufferParams p = params();
  p.num_frames = 1;
  p.pose_covariance_confidence = 1.0f;
  mrl::CudaDepthBuffer buffer(intrinsics(), p);

  fillDepth(4.0f);
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  buffer.setPosedDepthImage(d_depth_image.data(), d_pose.data(), 1, true);
  const std::array<double, 36> covariance = makeDiagonalCovariance(0.0, 0.01);
  ASSERT_TRUE(buffer.updatePoseAndCovarianceByTimestamp(1,
                                                        d_pose.data(),
                                                        covariance.data()));

  // Rotational variance only, so the margin is c * |r| * sigma_theta.
  const float near_distance =
      querySingle(buffer, 0.0f, 0.0f, 1.0f, false).distance;
  const float far_distance =
      querySingle(buffer, 0.0f, 0.0f, 3.0f, false).distance;
  EXPECT_NEAR(near_distance, 3.0f - 1.0f * 0.1f, 1e-3f);
  EXPECT_NEAR(far_distance, 1.0f - 3.0f * 0.1f, 1e-3f);
}

/**
 * @test Both handlings of unknown space come back from one query. The
 * conservative channel is never more permissive than the optimistic one, and it
 * is left as NaN when the query did not ask for it.
 */
TEST_F(CudaDepthBufferTest, BothChannelsAreReportedTogether) {
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  pushAllFrames(2.0f);

  std::vector<float> queries;
  for (float z : {0.5f, 1.0f, 1.5f}) {
    for (float x : {-0.3f, 0.0f, 0.3f}) {
      queries.insert(queries.end(), {x, 0.0f, z});
    }
  }
  const uint32_t n = static_cast<uint32_t>(queries.size() / 3);

  mrl::QueryResults conservative;
  depth_buffer_->query(queries.data(), n, true, conservative, 5.0f);
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FALSE(std::isnan(conservative.optimistic_distance[i])) << i;
    EXPECT_FALSE(std::isnan(conservative.conservative_distance[i])) << i;
    // Treating unknown space as occupied can only bring an obstacle closer.
    EXPECT_LE(conservative.conservative_distance[i],
              conservative.optimistic_distance[i] + 1e-5f)
        << "query " << i;
  }

  mrl::QueryResults optimistic;
  depth_buffer_->query(queries.data(), n, false, optimistic, 5.0f);
  for (uint32_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(optimistic.optimistic_distance[i],
                    conservative.optimistic_distance[i])
        << "the optimistic channel must not depend on the conservative flag, "
           "query "
        << i;
    EXPECT_TRUE(std::isnan(optimistic.conservative_distance[i])) << i;
    EXPECT_TRUE(std::isnan(optimistic.conservative_gradient[3 * i])) << i;
    EXPECT_TRUE(std::isnan(optimistic.conservative_closest[3 * i])) << i;
  }
}

/**
 * @test The gradient is the unit vector from the closest point towards the
 * query, and it is NaN exactly where the distance saturated and no closest
 * point exists.
 */
TEST_F(CudaDepthBufferTest, GradientPointsAwayFromTheClosestPoint) {
  fillPose(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
  pushAllFrames(2.0f);

  const std::vector<float> queries = {0.0f,
                                      0.0f,
                                      1.0f,  // in front, free
                                      0.2f,
                                      -0.1f,
                                      1.4f,  // off axis
                                      0.0f,
                                      0.0f,
                                      3.0f};  // behind the plane
  mrl::QueryResults r;
  depth_buffer_->query(queries.data(), 3, false, r, 5.0f);

  for (int i = 0; i < 3; ++i) {
    const float gx = r.optimistic_gradient[3 * i + 0];
    const float gy = r.optimistic_gradient[3 * i + 1];
    const float gz = r.optimistic_gradient[3 * i + 2];
    ASSERT_FALSE(std::isnan(gx)) << "query " << i;
    EXPECT_NEAR(std::sqrt(gx * gx + gy * gy + gz * gz), 1.0f, 1e-4f) << i;

    const float dx = queries[3 * i + 0] - r.optimistic_closest[3 * i + 0];
    const float dy = queries[3 * i + 1] - r.optimistic_closest[3 * i + 1];
    const float dz = queries[3 * i + 2] - r.optimistic_closest[3 * i + 2];
    const float norm = std::sqrt(dx * dx + dy * dy + dz * dz);
    const float sign = r.optimistic_distance[i] < 0.0f ? -1.0f : 1.0f;
    EXPECT_NEAR(gx, sign * dx / norm, 1e-4f) << i;
    EXPECT_NEAR(gy, sign * dy / norm, 1e-4f) << i;
    EXPECT_NEAR(gz, sign * dz / norm, 1e-4f) << i;
  }

  // Far outside the truncation radius: distance saturates, point and gradient
  // are unknown.
  const std::vector<float> far_query = {0.0f, 0.0f, -20.0f};
  mrl::QueryResults far_results;
  depth_buffer_->query(far_query.data(), 1, false, far_results, 1.0f);
  EXPECT_NEAR(far_results.optimistic_distance[0], 1.0f, 1e-5f);
  EXPECT_TRUE(std::isnan(far_results.optimistic_closest[0]));
  EXPECT_TRUE(std::isnan(far_results.optimistic_gradient[0]));
}

/**
 * @test isKeyframe reports on the thresholds in BufferParams, against the most
 * recently inserted keyframe. It never inserts anything itself, so a caller can
 * substitute its own criterion.
 */
TEST_F(CudaDepthBufferTest, IsKeyframeFollowsTheConfiguredThresholds) {
  mrl::BufferParams p = params();
  p.keyframe_dist_threshold = 0.5f;
  p.keyframe_angle_threshold = 0.1f;
  p.keyframe_time_threshold = 1.0f;
  mrl::CudaDepthBuffer buffer(intrinsics(), p);

  std::array<float, 7> pose = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  // Nothing inserted yet, so anything is the first keyframe.
  EXPECT_TRUE(buffer.isKeyframe(pose.data(), 0));

  fillDepth(1.0f);
  buffer.setPosedDepthImage(d_depth_image.data(), pose.data(), 0, true);
  EXPECT_FALSE(buffer.isKeyframe(pose.data(), 0));

  // translation
  std::array<float, 7> moved = pose;
  moved[0] = 0.49f;
  EXPECT_FALSE(buffer.isKeyframe(moved.data(), 0));
  moved[0] = 0.51f;
  EXPECT_TRUE(buffer.isKeyframe(moved.data(), 0));

  // rotation: a yaw of `angle` is (0, 0, sin(angle/2), cos(angle/2))
  auto yawPose = [&](float angle) {
    std::array<float, 7> q = pose;
    q[5] = std::sin(0.5f * angle);
    q[6] = std::cos(0.5f * angle);
    return q;
  };
  auto rotated = yawPose(0.05f);
  EXPECT_FALSE(buffer.isKeyframe(rotated.data(), 0));
  rotated = yawPose(0.2f);
  EXPECT_TRUE(buffer.isKeyframe(rotated.data(), 0));

  // time
  EXPECT_FALSE(buffer.isKeyframe(pose.data(), 900000000));
  EXPECT_TRUE(buffer.isKeyframe(pose.data(), 1100000000));

  // a non keyframe insert does not move the reference
  buffer.setPosedDepthImage(d_depth_image.data(), moved.data(), 5, false);
  EXPECT_TRUE(buffer.isKeyframe(moved.data(), 0));
}

TEST_F(CudaDepthBufferTest, ParamsAndIntrinsicsAreReadableBack) {
  EXPECT_EQ(depth_buffer_->params().num_frames, BUFFER_CAPACITY);
  EXPECT_FLOAT_EQ(depth_buffer_->intrinsics().fx, FX);
  EXPECT_FLOAT_EQ(depth_buffer_->intrinsics().cx, CX);
  depth_buffer_->setRobotRadius(0.25f);
  EXPECT_FLOAT_EQ(depth_buffer_->params().robot_radius, 0.25f);
}

// ---------------------------------------------------------------------------
// Analytic scene system tests
//
// The scene is generated, not recorded: cylinders and spheres a camera ray can
// be intersected with in closed form. Both shapes are present on purpose. An
// extruded scene is constant along its axis, so a pruning defect that only
// loses candidates along that axis leaves the answer unchanged and stays
// invisible; spheres curve in every direction and expose it.
//
// The reference is an exhaustive search over every pixel of every frame, which
// is what the optimistic query is defined to reproduce.
// ---------------------------------------------------------------------------

namespace {

mrl::CameraIntrinsics sceneIntrinsics() {
  // Deliberately asymmetric: swapping cx with fx, or the axes with each other,
  // moves every back projected point.
  mrl::CameraIntrinsics k;
  k.cx = 61.0f;
  k.cy = 51.0f;
  k.fx = 90.0f;
  k.fy = 110.0f;
  k.width = 128;
  k.height = 96;
  k.z_near = 0.2f;
  k.z_far = 12.0f;
  return k;
}

std::vector<float> randomQueriesInside(int count,
                                       float extent,
                                       float height,
                                       uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  std::vector<float> queries(3 * count);
  for (int i = 0; i < count; ++i) {
    queries[3 * i + 0] = (unit(rng) - 0.5f) * extent;
    queries[3 * i + 1] = (unit(rng) - 0.5f) * extent;
    queries[3 * i + 2] = unit(rng) * height;
  }
  return queries;
}

}  // namespace

/// A camera orbiting a forest of trunks and bushes, queried inside it. Every
/// optimistic distance must equal the exhaustive nearest back projected pixel.
TEST(CudaDepthBufferSceneTest, ForestMatchesExhaustiveSearch) {
  constexpr int kNumFrames = 6;
  constexpr int kNumQueries = 300;
  constexpr float kTruncation = 2.0f;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  const mrl::AnalyticScene scene = mrl::AnalyticScene::forest(/*seed=*/2024,
                                                              /*cylinders=*/18,
                                                              /*spheres=*/12,
                                                              /*extent=*/8.0f);
  const std::vector<mrl::Pose> poses =
      mrl::orbitPoses(kNumFrames, /*radius=*/7.0f, /*height=*/1.6f);

  std::vector<mrl::PosedDepthImage> frames;
  frames.reserve(kNumFrames);
  for (int f = 0; f < kNumFrames; ++f) {
    frames.push_back(scene.renderPosed(k, poses[f], f));
  }

  mrl::BufferParams params;
  params.num_frames = kNumFrames;
  params.pose_covariance_confidence = 0.0f;
  params.pool_compensation =
      mrl::PoolCompensation::kNone;  // this reference is the point-sample model
  mrl::CudaDepthBuffer buffer(k, params);
  for (const mrl::PosedDepthImage& frame : frames) {
    buffer.setPosedDepthImage(frame.depth.data(),
                              frame.pose.data(),
                              frame.stamp_ns,
                              false);
  }

  const std::vector<float> queries =
      randomQueriesInside(kNumQueries, /*extent=*/6.0f, /*height=*/3.0f, 11);
  mrl::QueryResults results;
  buffer.query(queries.data(),
               kNumQueries,
               /*conservative=*/false,
               results,
               kTruncation);

  int inside_truncation = 0;
  for (int i = 0; i < kNumQueries; ++i) {
    const std::array<float, 3> q{
        {queries[3 * i + 0], queries[3 * i + 1], queries[3 * i + 2]}};
    const float expected = mrl::bruteForceDistance(k, frames, q, kTruncation);
    if (expected < kTruncation) ++inside_truncation;
    EXPECT_NEAR(results.optimistic_distance[i], expected, 1e-4f)
        << "query " << i << " at (" << q[0] << ", " << q[1] << ", " << q[2]
        << ")";
  }
  // Guard against a scene arrangement where everything truncates and the test
  // would pass without checking anything.
  EXPECT_GT(inside_truncation, kNumQueries / 2);
}

/// The same forest with the approximate compensation: point samples searched,
/// half a footprint off the winning surface sample, occlusion candidates
/// untouched - the tiled path must equal the exhaustive statement of that rule.
TEST(CudaDepthBufferSceneTest, ForestMatchesExhaustiveApproximateSearch) {
  constexpr int kNumFrames = 6;
  constexpr int kNumQueries = 300;
  constexpr float kTruncation = 2.0f;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  const mrl::AnalyticScene scene = mrl::AnalyticScene::forest(/*seed=*/2024,
                                                              /*cylinders=*/18,
                                                              /*spheres=*/12,
                                                              /*extent=*/8.0f);
  const std::vector<mrl::Pose> poses =
      mrl::orbitPoses(kNumFrames, /*radius=*/7.0f, /*height=*/1.6f);
  std::vector<mrl::PosedDepthImage> frames;
  for (int f = 0; f < kNumFrames; ++f)
    frames.push_back(scene.renderPosed(k, poses[f], f));

  for (const uint16_t pool : {uint16_t{1}, uint16_t{4}}) {
    mrl::BufferParams params;
    params.num_frames = kNumFrames;
    params.pose_covariance_confidence = 0.0f;
    params.min_pool_size = pool;
    params.pool_compensation = mrl::PoolCompensation::kApproximate;
    mrl::CudaDepthBuffer buffer(k, params);
    for (const mrl::PosedDepthImage& frame : frames) {
      buffer.setPosedDepthImage(frame.depth.data(),
                                frame.pose.data(),
                                frame.stamp_ns,
                                false);
    }
    // The oracle sees what the buffer stores: the min-pooled image with the
    // pooled intrinsics.
    mrl::CameraIntrinsics pooled = k;
    std::vector<mrl::PosedDepthImage> pooled_frames = frames;
    if (pool > 1) {
      // The buffer registers a pooled pixel on the centre of its block.
      pooled.fx /= pool;
      pooled.fy /= pool;
      pooled.cx = (k.cx - 0.5f * (pool - 1)) / pool;
      pooled.cy = (k.cy - 0.5f * (pool - 1)) / pool;
      pooled.width = static_cast<uint16_t>((k.width + pool - 1) / pool);
      pooled.height = static_cast<uint16_t>((k.height + pool - 1) / pool);
      for (mrl::PosedDepthImage& frame : pooled_frames) {
        std::vector<float> out(
            static_cast<size_t>(pooled.width) * pooled.height,
            std::numeric_limits<float>::quiet_NaN());
        for (int v = 0; v < k.height; ++v) {
          for (int u = 0; u < k.width; ++u) {
            const float d = frame.depth[static_cast<size_t>(v) * k.width + u];
            if (std::isnan(d) || d <= 0.0f) continue;
            float& cell =
                out[static_cast<size_t>(v / pool) * pooled.width + u / pool];
            cell = std::isnan(cell) ? d : std::min(cell, d);
          }
        }
        frame.depth = out;
      }
    }
    const std::vector<float> queries =
        randomQueriesInside(kNumQueries, /*extent=*/6.0f, /*height=*/3.0f, 11);
    mrl::QueryResults results;
    buffer.query(queries.data(),
                 kNumQueries,
                 /*conservative=*/true,
                 results,
                 kTruncation);
    int inside_truncation = 0;
    for (int i = 0; i < kNumQueries; ++i) {
      const std::array<float, 3> q{
          {queries[3 * i + 0], queries[3 * i + 1], queries[3 * i + 2]}};
      const mrl::BruteForceDistances expected =
          mrl::bruteForceDistances(pooled,
                                   pooled_frames,
                                   q,
                                   kTruncation,
                                   mrl::PoolCompensation::kApproximate);
      if (expected.optimistic < kTruncation) ++inside_truncation;
      EXPECT_NEAR(results.optimistic_distance[i], expected.optimistic, 1e-4f)
          << "pool " << pool << " query " << i;
      EXPECT_NEAR(results.conservative_distance[i],
                  expected.conservative,
                  1e-4f)
          << "pool " << pool << " query " << i;
    }
    EXPECT_GT(inside_truncation, kNumQueries / 2);
  }
}

/// The same forest with pixels as patches (the default): the tiled search with
/// its widened bounds must still equal the exhaustive patch-model search.
TEST(CudaDepthBufferSceneTest, ForestMatchesExhaustivePatchSearch) {
  constexpr int kNumFrames = 6;
  constexpr int kNumQueries = 300;
  constexpr float kTruncation = 2.0f;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  const mrl::AnalyticScene scene = mrl::AnalyticScene::forest(/*seed=*/2024,
                                                              /*cylinders=*/18,
                                                              /*spheres=*/12,
                                                              /*extent=*/8.0f);
  const std::vector<mrl::Pose> poses =
      mrl::orbitPoses(kNumFrames, /*radius=*/7.0f, /*height=*/1.6f);

  std::vector<mrl::PosedDepthImage> frames;
  frames.reserve(kNumFrames);
  for (int f = 0; f < kNumFrames; ++f) {
    frames.push_back(scene.renderPosed(k, poses[f], f));
  }

  mrl::BufferParams params;
  params.num_frames = kNumFrames;
  params.pose_covariance_confidence = 0.0f;
  params.pool_compensation =
      mrl::PoolCompensation::kExact;  // pixels as patches, the default
  mrl::CudaDepthBuffer buffer(k, params);
  for (const mrl::PosedDepthImage& frame : frames) {
    buffer.setPosedDepthImage(frame.depth.data(),
                              frame.pose.data(),
                              frame.stamp_ns,
                              false);
  }

  const std::vector<float> queries =
      randomQueriesInside(kNumQueries, /*extent=*/6.0f, /*height=*/3.0f, 11);
  mrl::QueryResults results;
  buffer.query(queries.data(),
               kNumQueries,
               /*conservative=*/false,
               results,
               kTruncation);

  int inside_truncation = 0;
  for (int i = 0; i < kNumQueries; ++i) {
    const std::array<float, 3> q{
        {queries[3 * i + 0], queries[3 * i + 1], queries[3 * i + 2]}};
    const float expected = mrl::bruteForceDistance(k,
                                                   frames,
                                                   q,
                                                   kTruncation,
                                                   /*pixel_extent=*/true);
    if (expected < kTruncation) ++inside_truncation;
    EXPECT_NEAR(results.optimistic_distance[i], expected, 1e-4f)
        << "query " << i << " at (" << q[0] << ", " << q[1] << ", " << q[2]
        << ")";
  }
  // Guard against a scene arrangement where everything truncates and the test
  // would pass without checking anything.
  EXPECT_GT(inside_truncation, kNumQueries / 2);
}

/// Spheres alone, so that the nearest surface is reached in a genuinely three
/// dimensional direction rather than within a row or a column.
TEST(CudaDepthBufferSceneTest, ScatteredSpheresMatchExhaustiveSearch) {
  constexpr int kNumFrames = 4;
  constexpr int kNumQueries = 300;
  constexpr float kTruncation = 1.5f;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  mrl::AnalyticScene scene = mrl::AnalyticScene::forest(
      /*seed=*/7,
      /*cylinders=*/0,
      /*spheres=*/24,
      /*extent=*/6.0f,
      /*with_ground=*/false);

  const std::vector<mrl::Pose> poses = mrl::orbitPoses(kNumFrames, 6.0f, 1.0f);
  std::vector<mrl::PosedDepthImage> frames;
  for (int f = 0; f < kNumFrames; ++f) {
    frames.push_back(scene.renderPosed(k, poses[f], f));
  }

  mrl::BufferParams params;
  params.num_frames = kNumFrames;
  params.pose_covariance_confidence = 0.0f;
  params.pool_compensation =
      mrl::PoolCompensation::kNone;  // this reference is the point-sample model
  mrl::CudaDepthBuffer buffer(k, params);
  for (const mrl::PosedDepthImage& frame : frames) {
    buffer.setPosedDepthImage(frame.depth.data(),
                              frame.pose.data(),
                              frame.stamp_ns,
                              false);
  }

  const std::vector<float> queries =
      randomQueriesInside(kNumQueries, 5.0f, 3.0f, 13);
  mrl::QueryResults results;
  buffer.query(queries.data(), kNumQueries, false, results, kTruncation);

  int inside_truncation = 0;
  for (int i = 0; i < kNumQueries; ++i) {
    const std::array<float, 3> q{
        {queries[3 * i + 0], queries[3 * i + 1], queries[3 * i + 2]}};
    const float expected = mrl::bruteForceDistance(k, frames, q, kTruncation);
    if (expected < kTruncation) ++inside_truncation;
    EXPECT_NEAR(results.optimistic_distance[i], expected, 1e-4f)
        << "query " << i;
  }
  EXPECT_GT(inside_truncation, kNumQueries / 3);
}

/// Min pooling changes the resolution the scene is sampled at, but the answer
/// must still be the exhaustive search over the pooled images.
TEST(CudaDepthBufferSceneTest, MinPooledForestMatchesExhaustiveSearch) {
  constexpr int kNumFrames = 3;
  constexpr int kNumQueries = 150;
  constexpr float kTruncation = 2.0f;
  constexpr uint16_t kPool = 2;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  const mrl::AnalyticScene scene = mrl::AnalyticScene::forest(5, 12, 8, 8.0f);
  const std::vector<mrl::Pose> poses = mrl::orbitPoses(kNumFrames, 7.0f, 1.6f);

  mrl::BufferParams params;
  params.num_frames = kNumFrames;
  params.min_pool_size = kPool;
  params.pose_covariance_confidence = 0.0f;
  params.pool_compensation =
      mrl::PoolCompensation::kNone;  // this reference is the point-sample model
  mrl::CudaDepthBuffer buffer(k, params);

  // The oracle sees what the buffer stores: images pooled to half resolution,
  // with the intrinsics scaled to match.
  mrl::CameraIntrinsics pooled = k;
  pooled.width = k.width / kPool;
  pooled.height = k.height / kPool;
  // The buffer registers a pooled pixel on the centre of its block.
  pooled.cx = (k.cx - 0.5f * (kPool - 1)) / kPool;
  pooled.cy = (k.cy - 0.5f * (kPool - 1)) / kPool;
  pooled.fx = k.fx / kPool;
  pooled.fy = k.fy / kPool;

  std::vector<mrl::PosedDepthImage> pooled_frames;
  for (int f = 0; f < kNumFrames; ++f) {
    const mrl::PosedDepthImage full = scene.renderPosed(k, poses[f], f);
    buffer.setPosedDepthImage(full.depth.data(), full.pose.data(), f, false);

    mrl::PosedDepthImage small;
    small.pose = full.pose;
    small.depth.assign(static_cast<size_t>(pooled.width) * pooled.height, 0.0f);
    for (uint16_t v = 0; v < pooled.height; ++v) {
      for (uint16_t u = 0; u < pooled.width; ++u) {
        float best = k.z_far;
        for (uint16_t dv = 0; dv < kPool; ++dv) {
          for (uint16_t du = 0; du < kPool; ++du) {
            best = std::min(
                best,
                full.depth[static_cast<size_t>(v * kPool + dv) * k.width +
                           (u * kPool + du)]);
          }
        }
        small.depth[static_cast<size_t>(v) * pooled.width + u] = best;
      }
    }
    pooled_frames.push_back(std::move(small));
  }

  const std::vector<float> queries =
      randomQueriesInside(kNumQueries, 6.0f, 3.0f, 17);
  mrl::QueryResults results;
  buffer.query(queries.data(), kNumQueries, false, results, kTruncation);

  for (int i = 0; i < kNumQueries; ++i) {
    const std::array<float, 3> q{
        {queries[3 * i + 0], queries[3 * i + 1], queries[3 * i + 2]}};
    EXPECT_NEAR(results.optimistic_distance[i],
                mrl::bruteForceDistance(pooled, pooled_frames, q, kTruncation),
                1e-4f)
        << "query " << i;
  }
}

/// The pooled forest with pixels as patches (the default), against the
/// exhaustive patch-model search over the pooled images - the case the change
/// is for.
TEST(CudaDepthBufferSceneTest, MinPooledForestMatchesExhaustivePatchSearch) {
  constexpr int kNumFrames = 3;
  constexpr int kNumQueries = 150;
  constexpr float kTruncation = 2.0f;
  constexpr uint16_t kPool = 2;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  const mrl::AnalyticScene scene = mrl::AnalyticScene::forest(5, 12, 8, 8.0f);
  const std::vector<mrl::Pose> poses = mrl::orbitPoses(kNumFrames, 7.0f, 1.6f);

  mrl::BufferParams params;
  params.num_frames = kNumFrames;
  params.min_pool_size = kPool;
  params.pose_covariance_confidence = 0.0f;
  params.pool_compensation =
      mrl::PoolCompensation::kExact;  // pixels as patches, the default
  mrl::CudaDepthBuffer buffer(k, params);

  // The oracle sees what the buffer stores: images pooled to half resolution,
  // with the intrinsics scaled to match.
  mrl::CameraIntrinsics pooled = k;
  pooled.width = k.width / kPool;
  pooled.height = k.height / kPool;
  // The buffer registers a pooled pixel on the centre of its block.
  pooled.cx = (k.cx - 0.5f * (kPool - 1)) / kPool;
  pooled.cy = (k.cy - 0.5f * (kPool - 1)) / kPool;
  pooled.fx = k.fx / kPool;
  pooled.fy = k.fy / kPool;

  std::vector<mrl::PosedDepthImage> pooled_frames;
  for (int f = 0; f < kNumFrames; ++f) {
    const mrl::PosedDepthImage full = scene.renderPosed(k, poses[f], f);
    buffer.setPosedDepthImage(full.depth.data(), full.pose.data(), f, false);

    mrl::PosedDepthImage small;
    small.pose = full.pose;
    small.depth.assign(static_cast<size_t>(pooled.width) * pooled.height, 0.0f);
    for (uint16_t v = 0; v < pooled.height; ++v) {
      for (uint16_t u = 0; u < pooled.width; ++u) {
        float best = k.z_far;
        for (uint16_t dv = 0; dv < kPool; ++dv) {
          for (uint16_t du = 0; du < kPool; ++du) {
            best = std::min(
                best,
                full.depth[static_cast<size_t>(v * kPool + dv) * k.width +
                           (u * kPool + du)]);
          }
        }
        small.depth[static_cast<size_t>(v) * pooled.width + u] = best;
      }
    }
    pooled_frames.push_back(std::move(small));
  }

  const std::vector<float> queries =
      randomQueriesInside(kNumQueries, 6.0f, 3.0f, 17);
  mrl::QueryResults results;
  buffer.query(queries.data(), kNumQueries, false, results, kTruncation);

  for (int i = 0; i < kNumQueries; ++i) {
    const std::array<float, 3> q{
        {queries[3 * i + 0], queries[3 * i + 1], queries[3 * i + 2]}};
    EXPECT_NEAR(results.optimistic_distance[i],
                mrl::bruteForceDistance(pooled,
                                        pooled_frames,
                                        q,
                                        kTruncation,
                                        /*pixel_extent=*/true),
                1e-4f)
        << "query " << i;
  }
}

/// Conservative handling has its own oracle: the exhaustive form of the same
/// definition, occlusion segments and near plane included. Nothing about it is
/// covered by the optimistic comparison, which only searches back projected
/// pixels.
TEST(CudaDepthBufferSceneTest, ConservativeMatchesExhaustiveSearch) {
  constexpr int kNumFrames = 4;
  constexpr int kNumQueries = 250;
  constexpr float kTruncation = 3.0f;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  const mrl::AnalyticScene scene = mrl::AnalyticScene::forest(3, 14, 10, 8.0f);
  const std::vector<mrl::Pose> poses = mrl::orbitPoses(kNumFrames, 6.0f, 1.5f);

  std::vector<mrl::PosedDepthImage> frames;
  for (int f = 0; f < kNumFrames; ++f) {
    frames.push_back(scene.renderPosed(k, poses[f], f));
  }

  mrl::BufferParams params;
  params.num_frames = kNumFrames;
  params.pose_covariance_confidence = 0.0f;
  params.pool_compensation =
      mrl::PoolCompensation::kNone;  // this reference is the point-sample model
  mrl::CudaDepthBuffer buffer(k, params);
  for (const mrl::PosedDepthImage& frame : frames) {
    buffer.setPosedDepthImage(frame.depth.data(),
                              frame.pose.data(),
                              frame.stamp_ns,
                              false);
  }

  const std::vector<float> queries =
      randomQueriesInside(kNumQueries, 6.0f, 3.0f, 23);
  mrl::QueryResults results;
  buffer.query(queries.data(), kNumQueries, true, results, kTruncation);

  int signed_negative = 0;
  for (int i = 0; i < kNumQueries; ++i) {
    const std::array<float, 3> q{
        {queries[3 * i + 0], queries[3 * i + 1], queries[3 * i + 2]}};
    const mrl::BruteForceDistances expected =
        mrl::bruteForceDistances(k, frames, q, kTruncation);
    EXPECT_NEAR(results.optimistic_distance[i], expected.optimistic, 1e-4f)
        << "query " << i;
    EXPECT_NEAR(results.conservative_distance[i], expected.conservative, 1e-4f)
        << "query " << i;
    if (expected.conservative < 0.0f) ++signed_negative;
  }
  // Some queries must land in unknown space, or the sign is never exercised.
  EXPECT_GT(signed_negative, 0);
}

/// With nothing in view the only boundaries are the frustum itself, so the
/// conservative distance is the distance to the nearest side plane. This checks
/// the definition against closed form geometry rather than against another
/// implementation of it.
TEST(CudaDepthBufferSceneTest, EmptyViewGivesTheDistanceToTheFrustumWalls) {
  mrl::CameraIntrinsics k;
  k.width = 128;
  k.height = 96;
  k.cx = 0.5f * k.width - 0.5f;
  k.cy = 0.5f * k.height - 0.5f;
  k.fx = 40.0f;  // wide, so the walls are well inside the truncation radius
  k.fy = 40.0f;
  k.z_near = 0.3f;
  k.z_far = 12.0f;

  const mrl::AnalyticScene empty;
  const mrl::Pose pose{{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  const mrl::PosedDepthImage frame = empty.renderPosed(k, pose, 0);

  mrl::BufferParams params;
  params.num_frames = 1;
  params.pose_covariance_confidence = 0.0f;
  mrl::CudaDepthBuffer buffer(k, params);
  buffer.setPosedDepthImage(frame.depth.data(), frame.pose.data(), 0, false);

  // Half angles of the four side planes, as pixel ray slopes.
  const float px_lo = (0.0f - k.cx) / k.fx;
  const float px_hi = (k.width - 1 - k.cx) / k.fx;
  const float py_lo = (0.0f - k.cy) / k.fy;
  const float py_hi = (k.height - 1 - k.cy) / k.fy;

  for (const std::array<float, 3>& q :
       {std::array<float, 3>{{0.0f, 0.0f, 2.0f}},
        std::array<float, 3>{{0.8f, 0.0f, 2.0f}},
        std::array<float, 3>{{0.0f, -0.7f, 2.5f}},
        std::array<float, 3>{{-1.0f, 0.5f, 3.0f}}}) {
    // Distance from the query to each plane through the origin containing a
    // border ray, e.g. x = px_hi * z has normal (1, 0, -px_hi).
    const float walls[4] = {
        std::fabs(q[0] - px_lo * q[2]) / std::sqrt(1.0f + px_lo * px_lo),
        std::fabs(q[0] - px_hi * q[2]) / std::sqrt(1.0f + px_hi * px_hi),
        std::fabs(q[1] - py_lo * q[2]) / std::sqrt(1.0f + py_lo * py_lo),
        std::fabs(q[1] - py_hi * q[2]) / std::sqrt(1.0f + py_hi * py_hi)};
    float expected = walls[0];
    for (float w : walls) expected = std::min(expected, w);

    const float truncation = 4.0f;
    const std::vector<float> query = {q[0], q[1], q[2]};
    mrl::QueryResults results;
    buffer.query(query.data(), 1, true, results, truncation);

    // The walls are sampled at pixel spacing, so agreement is up to the metric
    // footprint of a pixel at that range.
    const float tolerance = 2.0f * q[2] / k.fx;
    EXPECT_NEAR(results.conservative_distance[0], expected, tolerance)
        << "query (" << q[0] << ", " << q[1] << ", " << q[2] << ")";
    // Nothing is in view, so the only surface the optimistic channel can see is
    // the far plane, which here is beyond the truncation radius.
    EXPECT_NEAR(results.optimistic_distance[0],
                std::min(k.z_far - q[2], truncation),
                tolerance);
  }
}

/// The near clipping plane is a boundary of unknown space like any other. Its
/// candidates sit where each pixel ray crosses z = z_near, which off the
/// optical axis is not z_near times the unit ray direction.
TEST(CudaDepthBufferSceneTest, NearPlaneIsBoundedCorrectlyOffAxis) {
  mrl::CameraIntrinsics k;
  k.width = 128;
  k.height = 96;
  k.cx = 0.5f * k.width - 0.5f;
  k.cy = 0.5f * k.height - 0.5f;
  k.fx = 40.0f;
  k.fy = 40.0f;
  k.z_near = 0.3f;
  k.z_far = 12.0f;

  const mrl::AnalyticScene empty;
  const mrl::Pose pose{{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  const mrl::PosedDepthImage frame = empty.renderPosed(k, pose, 0);

  mrl::BufferParams params;
  params.num_frames = 1;
  params.pose_covariance_confidence = 0.0f;
  mrl::CudaDepthBuffer buffer(k, params);
  buffer.setPosedDepthImage(frame.depth.data(), frame.pose.data(), 0, false);

  // Just in front of the near plane and well inside the frustum walls, so the
  // near plane is what the answer is made of.
  const float z = 0.5f;
  for (float x : {0.0f, 0.1f, 0.2f, 0.3f, 0.4f}) {
    const std::vector<float> query = {x, 0.0f, z};
    mrl::QueryResults results;
    buffer.query(query.data(), 1, true, results, 3.0f);
    EXPECT_NEAR(results.conservative_distance[0], z - k.z_near, 2e-3f)
        << "x = " << x;
  }
}

/// An occlusion segment runs along the ray of a pixel, between that pixel's
/// depth and its right or down neighbour's. When the two sit in different tiles
/// the segment leaves the depth slab of the tile that owns the pixel, so a tile
/// bound built from that tile's own min and max alone is not a lower bound on
/// the distance to it.
///
/// Hand built rather than rendered, because the discontinuity has to land on a
/// tile boundary exactly.
TEST(CudaDepthBufferSceneTest,
     DepthStepOnATileBoundaryKeepsItsOcclusionSegment) {
  mrl::CameraIntrinsics k;
  k.width = 256;
  k.height = 128;
  k.cx = 127.5f;
  k.cy = 63.5f;
  k.fx = 200.0f;
  k.fy = 200.0f;
  k.z_near = 0.1f;
  k.z_far = 20.0f;

  // Foreground at 1 m over the left half, background at 6 m over the right,
  // with the step at u = 96, a multiple of the 32 px tile width.
  constexpr int kStepCol = 96;
  std::vector<float> depth(static_cast<size_t>(k.width) * k.height);
  for (int v = 0; v < k.height; ++v) {
    for (int u = 0; u < k.width; ++u) {
      depth[static_cast<size_t>(v) * k.width + u] = u < kStepCol ? 1.0f : 6.0f;
    }
  }

  mrl::BufferParams params;
  params.num_frames = 1;
  params.pose_covariance_confidence = 0.0f;
  mrl::CudaDepthBuffer buffer(k, params);
  const mrl::Pose pose{{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  buffer.setPosedDepthImage(depth.data(), pose.data(), 0, false);

  const mrl::PosedDepthImage frame{depth, pose, 0};
  const float truncation = 2.0f;

  // Points beside the segment on the ray of the last foreground column, at a
  // depth the owning tile's slab does not contain.
  const float ray_x = (kStepCol - 1 - k.cx) / k.fx;
  const float ray_y = (63 - k.cy) / k.fy;
  const float ray_norm = std::sqrt(ray_x * ray_x + ray_y * ray_y + 1.0f);
  // Unit normal to the ray in the x-z plane, pointing into the visible side.
  const float nx = 1.0f / std::sqrt(1.0f + ray_x * ray_x);
  const float nz = -ray_x / std::sqrt(1.0f + ray_x * ray_x);

  for (float t : {2.5f, 3.0f, 4.0f}) {
    for (float offset : {0.2f, 0.3f}) {
      const std::array<float, 3> q{{t * ray_x / ray_norm + offset * nx,
                                    t * ray_y / ray_norm,
                                    t / ray_norm + offset * nz}};
      const mrl::BruteForceDistances expected =
          mrl::bruteForceDistances(k, {frame}, q, truncation);
      const std::vector<float> query = {q[0], q[1], q[2]};
      mrl::QueryResults results;
      buffer.query(query.data(), 1, true, results, truncation);
      EXPECT_NEAR(results.conservative_distance[0],
                  expected.conservative,
                  1e-3f)
          << "t = " << t << ", offset = " << offset;
    }
  }
}

/// Depth images and query points may already live on the device: a GPU depth
/// decoder writes there, and a planner sampling trajectories on the GPU queries
/// from there. Neither should have to make a detour through the host, and both
/// have to give the same answer as the host path.
TEST(CudaDepthBufferSceneTest, DevicePointersMatchHostPointers) {
  constexpr int kNumFrames = 3;
  constexpr int kNumQueries = 200;
  constexpr float kTruncation = 2.5f;

  const mrl::CameraIntrinsics k = sceneIntrinsics();
  const mrl::AnalyticScene scene = mrl::AnalyticScene::forest(5, 12, 8, 8.0f);
  const std::vector<mrl::Pose> poses = mrl::orbitPoses(kNumFrames, 6.0f, 1.5f);
  const std::vector<float> queries =
      randomQueriesInside(kNumQueries, 6.0f, 3.0f, 41);

  mrl::BufferParams params;
  params.num_frames = kNumFrames;
  params.pose_covariance_confidence = 0.0f;

  mrl::QueryResults from_host;
  {
    mrl::CudaDepthBuffer buffer(k, params);
    for (int f = 0; f < kNumFrames; ++f) {
      const mrl::PosedDepthImage frame = scene.renderPosed(k, poses[f], f);
      buffer.setPosedDepthImage(frame.depth.data(),
                                frame.pose.data(),
                                f,
                                false);
    }
    buffer.query(queries.data(), kNumQueries, true, from_host, kTruncation);
  }

  mrl::QueryResults from_device;
  {
    mrl::CudaDepthBuffer buffer(k, params);
    const size_t pixels = static_cast<size_t>(k.width) * k.height;
    float* d_depth = nullptr;
    float* d_queries = nullptr;
    ASSERT_EQ(cudaMalloc(&d_depth, pixels * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_queries, queries.size() * sizeof(float)),
              cudaSuccess);
    for (int f = 0; f < kNumFrames; ++f) {
      const mrl::PosedDepthImage frame = scene.renderPosed(k, poses[f], f);
      ASSERT_EQ(cudaMemcpy(d_depth,
                           frame.depth.data(),
                           pixels * sizeof(float),
                           cudaMemcpyHostToDevice),
                cudaSuccess);
      buffer.setPosedDepthImage(d_depth, frame.pose.data(), f, false);
    }
    ASSERT_EQ(cudaMemcpy(d_queries,
                         queries.data(),
                         queries.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    buffer.query(d_queries, kNumQueries, true, from_device, kTruncation);
    ASSERT_EQ(cudaFree(d_depth), cudaSuccess);
    ASSERT_EQ(cudaFree(d_queries), cudaSuccess);
  }

  for (int i = 0; i < kNumQueries; ++i) {
    EXPECT_FLOAT_EQ(from_device.optimistic_distance[i],
                    from_host.optimistic_distance[i])
        << "query " << i;
    EXPECT_FLOAT_EQ(from_device.conservative_distance[i],
                    from_host.conservative_distance[i])
        << "query " << i;
    for (int c = 0; c < 3; ++c) {
      EXPECT_FLOAT_EQ(from_device.conservative_closest[3 * i + c],
                      from_host.conservative_closest[3 * i + c]);
    }
  }
}

/// A SLAM backend does not always revise both halves at once: it may refine a
/// pose long before it has an uncertainty for it, or attach a covariance to a
/// pose it does not intend to move.
TEST_F(CudaDepthBufferTest, PoseAndCovarianceUpdateIndependently) {
  const int64_t stamp = 4242;
  fillDepth(1.0f);  // a wall one metre in front of the camera
  std::vector<float> pose = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  depth_buffer_->setPosedDepthImage(d_depth_image.data(), pose.data(), stamp);

  std::vector<double> covariance(36, 0.0);
  for (int i = 0; i < 6; ++i) covariance[i * 6 + i] = 0.01 * (i + 1);
  EXPECT_TRUE(
      depth_buffer_->updatePoseAndCovarianceByTimestamp(stamp,
                                                        nullptr,
                                                        covariance.data()));
  std::vector<double> read_back(36, -1.0);
  ASSERT_TRUE(depth_buffer_->getCovarianceByTimestamp(stamp, read_back.data()));
  EXPECT_EQ(read_back, covariance);

  // The pose alone, leaving the covariance where it is.
  std::vector<float> moved = {2.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  EXPECT_TRUE(depth_buffer_->updatePoseAndCovarianceByTimestamp(stamp,
                                                                moved.data(),
                                                                nullptr));
  std::fill(read_back.begin(), read_back.end(), -1.0);
  ASSERT_TRUE(depth_buffer_->getCovarianceByTimestamp(stamp, read_back.data()));
  EXPECT_EQ(read_back, covariance) << "an unrelated pose update dropped it";

  // The frame moved with its pose: the wall is now a metre in front of x = 2.5,
  // so a query two metres out along the optical axis sits a metre behind it.
  const SingleResult after =
      querySingle(*depth_buffer_, 2.5f, 0.0f, 2.0f, false);
  EXPECT_NEAR(after.distance, 1.0f, 1e-3f);

  EXPECT_FALSE(depth_buffer_->updatePoseAndCovarianceByTimestamp(stamp,
                                                                 nullptr,
                                                                 nullptr));
  EXPECT_FALSE(
      depth_buffer_->updatePoseAndCovarianceByTimestamp(stamp + 1,
                                                        moved.data(),
                                                        covariance.data()));
}
