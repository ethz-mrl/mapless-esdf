// The sweep is a branch and bound, so its answer is only correct because every
// bound it prunes with is a genuine lower bound on the distance to anything the
// pruned region could contain. A bound that is slightly too large loses a
// candidate and the query reports more clearance than there is, which is the
// one direction that matters and the one that leaves no trace in the output.
// These tests therefore compare each bound against the set it is supposed to
// cover, exhaustively rather than by sampling where that is possible.
//
// Compiled as CUDA so the host can evaluate the same __host__ __device__
// helpers the kernels run.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "kernels/bounds.cuh"
#include "kernels/common.cuh"

namespace {

struct Camera {
  float cx, cy, fx, fy;
  int width, height;
  float z_near, z_far;
};

/// Squared distance from `q` to the nearest point of the ray of pixel (u, v)
/// between depths dmin and dmax, in closed form: the candidates are
/// d * (px, py, 1) for d in [dmin, dmax].
float rayMinDistSq(const Camera& k,
                   int u,
                   int v,
                   float dmin,
                   float dmax,
                   float qx,
                   float qy,
                   float qz) {
  const double px = (u - static_cast<double>(k.cx)) / k.fx;
  const double py = (v - static_cast<double>(k.cy)) / k.fy;
  const double denom = px * px + py * py + 1.0;
  const double d_star = (qx * px + qy * py + qz) / denom;
  const double d = std::min(static_cast<double>(dmax),
                            std::max(static_cast<double>(dmin), d_star));
  const double ex = d * px - qx;
  const double ey = d * py - qy;
  const double ez = d - qz;
  return static_cast<float>(ex * ex + ey * ey + ez * ez);
}

Camera randomCamera(std::mt19937& rng, int width = 64, int height = 48) {
  std::uniform_real_distribution<float> focal(20.0f, 400.0f);
  std::uniform_real_distribution<float> offset(-8.0f, 8.0f);
  Camera k;
  k.width = width;
  k.height = height;
  k.fx = focal(rng);
  k.fy = focal(rng);
  k.cx = 0.5f * k.width - 0.5f + offset(rng);
  k.cy = 0.5f * k.height - 0.5f + offset(rng);
  k.z_near = 0.1f;
  k.z_far = 20.0f;
  return k;
}

}  // namespace

// ---------------------------------------------------------------------------
// tileSlabBoundSq
// ---------------------------------------------------------------------------

/// The claim under test: for every pixel of the tile and every depth in
/// [dmin, dmax], the bound is no larger than the squared distance from the
/// query to that back projected point. Checked against every pixel of the tile,
/// with the depth minimised in closed form, so nothing is sampled away.
TEST(TileSlabBoundTest, NeverExceedsTheClosestCandidate) {
  // Large enough that a macro tile, 256 x 128 px, fits inside it: the same
  // bound is used at both levels of the hierarchy and the relaxation it makes
  // grows with the rectangle.
  constexpr int kWidth = 320;
  constexpr int kHeight = 192;
  constexpr int kTrials = 600;

  std::mt19937 rng(20260828);
  std::uniform_real_distribution<float> depth(0.15f, 12.0f);
  std::uniform_real_distribution<float> coord(-6.0f, 6.0f);
  std::uniform_real_distribution<float> forward(-1.0f, 14.0f);
  std::uniform_int_distribution<int> tile_x(0, kWidth / mrl::kTileW - 1);
  std::uniform_int_distribution<int> tile_y(0, kHeight / mrl::kTileH - 1);
  std::uniform_int_distribution<int> span_x(1, mrl::kTileW * mrl::kMacroTiles);
  std::uniform_int_distribution<int> span_y(1, mrl::kTileH * mrl::kMacroTiles);

  for (int trial = 0; trial < kTrials; ++trial) {
    const Camera k = randomCamera(rng, kWidth, kHeight);
    const int u0 = tile_x(rng) * mrl::kTileW;
    const int v0 = tile_y(rng) * mrl::kTileH;
    const int u1 = std::min(u0 + span_x(rng), k.width);
    const int v1 = std::min(v0 + span_y(rng), k.height);

    float dmin = depth(rng);
    float dmax = depth(rng);
    if (dmin > dmax) std::swap(dmin, dmax);

    const float qx = coord(rng);
    const float qy = coord(rng);
    const float qz = forward(rng);

    const float bound_sq = mrl::tileSlabBoundSq(qx,
                                                qy,
                                                qz,
                                                static_cast<float>(u0),
                                                static_cast<float>(u1),
                                                static_cast<float>(v0),
                                                static_cast<float>(v1),
                                                dmin,
                                                dmax,
                                                k.cx,
                                                k.cy,
                                                1.0f / k.fx,
                                                1.0f / k.fy);

    float closest_sq = std::numeric_limits<float>::infinity();
    for (int v = v0; v < v1; ++v) {
      for (int u = u0; u < u1; ++u) {
        closest_sq =
            std::min(closest_sq, rayMinDistSq(k, u, v, dmin, dmax, qx, qy, qz));
      }
    }

    // A lower bound, up to the rounding of the bound's own arithmetic.
    ASSERT_LE(bound_sq, closest_sq * (1.0f + 1e-4f) + 1e-6f)
        << "trial " << trial << ": tile [" << u0 << ", " << u1 << ") x [" << v0
        << ", " << v1 << ") depth [" << dmin << ", " << dmax << "] query ("
        << qx << ", " << qy << ", " << qz << ")";
    ASSERT_GE(bound_sq, 0.0f);
  }
}

/// A tile of one pixel at one depth holds exactly one candidate, so there is
/// nothing to relax and the bound has to be that candidate's distance. This
/// pins the bound to the same back projection the scan uses; an off by one in
/// the pixel range or a swapped axis shows up here and not in the inequality
/// above, which any small enough bound satisfies.
TEST(TileSlabBoundTest, IsExactForASinglePixelAtASingleDepth) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> coord(-4.0f, 4.0f);
  std::uniform_real_distribution<float> forward(0.0f, 10.0f);
  std::uniform_real_distribution<float> depth(0.2f, 9.0f);

  for (int trial = 0; trial < 500; ++trial) {
    const Camera k = randomCamera(rng);
    const int u = std::uniform_int_distribution<int>(0, k.width - 1)(rng);
    const int v = std::uniform_int_distribution<int>(0, k.height - 1)(rng);
    const float d = depth(rng);
    const float qx = coord(rng), qy = coord(rng), qz = forward(rng);

    const float bound_sq = mrl::tileSlabBoundSq(qx,
                                                qy,
                                                qz,
                                                static_cast<float>(u),
                                                static_cast<float>(u + 1),
                                                static_cast<float>(v),
                                                static_cast<float>(v + 1),
                                                d,
                                                d,
                                                k.cx,
                                                k.cy,
                                                1.0f / k.fx,
                                                1.0f / k.fy);

    const float ex = (u - k.cx) * d / k.fx - qx;
    const float ey = (v - k.cy) * d / k.fy - qy;
    const float ez = d - qz;
    const float exact = (ex * ex + ey * ey + ez * ez) * mrl::kTileBoundPad;
    EXPECT_NEAR(bound_sq, exact, 1e-4f * std::max(1.0f, exact))
        << "trial " << trial;
  }
}

/// The bound is deflated so that a candidate which exactly ties the running
/// best is never pruned. The sweep prunes on `bound <= best`, so a bound that
/// came back equal to the true distance would drop the tie; kTileBoundPad is
/// what keeps it.
TEST(TileSlabBoundTest, TiesSurviveTheDeflation) {
  const Camera k{31.5f, 23.5f, 100.0f, 100.0f, 64, 48, 0.1f, 20.0f};
  const float d = 2.0f;
  const int u = 40, v = 30;
  // Query sitting on the candidate's own ray, one metre nearer.
  const float px = (u - k.cx) / k.fx, py = (v - k.cy) / k.fy;
  const float qx = px * 1.0f, qy = py * 1.0f, qz = 1.0f;
  const float bound_sq = mrl::tileSlabBoundSq(qx,
                                              qy,
                                              qz,
                                              static_cast<float>(u),
                                              static_cast<float>(u + 1),
                                              static_cast<float>(v),
                                              static_cast<float>(v + 1),
                                              d,
                                              d,
                                              k.cx,
                                              k.cy,
                                              1.0f / k.fx,
                                              1.0f / k.fy);
  const float ex = px * d - qx, ey = py * d - qy, ez = d - qz;
  const float exact = ex * ex + ey * ey + ez * ez;
  EXPECT_LT(bound_sq, exact) << "a tie would be pruned";
}

/// Standing inside the slab there is nothing to prune against.
TEST(TileSlabBoundTest, IsZeroForAQueryOnACandidate) {
  const Camera k{31.5f, 23.5f, 120.0f, 90.0f, 64, 48, 0.1f, 20.0f};
  for (int u : {0, 17, 63}) {
    for (int v : {0, 11, 47}) {
      const float d = 3.0f;
      const float qx = (u - k.cx) * d / k.fx;
      const float qy = (v - k.cy) * d / k.fy;
      EXPECT_FLOAT_EQ(mrl::tileSlabBoundSq(qx,
                                           qy,
                                           d,
                                           0.0f,
                                           static_cast<float>(k.width),
                                           0.0f,
                                           static_cast<float>(k.height),
                                           1.0f,
                                           5.0f,
                                           k.cx,
                                           k.cy,
                                           1.0f / k.fx,
                                           1.0f / k.fy),
                      0.0f);
    }
  }
}

// ---------------------------------------------------------------------------
// computeSpheresProjectedAABB
// ---------------------------------------------------------------------------

/// The AABB is the only thing that keeps the sweep from scanning the whole
/// image, so it must contain every pixel that could hold a candidate within the
/// truncation radius. Checked over the entire pixel grid, with each pixel's
/// nearest reachable point found in closed form along its ray.
TEST(ProjectedAABBTest, ContainsEveryPixelThatCanReachTheSphere) {
  std::mt19937 rng(7788);
  std::uniform_real_distribution<float> coord(-6.0f, 6.0f);
  // Deliberately includes points behind the camera: the sweep is handed
  // whatever the caller queries, and those must be either bounded or empty.
  std::uniform_real_distribution<float> forward(-4.0f, 14.0f);
  std::uniform_real_distribution<float> radius(0.01f, 5.0f);

  for (int trial = 0; trial < 2000; ++trial) {
    const Camera k = randomCamera(rng);
    const float center[3] = {coord(rng), coord(rng), forward(rng)};
    const float r = radius(rng);

    const mrl::AABB2D aabb =
        mrl::computeSpheresProjectedAABB(center,
                                         static_cast<uint16_t>(k.height),
                                         static_cast<uint16_t>(k.width),
                                         k.cx,
                                         k.cy,
                                         k.fx,
                                         k.fy,
                                         r);

    ASSERT_LE(aabb.u_max, k.width);
    ASSERT_LE(aabb.v_max, k.height);

    for (int v = 0; v < k.height; ++v) {
      for (int u = 0; u < k.width; ++u) {
        const float reach_sq = rayMinDistSq(k,
                                            u,
                                            v,
                                            k.z_near,
                                            k.z_far,
                                            center[0],
                                            center[1],
                                            center[2]);
        if (reach_sq > r * r) continue;
        const bool inside = u >= aabb.u_min && u < aabb.u_max &&
                            v >= aabb.v_min && v < aabb.v_max;
        ASSERT_TRUE(inside)
            << "trial " << trial << ": pixel (" << u << ", " << v
            << ") reaches within " << std::sqrt(reach_sq) << " of a sphere of "
            << r << " at (" << center[0] << ", " << center[1] << ", "
            << center[2] << ") but the AABB is [" << aabb.u_min << ", "
            << aabb.u_max << ") x [" << aabb.v_min << ", " << aabb.v_max << ")";
      }
    }
  }
}

/// Bounding is only useful if it is also tight: a small sphere well inside the
/// frustum must not fall back to the whole image.
TEST(ProjectedAABBTest, ShrinksWithTheRadius) {
  const Camera k{31.5f, 23.5f, 200.0f, 200.0f, 64, 48, 0.1f, 20.0f};
  const float center[3] = {0.0f, 0.0f, 5.0f};
  int previous = k.width + 1;
  for (float r : {1.0f, 0.5f, 0.2f, 0.05f}) {
    const mrl::AABB2D aabb =
        mrl::computeSpheresProjectedAABB(center,
                                         static_cast<uint16_t>(k.height),
                                         static_cast<uint16_t>(k.width),
                                         k.cx,
                                         k.cy,
                                         k.fx,
                                         k.fy,
                                         r);
    const int span = aabb.u_max - aabb.u_min;
    EXPECT_LT(span, previous) << "radius " << r;
    previous = span;
  }
  EXPECT_LT(previous, k.width);
}

/// A camera inside the sphere has no bound to give, and a sphere off to the
/// side must come back empty rather than clamped onto the edge column.
TEST(ProjectedAABBTest, DegenerateConfigurations) {
  const Camera k{31.5f, 23.5f, 200.0f, 200.0f, 64, 48, 0.1f, 20.0f};
  const auto aabb = [&](float x, float y, float z, float r) {
    const float c[3] = {x, y, z};
    return mrl::computeSpheresProjectedAABB(c,
                                            static_cast<uint16_t>(k.height),
                                            static_cast<uint16_t>(k.width),
                                            k.cx,
                                            k.cy,
                                            k.fx,
                                            k.fy,
                                            r);
  };

  const mrl::AABB2D inside = aabb(0.0f, 0.0f, 0.5f, 2.0f);
  EXPECT_EQ(inside.u_min, 0);
  EXPECT_EQ(inside.u_max, k.width);
  EXPECT_EQ(inside.v_min, 0);
  EXPECT_EQ(inside.v_max, k.height);

  // Far to the left of the frustum: nothing to scan.
  const mrl::AABB2D beside = aabb(-20.0f, 0.0f, 5.0f, 0.5f);
  EXPECT_EQ(beside.u_max, beside.u_min);
}

// ---------------------------------------------------------------------------
// outwardCoord
// ---------------------------------------------------------------------------

/// The sweep visits macro tiles outward from the projected pixel so the running
/// best tightens early. Correctness of the argmin does not depend on the order,
/// but it does depend on every macro tile being visited exactly once, and the
/// speedup depends on the order being by distance from the centre.
TEST(OutwardCoordTest, IsAPermutationOrderedByDistanceFromTheCentre) {
  for (int lo = -3; lo <= 3; ++lo) {
    for (int span = 0; span <= 20; ++span) {
      const int hi = lo + span;
      for (int center = lo; center <= hi; ++center) {
        std::vector<int> seen;
        int previous_distance = -1;
        for (int k = 0; k <= span; ++k) {
          const int value = mrl::outwardCoord(k, center, lo, hi);
          ASSERT_GE(value, lo)
              << "lo " << lo << " hi " << hi << " c " << center << " k " << k;
          ASSERT_LE(value, hi)
              << "lo " << lo << " hi " << hi << " c " << center << " k " << k;
          const int distance = std::abs(value - center);
          EXPECT_GE(distance, previous_distance)
              << "lo " << lo << " hi " << hi << " c " << center << " k " << k;
          previous_distance = distance;
          seen.push_back(value);
        }
        std::sort(seen.begin(), seen.end());
        ASSERT_EQ(seen.size(), static_cast<size_t>(span + 1));
        for (int i = 0; i <= span; ++i) {
          ASSERT_EQ(seen[i], lo + i)
              << "lo " << lo << " hi " << hi << " centre " << center;
        }
      }
    }
  }
}

TEST(OutwardCoordTest, StartsAtTheCentre) {
  EXPECT_EQ(mrl::outwardCoord(0, 5, 0, 9), 5);
  EXPECT_EQ(mrl::outwardCoord(0, 0, 0, 0), 0);
  // Centre against the low end: the walk can only go one way.
  for (int k = 0; k <= 4; ++k) EXPECT_EQ(mrl::outwardCoord(k, 0, 0, 4), k);
  // ... and against the high end.
  for (int k = 0; k <= 4; ++k) EXPECT_EQ(mrl::outwardCoord(k, 4, 0, 4), 4 - k);
}

// ---------------------------------------------------------------------------
// PackedResult
// ---------------------------------------------------------------------------

/// The argmin is a single 64 bit atomicMin, which only works because the
/// squared distance occupies the high word and non negative IEEE floats compare
/// as unsigned integers do.
TEST(PackedResultTest, OrdersByDistanceAcrossThePayload) {
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> dist_sq(0.0f, 100.0f);
  std::uniform_int_distribution<int> pixel(0, 8190);
  std::uniform_int_distribution<int> flag(0, 1);

  std::vector<std::pair<float, uint64_t>> packed;
  for (int i = 0; i < 5000; ++i) {
    const float d = dist_sq(rng);
    packed.emplace_back(d,
                        mrl::PackedResult(static_cast<uint16_t>(pixel(rng)),
                                          static_cast<uint16_t>(pixel(rng)),
                                          d,
                                          flag(rng),
                                          flag(rng),
                                          flag(rng))
                            .val);
  }
  std::sort(packed.begin(), packed.end(), [](const auto& a, const auto& b) {
    return a.second < b.second;
  });
  for (size_t i = 1; i < packed.size(); ++i) {
    EXPECT_LE(packed[i - 1].first, packed[i].first)
        << "packed order disagrees with distance order at " << i;
  }

  // Subnormals and zero are the ends of the range the sweep actually reaches.
  EXPECT_LT(mrl::PackedResult(0, 0, 0.0f).val,
            mrl::PackedResult(8190, 8190, 1e-30f).val);
}

TEST(PackedResultTest, InfinityLosesToEveryCandidate) {
  const uint64_t infinite = mrl::PackedResult::Infinity().val;
  EXPECT_TRUE(mrl::PackedResult::Infinity().isInfinity());
  for (float d :
       {0.0f, 1e-20f, 1.0f, 1e20f, std::numeric_limits<float>::infinity()}) {
    EXPECT_LT(mrl::PackedResult(1, 2, d).val, infinite) << "dist_sq " << d;
  }
}

TEST(PackedResultTest, RoundTripsItsPayload) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist_sq(0.0f, 1000.0f);
  std::uniform_int_distribution<int> pixel(0, 8191);

  for (int i = 0; i < 2000; ++i) {
    const uint16_t row = static_cast<uint16_t>(pixel(rng));
    const uint16_t col = static_cast<uint16_t>(pixel(rng));
    const float d = dist_sq(rng);
    const bool truncated = (i & 1) != 0;
    const bool occlusion = (i & 2) != 0;
    const bool extremum = (i & 4) != 0;

    const mrl::PackedResult packed(row, col, d, truncated, occlusion, extremum);
    EXPECT_EQ(packed.getRow(), row);
    EXPECT_EQ(packed.getCol(), col);
    EXPECT_FLOAT_EQ(packed.getDistSq(), d);
    EXPECT_EQ(packed.isTruncated(), truncated);
    EXPECT_EQ(packed.isOcclusion(), occlusion);
    EXPECT_EQ(packed.isExtremum(), extremum);
    EXPECT_NEAR(packed.getDist(), std::sqrt(d), 1e-4f * std::sqrt(d) + 1e-6f);
  }
}

/// Row and column share 26 bits, so the largest image the packing can address
/// is 8191 either way. The truncation sentinel deliberately uses that corner as
/// its pixel, and nothing else may alias onto the flags.
TEST(PackedResultTest, PixelIndicesFillThirteenBitsEach) {
  const mrl::PackedResult corner(mrl::PackedResult::maxNumRows(),
                                 mrl::PackedResult::maxNumCols(),
                                 4.0f);
  EXPECT_EQ(corner.getRow(), 8191);
  EXPECT_EQ(corner.getCol(), 8191);
  EXPECT_FALSE(corner.isTruncated());
  EXPECT_FALSE(corner.isOcclusion());
  EXPECT_FALSE(corner.isExtremum());

  const mrl::PackedResult truncated = mrl::PackedResult::Truncated(9.0f);
  EXPECT_TRUE(truncated.isTruncated());
  EXPECT_FLOAT_EQ(truncated.getDistSq(), 9.0f);
  EXPECT_FLOAT_EQ(truncated.getDist(), 3.0f);
}

/// The sign is stamped on after the argmin, by flipping the float sign bit of
/// the squared distance. The magnitude has to survive it, and a negative result
/// has to read back negative.
TEST(PackedResultTest, SignIsStampedWithoutDisturbingTheMagnitude) {
  for (float d : {0.25f, 1.0f, 17.5f}) {
    mrl::PackedResult packed(3, 4, d, false, true);
    packed.setSign(false);
    EXPECT_FLOAT_EQ(packed.getDistSq(), -d);
    EXPECT_FLOAT_EQ(packed.getDist(), -std::sqrt(d));
    EXPECT_EQ(packed.getRow(), 3);
    EXPECT_EQ(packed.getCol(), 4);
    EXPECT_TRUE(packed.isOcclusion());

    packed.setSign(true);
    EXPECT_FLOAT_EQ(packed.getDistSq(), d);
    EXPECT_FLOAT_EQ(packed.getDist(), std::sqrt(d));
  }
}
