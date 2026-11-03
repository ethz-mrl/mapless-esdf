// Is the exact pixel patch worth its arithmetic?
//
// The buffer treats a pooled pixel as the patch [u, u+1) x [v, v+1) at its
// depth and finds the nearest point of the patch to the query, which is a
// lower bound on the distance to any surface inside the pooled block. The
// cheaper alternative is the point sample at (u, v) minus a margin sized by the
// pooled footprint at that pixel's depth. This measures what each returns
// against the finest answer the sensor gives - the exact patch model at pool 1
// on the same frames - and against the scene's true surface, for every pool
// the sweep uses.
//
//   pixel_extent_study [--pools 1,4,9,18,37] [--queries N] [--frames N]
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <mrl/cuda_depth_buffer/scene.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Est {
  const char* name;
  std::vector<float> d;
};

struct Row {
  double mean = 0, mean_abs = 0, p95_abs = 0, max_opt = 0;
  size_t n = 0, optimistic = 0;
};

Row score(const std::vector<float>& est,
          const std::vector<float>& truth,
          float trunc) {
  Row r;
  std::vector<double> abs;
  for (size_t i = 0; i < est.size(); ++i) {
    if (!(truth[i] < trunc) || !(est[i] < trunc))
      continue;  // both inside the radius
    const double e = static_cast<double>(est[i]) - truth[i];
    ++r.n;
    r.mean += e;
    abs.push_back(std::fabs(e));
    if (e > 0.01) ++r.optimistic;
    r.max_opt = std::max(r.max_opt, e);
  }
  if (!r.n) return r;
  r.mean /= r.n;
  std::sort(abs.begin(), abs.end());
  r.mean_abs = 0;
  for (double a : abs) r.mean_abs += a;
  r.mean_abs /= r.n;
  r.p95_abs = abs[static_cast<size_t>(0.95 * (abs.size() - 1))];
  return r;
}

void print(const char* name, const Row& r) {
  std::printf(
      "      %-26s n=%6zu  mean e %+7.1f mm  mean|e| %6.1f mm  p95|e| %6.1f mm "
      " "
      "optimistic (>1 cm) %5.1f %%  max optimistic %+6.1f mm\n",
      name,
      r.n,
      1e3 * r.mean,
      1e3 * r.mean_abs,
      1e3 * r.p95_abs,
      r.n ? 100.0 * r.optimistic / r.n : 0.0,
      1e3 * r.max_opt);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> pools = {1, 4, 9, 18, 37};
  int num_queries = 20000;
  int num_frames = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--pools" && i + 1 < argc) {
      pools.clear();
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ','))
        pools.push_back(std::atoi(tok.c_str()));
    } else if (flag == "--queries" && i + 1 < argc) {
      num_queries = std::atoi(argv[++i]);
    } else if (flag == "--frames" && i + 1 < argc) {
      num_frames = std::atoi(argv[++i]);
    } else {
      std::printf("usage: %s [--pools 1,4,9] [--queries N] [--frames N]\n",
                  argv[0]);
      return 1;
    }
  }
  mrl::CameraIntrinsics intr;
  intr.fx = intr.fy = 525.0f;
  intr.cx = 320.0f;
  intr.cy = 240.0f;
  intr.width = 640;
  intr.height = 480;
  intr.z_near = 0.1f;
  intr.z_far = 10.0f;
  const float trunc = 0.7f;  // the planner's radius

  const mrl::AnalyticScene scene =
      mrl::AnalyticScene::forest(7, 24, 16, 12.0f, true);
  std::vector<mrl::Pose> poses = mrl::orbitPoses(num_frames, 6.0f, 2.0f);
  std::vector<std::vector<float>> frames;
  for (const auto& pose : poses) frames.push_back(scene.render(intr, pose));

  std::mt19937 rng(11);
  std::uniform_real_distribution<float> uxy(-7.0f, 7.0f), uz(0.0f, 4.0f);
  std::vector<float> queries(static_cast<size_t>(num_queries) * 3);
  for (int i = 0; i < num_queries; ++i) {
    queries[3 * i + 0] = uxy(rng);
    queries[3 * i + 1] = uxy(rng);
    queries[3 * i + 2] = uz(rng);
  }

  auto run = [&](int pool, mrl::PoolCompensation mode, mrl::QueryResults& out) {
    mrl::BufferParams params;
    params.num_frames = static_cast<size_t>(num_frames);
    params.use_keyframing = false;
    params.pose_covariance_confidence = 0.0f;
    params.min_pool_size = static_cast<uint16_t>(pool);
    params.pool_compensation = mode;
    mrl::CudaDepthBuffer buffer(intr, params);
    for (size_t f = 0; f < frames.size(); ++f) {
      buffer.setPosedDepthImage(frames[f].data(),
                                poses[f].data(),
                                static_cast<int64_t>(f),
                                true);
    }
    buffer.query(queries.data(),
                 static_cast<uint32_t>(num_queries),
                 true,
                 out,
                 trunc);
  };

  // The finest answer the sensor gives: exact patches at pool 1. And the
  // scene itself, which includes surfaces no frame saw.
  mrl::QueryResults fine;
  run(1, mrl::PoolCompensation::kExact, fine);
  std::vector<float> truth_scene(static_cast<size_t>(num_queries));
  for (int i = 0; i < num_queries; ++i) {
    truth_scene[static_cast<size_t>(i)] = std::min(
        trunc,
        scene.distanceToSurface(
            {{queries[3 * i], queries[3 * i + 1], queries[3 * i + 2]}}));
  }
  {
    const Row r = score(fine.optimistic_distance, truth_scene, trunc);
    std::printf(
        "pool 1 exact patch vs the scene's surface (surfaces no frame saw "
        "included):\n");
    print("patch, pool 1", r);
  }

  for (int pool : pools) {
    std::printf(
        "pool %d (footprint %.1f cm at 5 m), %d queries, %d frames, truncation "
        "%.1f m\n",
        pool,
        100.0 * 5.0 * pool / intr.fx,
        num_queries,
        num_frames,
        trunc);
    mrl::QueryResults patch, point, approx;
    run(pool, mrl::PoolCompensation::kExact, patch);
    run(pool, mrl::PoolCompensation::kNone, point);
    run(pool, mrl::PoolCompensation::kApproximate, approx);
    std::vector<Est> ests = {
        {"exact patch", patch.optimistic_distance},
        {"point sample (none)", point.optimistic_distance},
        {"approximate (half footprint)", approx.optimistic_distance}};
    std::printf(
        "    against the exact patch at pool 1 (e = estimate - fine; "
        "optimistic = farther than the fine sensor says):\n");
    for (const Est& e : ests)
      print(e.name, score(e.d, fine.optimistic_distance, trunc));
    std::printf("    against the scene's true surface:\n");
    for (const Est& e : ests) print(e.name, score(e.d, truth_scene, trunc));
    // The conservative channel, magnitude where both are positive, so the
    // occlusion candidates (untouched by the margin) are in the mix.
    std::printf(
        "    conservative channel, positive side, against the exact patch at "
        "pool 1:\n");
    for (auto& e : std::vector<Est>{
             {"exact patch", patch.conservative_distance},
             {"point sample (none)", point.conservative_distance},
             {"approximate (half footprint)", approx.conservative_distance}}) {
      std::vector<float> a(e.d.size()), b(fine.conservative_distance.size());
      for (size_t i = 0; i < a.size(); ++i) {
        const bool pos = e.d[i] > 0.0f && fine.conservative_distance[i] > 0.0f;
        a[i] = pos ? e.d[i] : trunc;
        b[i] = pos ? fine.conservative_distance[i] : trunc;
      }
      print(e.name, score(a, b, trunc));
    }
  }
  return 0;
}
