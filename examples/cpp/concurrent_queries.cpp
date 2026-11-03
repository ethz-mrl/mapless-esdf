// Shows what the buffer's thread safety buys: several planning threads querying
// while one thread keeps feeding depth images, measured against doing the same
// total work on a single thread.
//
// Queries share the buffer and hold it only against inserts, so they overlap on
// the device; each carries its own scratch memory and its own stream. Inserts
// take the buffer exclusively for their duration, because they overwrite the
// slots a query in flight would read.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <mrl/cuda_depth_buffer/profiling.hpp>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace {

constexpr uint16_t kWidth = 640;
constexpr uint16_t kHeight = 480;
constexpr int kQueriesPerCall = 2000;
constexpr int kCallsTotal = 400;
constexpr float kTruncation = 1.5f;

mrl::CameraIntrinsics intrinsics() {
  mrl::CameraIntrinsics k;
  k.cx = 0.5f * kWidth - 0.5f;
  k.cy = 0.5f * kHeight - 0.5f;
  k.fx = 380.0f;
  k.fy = 380.0f;
  k.width = kWidth;
  k.height = kHeight;
  k.z_near = 0.2f;
  k.z_far = 8.0f;
  return k;
}

/// A depth image of a wall with some structure, cheap to make and varied enough
/// that the tile pruning has to work for it.
std::vector<float> makeDepth(int seed) {
  std::vector<float> depth(static_cast<size_t>(kWidth) * kHeight);
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
  for (uint16_t v = 0; v < kHeight; ++v) {
    for (uint16_t u = 0; u < kWidth; ++u) {
      const float wall =
          3.0f + 0.5f * std::sin(0.02f * u) * std::cos(0.02f * v);
      depth[static_cast<size_t>(v) * kWidth + u] = wall + jitter(rng);
    }
  }
  return depth;
}

std::vector<float> makeQueries(int seed) {
  std::mt19937 rng(static_cast<unsigned>(seed));
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  std::vector<float> queries(3 * kQueriesPerCall);
  for (int i = 0; i < kQueriesPerCall; ++i) {
    queries[3 * i + 0] = (unit(rng) - 0.5f) * 4.0f;
    queries[3 * i + 1] = (unit(rng) - 0.5f) * 3.0f;
    queries[3 * i + 2] = 0.5f + unit(rng) * 3.0f;
  }
  return queries;
}

double millisSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main() {
  std::printf("total work: %d calls x %d queries, 8 frames of %dx%d\n\n",
              kCallsTotal,
              kQueriesPerCall,
              kWidth,
              kHeight);
  mrl::BufferParams params;
  params.num_frames = 8;
  mrl::CudaDepthBuffer buffer(intrinsics(), params);
  buffer.warmup();

  const std::vector<float> depth = makeDepth(1);
  const float pose[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  for (int f = 0; f < 8; ++f) {
    buffer.setPosedDepthImage(depth.data(), pose, f, false);
  }

  const auto runQueries = [&](int calls, int seed) {
    const std::vector<float> queries = makeQueries(seed);
    mrl::QueryResults results;
    for (int c = 0; c < calls; ++c) {
      buffer.query(queries.data(),
                   kQueriesPerCall,
                   /*conservative=*/true,
                   results,
                   kTruncation);
    }
  };

  // ---- one thread doing all the work -------------------------------------
  auto start = std::chrono::steady_clock::now();
  runQueries(kCallsTotal, 7);
  const double serial_ms = millisSince(start);

  // ---- the same work split across query threads --------------------------
  for (int num_threads : {2, 4}) {
    start = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back(runQueries, kCallsTotal / num_threads, 7 + t);
    }
    for (std::thread& thread : threads) thread.join();
    const double parallel_ms = millisSince(start);
    std::printf("  %d query threads      %8.1f ms   %.2fx\n",
                num_threads,
                parallel_ms,
                serial_ms / parallel_ms);
    if (num_threads == 2) {
      std::printf("  1 query thread       %8.1f ms   1.00x\n", serial_ms);
    }
  }

  // ---- query threads competing with a camera feed ------------------------
  // The writer runs at a camera rate rather than flat out. The lock gives the
  // writer priority, so a busy loop here would (correctly) starve the queries;
  // what matters on a robot is that a 30 Hz feed never waits behind planners.
  constexpr int kCameraHz = 30;
  std::atomic<bool> stop{false};
  std::atomic<int> inserted{0};
  double worst_insert_ms = 0.0;
  std::mutex worst_mutex;
  start = std::chrono::steady_clock::now();
  std::thread writer([&] {
    int64_t stamp = 100;
    while (!stop.load(std::memory_order_relaxed)) {
      const auto insert_start = std::chrono::steady_clock::now();
      buffer.setPosedDepthImage(depth.data(), pose, stamp++, false);
      const double insert_ms = millisSince(insert_start);
      {
        const std::lock_guard<std::mutex> lock(worst_mutex);
        worst_insert_ms = std::max(worst_insert_ms, insert_ms);
      }
      inserted.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::milliseconds(1000 / kCameraHz));
    }
  });
  std::vector<std::thread> threads;
  for (int t = 0; t < 2; ++t) {
    threads.emplace_back(runQueries, kCallsTotal / 2, 7 + t);
  }
  for (std::thread& thread : threads) thread.join();
  stop.store(true);
  writer.join();
  const double contended_ms = millisSince(start);
  std::printf(
      "  2 threads + %d Hz feed  %8.1f ms   %.2fx   (%d inserts, worst insert "
      "%.2f ms)\n",
      kCameraHz,
      contended_ms,
      serial_ms / contended_ms,
      inserted.load(),
      worst_insert_ms);

  // A few timed calls so the report below has something to show.
  {
    const std::vector<float> queries = makeQueries(7);
    mrl::QueryResults results;
    for (int c = 0; c < 20; ++c) {
      buffer.query(queries.data(),
                   kQueriesPerCall,
                   true,
                   results,
                   kTruncation,
                   /*verbose=*/true);
    }
  }
  std::printf("\n%s", mrl::profiling::Timing::report().c_str());
  return 0;
}
