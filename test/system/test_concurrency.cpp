#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <thread>
#include <vector>

namespace {

constexpr uint16_t kRows = 96;
constexpr uint16_t kCols = 128;
constexpr float kZNear = 0.2f;
constexpr float kZFar = 8.0f;

mrl::CameraIntrinsics intrinsics() {
  mrl::CameraIntrinsics k;
  k.cx = 63.5f;
  k.cy = 47.5f;
  k.fx = 90.0f;
  k.fy = 90.0f;
  k.width = kCols;
  k.height = kRows;
  k.z_near = kZNear;
  k.z_far = kZFar;
  return k;
}

std::vector<float> gridQueries(int n) {
  std::vector<float> q;
  q.reserve(3 * n);
  for (int i = 0; i < n; ++i) {
    const float t = static_cast<float>(i) / static_cast<float>(n);
    q.insert(q.end(), {2.0f * t - 1.0f, 0.5f * t - 0.25f, 0.5f + 3.0f * t});
  }
  return q;
}

}  // namespace

/// Many threads querying the same buffer must each get the answer they would
/// have got alone. Concurrent queries share the depth data but nothing else:
/// each holds its own scratch memory and its own stream.
TEST(CudaDepthBufferConcurrencyTest, ConcurrentQueriesAgreeWithSerialOnes) {
  mrl::BufferParams params;
  params.num_frames = 4;
  mrl::CudaDepthBuffer buffer(intrinsics(), params);

  std::vector<float> depth(static_cast<size_t>(kRows) * kCols);
  for (uint16_t v = 0; v < kRows; ++v) {
    for (uint16_t u = 0; u < kCols; ++u) {
      depth[static_cast<size_t>(v) * kCols + u] =
          2.0f + 0.01f * static_cast<float>((u * 7 + v * 3) % 100);
    }
  }
  for (int f = 0; f < 4; ++f) {
    const float pose[7] = {0.1f * f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    buffer.setPosedDepthImage(depth.data(), pose, f, false);
  }

  const std::vector<float> queries = gridQueries(500);
  const uint32_t n = static_cast<uint32_t>(queries.size() / 3);

  mrl::QueryResults reference;
  buffer.query(queries.data(), n, true, reference, 2.0f);

  constexpr int kThreads = 4;
  constexpr int kRepeats = 8;
  std::vector<std::thread> threads;
  std::atomic<int> mismatches{0};
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (int r = 0; r < kRepeats; ++r) {
        mrl::QueryResults mine;
        buffer.query(queries.data(), n, true, mine, 2.0f);
        for (uint32_t i = 0; i < n; ++i) {
          if (mine.optimistic_distance[i] != reference.optimistic_distance[i] ||
              mine.conservative_distance[i] !=
                  reference.conservative_distance[i]) {
            ++mismatches;
          }
        }
      }
    });
  }
  for (std::thread& thread : threads) thread.join();
  EXPECT_EQ(mismatches.load(), 0);
}

/// Inserts exclude queries, so a query never observes a half written frame. The
/// writer keeps replacing frames while readers query; every answer must stay a
/// plausible distance for one of the states the buffer passed through, and none
/// may be garbage.
TEST(CudaDepthBufferConcurrencyTest, QueriesStayValidWhileFramesAreReplaced) {
  mrl::BufferParams params;
  params.num_frames = 3;
  mrl::CudaDepthBuffer buffer(intrinsics(), params);

  const auto makeDepth = [](float value) {
    return std::vector<float>(static_cast<size_t>(kRows) * kCols, value);
  };
  const float pose[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
  std::vector<float> initial = makeDepth(3.0f);
  buffer.setPosedDepthImage(initial.data(), pose, 0, false);

  const std::vector<float> queries = gridQueries(200);
  const uint32_t n = static_cast<uint32_t>(queries.size() / 3);

  std::atomic<bool> stop{false};
  std::atomic<int> bad{0};
  std::thread writer([&] {
    int64_t stamp = 1;
    while (!stop.load()) {
      std::vector<float> d =
          makeDepth(2.0f + 0.5f * static_cast<float>(stamp % 5));
      buffer.setPosedDepthImage(d.data(), pose, stamp++, false);
    }
  });

  std::vector<std::thread> readers;
  for (int t = 0; t < 2; ++t) {
    readers.emplace_back([&] {
      for (int r = 0; r < 40; ++r) {
        mrl::QueryResults results;
        buffer.query(queries.data(), n, true, results, 4.0f);
        for (uint32_t i = 0; i < n; ++i) {
          const float d = results.optimistic_distance[i];
          // Every buffered frame is a plane between z = 2 and z = 4, so no
          // query can be farther than the truncation radius or negative here.
          if (!std::isfinite(d) || d < 0.0f || d > 4.0f + 1e-3f) ++bad;
        }
      }
    });
  }
  for (std::thread& reader : readers) reader.join();
  stop.store(true);
  writer.join();
  EXPECT_EQ(bad.load(), 0);
}
