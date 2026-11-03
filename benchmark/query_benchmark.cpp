// Throughput of insertion and of both query channels, on a generated scene.
//
// No dataset: the frames come from mrl::AnalyticScene, so the numbers are
// reproducible anywhere and a regression can be bisected without shipping
// images. The scene is a forest, which is the case the sweep is tuned for: most
// of the image is far, the near geometry is thin, and the branch and bound gets
// to prune hard.
//
// Reports the mean and the 99th percentile of the wall time per call. The tail
// is what a controller running at a fixed rate actually has to fit in, so it is
// the number to watch, not the mean.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mrl/cuda_depth_buffer/depth_buffer.hpp>
#include <mrl/cuda_depth_buffer/profiling.hpp>
#include <mrl/cuda_depth_buffer/scene.hpp>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
  int width = 640;
  int height = 480;
  int num_frames = 20;
  int pool = 1;
  mrl::PoolCompensation pool_compensation = mrl::PoolCompensation::kExact;
  int repeats = 200;
  float truncation = 2.5f;
  uint32_t seed = 1;
  std::vector<int> query_counts = {1, 10, 100, 1000, 10000, 100000};
};

void usage(const char* argv0) {
  std::printf(
      "usage: %s [--width N] [--height N] [--frames N] [--repeats N]\n"
      "          [--truncation M] [--seed N] [--queries N,N,...]\n"
      "          [--pool N] [--pool-compensation none|exact|approximate]\n",
      argv0);
}

Args parseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        usage(argv[0]);
        std::exit(1);
      }
      return argv[++i];
    };
    if (flag == "--width")
      args.width = std::atoi(value());
    else if (flag == "--height")
      args.height = std::atoi(value());
    else if (flag == "--frames")
      args.num_frames = std::atoi(value());
    else if (flag == "--pool")
      args.pool = std::atoi(value());
    else if (flag == "--pool-compensation")
      args.pool_compensation = mrl::poolCompensationFromString(value());
    else if (flag == "--repeats")
      args.repeats = std::atoi(value());
    else if (flag == "--truncation")
      args.truncation = std::atof(value());
    else if (flag == "--seed")
      args.seed = std::strtoul(value(), nullptr, 10);
    else if (flag == "--queries") {
      args.query_counts.clear();
      std::string list = value();
      size_t start = 0;
      while (start <= list.size()) {
        const size_t comma = list.find(',', start);
        const std::string item = list.substr(start, comma - start);
        if (!item.empty()) args.query_counts.push_back(std::atoi(item.c_str()));
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
    } else {
      usage(argv[0]);
      std::exit(flag == "--help" ? 0 : 1);
    }
  }
  return args;
}

/// Mean and 99th percentile of a sample of durations, in milliseconds.
struct Summary {
  double mean_ms = 0.0;
  double p99_ms = 0.0;
};

Summary summarize(std::vector<double> samples_s) {
  if (samples_s.empty()) return {};
  std::sort(samples_s.begin(), samples_s.end());
  double total = 0.0;
  for (double s : samples_s) total += s;
  const size_t p99 = std::min(samples_s.size() - 1,
                              static_cast<size_t>(0.99 * samples_s.size()));
  return {1e3 * total / samples_s.size(), 1e3 * samples_s[p99]};
}

using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// Query points scattered through the volume the camera orbits.
std::vector<float> randomQueries(int count,
                                 float extent,
                                 float height,
                                 uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(0.0f, 1.0f);
  std::vector<float> queries(3 * static_cast<size_t>(count));
  for (int i = 0; i < count; ++i) {
    queries[3 * i + 0] = (unit(rng) - 0.5f) * extent;
    queries[3 * i + 1] = (unit(rng) - 0.5f) * extent;
    queries[3 * i + 2] = unit(rng) * height;
  }
  return queries;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parseArgs(argc, argv);

  mrl::CameraIntrinsics intrinsics;
  intrinsics.width = static_cast<uint16_t>(args.width);
  intrinsics.height = static_cast<uint16_t>(args.height);
  intrinsics.cx = 0.5f * args.width - 0.5f;
  intrinsics.cy = 0.5f * args.height - 0.5f;
  intrinsics.fx = 0.7f * args.width;
  intrinsics.fy = 0.7f * args.width;
  intrinsics.z_near = 0.2f;
  intrinsics.z_far = 20.0f;

  constexpr float kSceneExtent = 16.0f;
  const mrl::AnalyticScene scene =
      mrl::AnalyticScene::forest(args.seed, 40, 24, kSceneExtent);
  const std::vector<mrl::Pose> poses =
      mrl::orbitPoses(args.num_frames, 7.0f, 1.5f);

  std::printf("cuda-depth-buffer benchmark\n");
  std::printf("  image      %d x %d, fx %.0f\n",
              args.width,
              args.height,
              intrinsics.fx);
  std::printf("  buffer     %d frames\n", args.num_frames);
  std::printf("  truncation %.2f m\n", args.truncation);
  std::printf("  repeats    %d\n\n", args.repeats);

  std::vector<mrl::PosedDepthImage> frames;
  frames.reserve(args.num_frames);
  for (int f = 0; f < args.num_frames; ++f) {
    frames.push_back(scene.renderPosed(intrinsics, poses[f], f));
  }

  mrl::BufferParams params;
  params.num_frames = static_cast<size_t>(args.num_frames);
  params.min_pool_size = static_cast<uint16_t>(std::max(1, args.pool));
  params.pool_compensation = args.pool_compensation;
  std::printf("  pool       %d, pool compensation %s\n",
              args.pool,
              mrl::toString(args.pool_compensation));
  mrl::CudaDepthBuffer buffer(intrinsics, params);
  buffer.warmup();

  // ---------------------------------------------------------------- insertion
  std::vector<double> insert_samples;
  insert_samples.reserve(args.repeats);
  for (int r = 0; r < args.repeats; ++r) {
    const mrl::PosedDepthImage& frame = frames[r % frames.size()];
    const Clock::time_point start = Clock::now();
    buffer.setPosedDepthImage(frame.depth.data(),
                              frame.pose.data(),
                              static_cast<int64_t>(r) * 33000000,
                              false);
    insert_samples.push_back(elapsed(start));
  }
  const Summary insertion = summarize(insert_samples);
  std::printf("insertion\n");
  std::printf("  %-12s %8s %8s %10s\n", "", "mean ms", "p99 ms", "rate Hz");
  std::printf("  %-12s %8.3f %8.3f %10.0f\n\n",
              "frame",
              insertion.mean_ms,
              insertion.p99_ms,
              1e3 / insertion.mean_ms);

  // ------------------------------------------------------------------ queries
  std::printf("queries\n");
  std::printf("  %8s %-13s %9s %9s %12s %12s\n",
              "count",
              "channel",
              "mean ms",
              "p99 ms",
              "us / query",
              "queries / s");
  for (int count : args.query_counts) {
    if (count <= 0) continue;
    const std::vector<float> queries =
        randomQueries(count, kSceneExtent * 0.5f, 3.0f, 7);
    for (bool conservative : {false, true}) {
      mrl::QueryResults results;
      // One untimed call so the workspace for this size is already allocated.
      buffer.query(queries.data(),
                   static_cast<uint32_t>(count),
                   conservative,
                   results,
                   args.truncation);

      std::vector<double> samples;
      samples.reserve(args.repeats);
      for (int r = 0; r < args.repeats; ++r) {
        const Clock::time_point start = Clock::now();
        buffer.query(queries.data(),
                     static_cast<uint32_t>(count),
                     conservative,
                     results,
                     args.truncation);
        samples.push_back(elapsed(start));
      }
      const Summary s = summarize(samples);
      std::printf("  %8d %-13s %9.3f %9.3f %12.3f %12.0f\n",
                  count,
                  conservative ? "conservative" : "optimistic",
                  s.mean_ms,
                  s.p99_ms,
                  1e3 * s.mean_ms / count,
                  count / (1e-3 * s.mean_ms));
    }
  }

  // A last pass with the timers on, for the stage breakdown. Timing individual
  // stages means synchronizing the device between them, so these add up to more
  // than the wall times above; the split is what they are for, not the total.
  const int profile_count =
      args.query_counts.empty() ? 1000 : args.query_counts.back();
  const std::vector<float> profile_queries =
      randomQueries(profile_count, kSceneExtent * 0.5f, 3.0f, 7);
  mrl::profiling::Timing::reset();
  mrl::QueryResults profile_results;
  for (int r = 0; r < std::max(1, args.repeats / 10); ++r) {
    buffer.query(profile_queries.data(),
                 static_cast<uint32_t>(profile_count),
                 true,
                 profile_results,
                 args.truncation,
                 /*verbose=*/true);
  }
  const std::string report = mrl::profiling::Timing::report();
  if (report.empty()) {
    std::printf(
        "\nstage breakdown unavailable: built without"
        " CUDA_DEPTH_BUFFER_ENABLE_PROFILING\n");
  } else {
    std::printf(
        "\nstage breakdown, %d conservative queries, device"
        " synchronized per stage\n%s\n",
        profile_count,
        report.c_str());
  }
  return 0;
}
