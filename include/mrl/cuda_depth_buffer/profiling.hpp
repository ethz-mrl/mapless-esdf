#ifndef MRL_CUDA_DEPTH_BUFFER_PROFILING_HPP
#define MRL_CUDA_DEPTH_BUFFER_PROFILING_HPP

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

/// Lightweight timing registry.
///
/// Timers are addressed by a slash separated tag ("query/occlusion"), which the
/// report renders as a tree. Recording is compiled out entirely unless
/// CUDA_DEPTH_BUFFER_ENABLE_PROFILING is defined, so instrumentation can stay
/// in the hot path.
///
/// Robot deployments care about the tail, not the mean, so the report carries
/// the median and the 99th percentile alongside it. Those two are computed over
/// a bounded window of the most recent samples (see kQuantileWindow); count,
/// total, min and max are exact over the whole run.
namespace mrl {
namespace profiling {

/// Number of recent samples retained per tag for the quantile estimates.
constexpr size_t kQuantileWindow = 8192;

struct Stats {
  size_t count = 0;
  double total_s = 0.0;
  double mean_s = 0.0;
  double median_s = 0.0;  ///< over the last kQuantileWindow samples
  double p99_s = 0.0;     ///< over the last kQuantileWindow samples
  double min_s = 0.0;
  double max_s = 0.0;

  /// Rate a stage of this duration could sustain if it were the only work.
  double meanHz() const { return mean_s > 0.0 ? 1.0 / mean_s : 0.0; }
};

class Timing {
 public:
  static void record(const std::string& tag, double seconds);
  static bool has(const std::string& tag);
  static Stats stats(const std::string& tag);
  /// Tags in insertion order, so a report follows the call order.
  static std::vector<std::string> tags();
  /// Human readable table, indented along the tag hierarchy.
  static std::string report();
  static void reset();
};

/// Times the region between construction and stop() or destruction, and records
/// the duration under `tag`.
///
/// `enabled` is a runtime switch on top of the compile time one: timing GPU
/// work requires synchronizing the device, which the caller only wants to pay
/// for when it is actually profiling.
class Timer {
 public:
  explicit Timer(const char* tag, bool enabled = true)
#ifdef CUDA_DEPTH_BUFFER_ENABLE_PROFILING
      : tag_(tag),
        running_(enabled),
        start_(Clock::now())
#endif
  {
#ifndef CUDA_DEPTH_BUFFER_ENABLE_PROFILING
    (void)tag;
    (void)enabled;
#endif
  }

  ~Timer() { stop(); }

  /// Records the elapsed time. Further calls do nothing.
  void stop() {
#ifdef CUDA_DEPTH_BUFFER_ENABLE_PROFILING
    if (!running_) return;
    running_ = false;
    Timing::record(
        tag_,
        std::chrono::duration<double>(Clock::now() - start_).count());
#endif
  }

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

 private:
#ifdef CUDA_DEPTH_BUFFER_ENABLE_PROFILING
  using Clock = std::chrono::steady_clock;
  const char* tag_;
  bool running_;
  Clock::time_point start_;
#endif
};

}  // namespace profiling
}  // namespace mrl

#endif  // MRL_CUDA_DEPTH_BUFFER_PROFILING_HPP
