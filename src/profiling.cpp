#include <algorithm>
#include <functional>
#include <iomanip>
#include <mrl/cuda_depth_buffer/profiling.hpp>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace mrl {
namespace profiling {
namespace {

struct Accumulator {
  size_t count = 0;
  double total_s = 0.0;
  double min_s = 0.0;
  double max_s = 0.0;
  std::vector<double> window;  ///< ring of the most recent samples
  size_t write_index = 0;
};

/// Registry of all timers. Tag order is kept so that a report reads in the
/// order the stages were first hit, which is the order they run in.
struct Registry {
  std::mutex mutex;
  std::unordered_map<std::string, Accumulator> accumulators;
  std::vector<std::string> order;

  static Registry& instance() {
    static Registry registry;
    return registry;
  }
};

/// Nearest rank quantile of an unsorted copy of the sample window.
double quantile(std::vector<double> samples, double fraction) {
  if (samples.empty()) return 0.0;
  const size_t rank = static_cast<size_t>(
      fraction * static_cast<double>(samples.size() - 1) + 0.5);
  std::nth_element(samples.begin(), samples.begin() + rank, samples.end());
  return samples[rank];
}

std::string formatMilliseconds(double seconds) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << seconds * 1e3;
  return out.str();
}

/// Rate a stage could sustain on its own. Only meaningful while the stage is
/// the bottleneck, so anything faster than a kHz is reported as a bound.
std::string formatRate(double hz) {
  std::ostringstream out;
  if (hz >= 1e5) return "(>100 kHz)";
  out << "(" << std::fixed << std::setprecision(hz < 100.0 ? 1 : 0) << hz
      << " Hz)";
  return out.str();
}

std::string tagParent(const std::string& tag) {
  const size_t slash = tag.rfind('/');
  return slash == std::string::npos ? std::string() : tag.substr(0, slash);
}

size_t tagDepth(const std::string& tag) {
  return static_cast<size_t>(std::count(tag.begin(), tag.end(), '/'));
}

std::string tagLeaf(const std::string& tag) {
  const size_t slash = tag.rfind('/');
  return slash == std::string::npos ? tag : tag.substr(slash + 1);
}

}  // namespace

void Timing::record(const std::string& tag, const double seconds) {
  Registry& registry = Registry::instance();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  const auto inserted = registry.accumulators.emplace(tag, Accumulator{});
  Accumulator& accumulator = inserted.first->second;
  if (inserted.second) {
    registry.order.push_back(tag);
    accumulator.window.reserve(64);
    accumulator.min_s = seconds;
    accumulator.max_s = seconds;
  } else {
    accumulator.min_s = std::min(accumulator.min_s, seconds);
    accumulator.max_s = std::max(accumulator.max_s, seconds);
  }
  ++accumulator.count;
  accumulator.total_s += seconds;
  if (accumulator.window.size() < kQuantileWindow) {
    accumulator.window.push_back(seconds);
  } else {
    accumulator.window[accumulator.write_index] = seconds;
    accumulator.write_index = (accumulator.write_index + 1) % kQuantileWindow;
  }
}

bool Timing::has(const std::string& tag) {
  Registry& registry = Registry::instance();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  return registry.accumulators.count(tag) != 0;
}

Stats Timing::stats(const std::string& tag) {
  Registry& registry = Registry::instance();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  const auto it = registry.accumulators.find(tag);
  if (it == registry.accumulators.end()) return Stats{};
  const Accumulator& accumulator = it->second;

  Stats stats;
  stats.count = accumulator.count;
  stats.total_s = accumulator.total_s;
  stats.mean_s = accumulator.total_s / static_cast<double>(accumulator.count);
  stats.min_s = accumulator.min_s;
  stats.max_s = accumulator.max_s;
  stats.median_s = quantile(accumulator.window, 0.5);
  stats.p99_s = quantile(accumulator.window, 0.99);
  return stats;
}

std::vector<std::string> Timing::tags() {
  Registry& registry = Registry::instance();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  return registry.order;
}

void Timing::reset() {
  Registry& registry = Registry::instance();
  const std::lock_guard<std::mutex> lock(registry.mutex);
  registry.accumulators.clear();
  registry.order.clear();
}

std::string Timing::report() {
  const std::vector<std::string> all_tags = tags();
  if (all_tags.empty()) return "no timings recorded\n";

  // A parent stops after its children, so insertion order lists children
  // first. Walking the hierarchy restores parent-then-children while keeping
  // siblings in the order they ran.
  std::vector<std::string> ordered;
  ordered.reserve(all_tags.size());
  std::function<void(const std::string&)> emitChildren =
      [&](const std::string& parent) {
        for (const std::string& tag : all_tags) {
          if (tagParent(tag) != parent) continue;
          ordered.push_back(tag);
          emitChildren(tag);
        }
      };
  emitChildren(std::string());

  struct Row {
    std::string name;
    Stats stats;
  };
  std::vector<Row> rows;
  rows.reserve(ordered.size());
  size_t name_width = 5;
  for (const std::string& tag : ordered) {
    Row row;
    row.name = std::string(2 * tagDepth(tag), ' ') + tagLeaf(tag);
    row.stats = stats(tag);
    name_width = std::max(name_width, row.name.size());
    rows.push_back(std::move(row));
  }

  std::ostringstream out;
  out << std::left << std::setw(static_cast<int>(name_width)) << "stage"
      << std::right << std::setw(9) << "calls" << std::setw(12) << "total[ms]"
      << std::setw(24) << "mean[ms]" << std::setw(12) << "median[ms]"
      << std::setw(12) << "p99[ms]" << std::setw(12) << "max[ms]" << "\n";
  for (const Row& row : rows) {
    std::ostringstream mean;
    mean << formatMilliseconds(row.stats.mean_s) << " "
         << formatRate(row.stats.meanHz());
    out << std::left << std::setw(static_cast<int>(name_width)) << row.name
        << std::right << std::setw(9) << row.stats.count << std::setw(12)
        << formatMilliseconds(row.stats.total_s) << std::setw(24) << mean.str()
        << std::setw(12) << formatMilliseconds(row.stats.median_s)
        << std::setw(12) << formatMilliseconds(row.stats.p99_s) << std::setw(12)
        << formatMilliseconds(row.stats.max_s) << "\n";
  }
  return out.str();
}

}  // namespace profiling
}  // namespace mrl
