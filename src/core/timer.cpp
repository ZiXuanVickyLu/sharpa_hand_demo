#include "timer.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

// timer.cpp is part of cs_host, which is compiled without the CUDA toolkit on its include path.
// The single runtime entry point needed here is taken from the header when it is visible and
// declared by hand otherwise (cudaError_t is an int-sized enum, CUDARTAPI is empty on Linux);
// the symbol resolves against the shared cudart that every consumer of cs_device links.
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
namespace {
int device_synchronize() { return static_cast<int>(cudaDeviceSynchronize()); }
}  // namespace
#else
extern "C" int cudaDeviceSynchronize(void);
namespace {
int device_synchronize() { return cudaDeviceSynchronize(); }
}  // namespace
#endif

namespace cs {

namespace {

struct Registry {
    std::mutex mutex;
    std::map<std::string, TimerRegistry::Entry> entries;
};

// Read on every ScopedTimer construction; written once at start-up.
std::atomic<bool> g_sync_device{false};

Registry& registry()
{
    static Registry r;
    return r;
}

}  // namespace

void TimerRegistry::add(const std::string& name, double ms)
{
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    Entry& e = r.entries[name];
    e.total_ms += ms;
    e.count += 1;
}

std::map<std::string, TimerRegistry::Entry> TimerRegistry::report()
{
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.entries;
}

double TimerRegistry::total_ms(const std::string& name)
{
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = r.entries.find(name);
    return it == r.entries.end() ? 0.0 : it->second.total_ms;
}

std::uint64_t TimerRegistry::count(const std::string& name)
{
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = r.entries.find(name);
    return it == r.entries.end() ? 0 : it->second.count;
}

void TimerRegistry::set_sync_device(bool on) { g_sync_device.store(on, std::memory_order_relaxed); }

bool TimerRegistry::sync_device() { return g_sync_device.load(std::memory_order_relaxed); }

void TimerRegistry::reset()
{
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.entries.clear();
}

std::string TimerRegistry::to_string()
{
    std::vector<std::pair<std::string, Entry>> rows;
    {
        Registry& r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        rows.assign(r.entries.begin(), r.entries.end());
    }
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second.total_ms > b.second.total_ms; });
    std::size_t width = 8;
    for (const auto& row : rows) width = std::max(width, row.first.size());
    std::string out;
    char line[256];
    std::snprintf(line, sizeof(line), "%-*s %12s %10s %12s\n", static_cast<int>(width), "timer", "total_ms", "count",
                  "mean_ms");
    out += line;
    for (const auto& row : rows) {
        std::snprintf(line, sizeof(line), "%-*s %12.3f %10llu %12.4f\n", static_cast<int>(width), row.first.c_str(),
                      row.second.total_ms, static_cast<unsigned long long>(row.second.count), row.second.mean_ms());
        out += line;
    }
    return out;
}

ScopedTimer::ScopedTimer(const char* name, bool sync_device)
    : name_(name), sync_device_(sync_device || TimerRegistry::sync_device())
{
    if (sync_device_) (void)device_synchronize();
    start_ = clock::now();
}

ScopedTimer::~ScopedTimer()
{
    if (!stopped_) {
        try {
            stop();
        } catch (...) {
        }
    }
}

double ScopedTimer::stop()
{
    if (stopped_) return recorded_ms_;
    if (sync_device_) {
        const int rc = device_synchronize();
        if (rc != 0) {
            stopped_ = true;
            throw std::runtime_error(std::string("ScopedTimer(") + name_ + "): cudaDeviceSynchronize failed with code " +
                                     std::to_string(rc));
        }
    }
    recorded_ms_ = std::chrono::duration<double, std::milli>(clock::now() - start_).count();
    stopped_ = true;
    TimerRegistry::add(name_, recorded_ms_);
    return recorded_ms_;
}

double ScopedTimer::elapsed_ms() const
{
    if (stopped_) return recorded_ms_;
    return std::chrono::duration<double, std::milli>(clock::now() - start_).count();
}

}  // namespace cs
