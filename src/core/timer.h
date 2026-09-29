// timer.h - scoped wall-clock timers accumulated into a process-wide registry.
//
//   {
//       ScopedTimer t("pcg", /*sync_device=*/true);   // synchronises the device on entry and exit
//       ...
//   }                                                 // adds the elapsed ms to TimerRegistry["pcg"]
//
//   TimerRegistry::report()     name -> {total_ms, count}
//   TimerRegistry::to_string()  one line per timer, sorted by total time
//   TimerRegistry::reset()      clears everything
#pragma once
#ifndef CS_TIMER_H
#define CS_TIMER_H

#include <chrono>
#include <cstdint>
#include <map>
#include <string>

namespace cs {

class TimerRegistry {
public:
    struct Entry {
        double total_ms = 0.0;
        std::uint64_t count = 0;
        double mean_ms() const { return count ? total_ms / static_cast<double>(count) : 0.0; }
    };

    /// Adds one measurement to the named timer (thread-safe).
    static void add(const std::string& name, double ms);
    /// Snapshot of every timer.
    static std::map<std::string, Entry> report();
    /// Total milliseconds accumulated under name (0 if unknown).
    static double total_ms(const std::string& name);
    /// Number of measurements accumulated under name (0 if unknown).
    static std::uint64_t count(const std::string& name);
    /// Clears every timer.
    static void reset();
    /// Human-readable table, sorted by descending total time.
    static std::string to_string();
    /// When set, every ScopedTimer synchronises the device around its scope even if it was
    /// constructed with sync_device = false. Stage timings on the GPU are meaningless without
    /// this (a scope that only launches kernels measures ~0 and its cost is charged to the next
    /// scope that happens to read something back), so the driver turns it on whenever
    /// logging.timing is set. Off by default: the syncs serialise the pipeline.
    static void set_sync_device(bool on);
    static bool sync_device();
};

class ScopedTimer {
public:
    using clock = std::chrono::steady_clock;

    /// Starts timing. With sync_device the device is synchronised before the start and before
    /// the stop so that queued asynchronous work is attributed to the right scope.
    explicit ScopedTimer(const char* name, bool sync_device = false);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    /// Stops early and records the measurement; the destructor then does nothing.
    double stop();
    /// Milliseconds elapsed so far (or the recorded value after stop()).
    double elapsed_ms() const;

private:
    const char* name_;
    bool sync_device_;
    bool stopped_ = false;
    double recorded_ms_ = 0.0;
    clock::time_point start_;
};

}  // namespace cs

#endif  // CS_TIMER_H
