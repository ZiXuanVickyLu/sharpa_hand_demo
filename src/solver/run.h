#pragma once
// Headless driver used by cs_run and the integration tests: load config -> build scene ->
// step -> export bgeo/obj surfaces and per-step statistics.
#include "scene/scene_desc.h"
#include <functional>
#include <optional>
#include <string>

namespace cs {

struct RunOptions {
    std::optional<int> frames;            // override simulation.frames
    std::optional<std::string> output_dir; // override simulation.output_dir
    bool export_frames = true;            // false: no per-frame files at all (viewer --no-export); stats CSV still written
    bool export_surfaces = true;          // write surface bgeo/obj per saved frame
    bool export_tets = false;             // also write the full tet meshes
    bool quiet = false;
    // Called after every step with the frame index; return false to stop early.
    std::function<bool(int)> on_step;
};

struct Scene;
class Stepper;
struct StepStats;

// One run as an object (IS §5.1, §5.x viewer): the constructor does everything cs_run does before
// its loop (resolved config, scene, stepper, restart state, frame-0 export), step() advances one
// step with the exports, the statistics row, the checkpoint and the capped-step guard. cs_run
// loops over it; cs_view calls step() from its UI callback, so both share one code path.
class RunSession {
public:
    RunSession(const SceneDesc& desc, const RunOptions& options);
    ~RunSession();
    RunSession(const RunSession&) = delete;
    RunSession& operator=(const RunSession&) = delete;

    // False when the run is over: all frames done, on_step asked to stop, or the capped-step
    // guard fired (exit_code() == 4). Calling it again after that does nothing.
    bool step();
    bool finished() const;
    int exit_code() const;
    int frame() const;          // index of the next step (absolute, restart included)
    int first_frame() const;    // 0, or simulation.restart.step
    int total_frames() const;   // simulation.frames
    const SceneDesc& desc() const;
    const Scene& scene() const;
    Stepper& stepper();
    const StepStats& last_stats() const;
    std::string output_dir() const;

private:
    struct Impl;
    Impl* impl_;
};

// Returns 0 on success. Writes <output_dir>/resolved_config.json, <output_dir>/stats.csv,
// and <output_dir>/frame_<n>.bgeo (surface) when saving is enabled.
int run_simulation(const SceneDesc& desc, const RunOptions& options);
int run_simulation_file(const std::string& config_path, const RunOptions& options);

}  // namespace cs
