#pragma once
// The time stepper: math spec §6 (outer loop), §7 (subproblem), §15 (termination).
// M1 runs the elastic-only path (no contact system attached); M2+ attach the contact stage.
#include "scene/scene.h"
#include <Eigen/Core>
#include <map>
#include <memory>
#include <string>

namespace cs {

struct StepStats {
    int frame = 0;
    int outer_iters = 0;        // k at termination
    int inner_newton = 0;       // Newton steps summed over the outer iterations
    int line_search_halvings = 0;
    int pcg_iters_total = 0;
    double pcg_iters_avg = 0.0;
    int pairs_pt = 0, pairs_ee = 0, pairs_ph = 0;   // |C| by type at the end of the step
    // candidates is summed over the outer iterations of a step and reaches 7.6e8 on the animal
    // well and 5.8e8 on the teaser probe, so it needs 64 bits; the other three are bounded by |C|.
    long long candidates = 0;
    int hits = 0, kept = 0, removed = 0;
    int ccd_filtered = 0;
    long long prescribed_hits = 0; // CCD hits between prescribed-only primitives, summed over the step (IS §5.x diagnostic)
    int inversion_limited = 0;  // outer iterations whose fraction the inversion guard reduced below the CCD's (MS §10.4)
    int inversion_pullbacks = 0; // of those, iterations that reset the iterate to the anchor (MS §10.4 pull-back)
    int bc_partial_iters = 0;    // outer iterations solved with the prescribed vertices short of their targets (MS §6 guarded advance)
    int outer_continued = 0;    // outer iterations continued past β ≤ ε by the increment-velocity rule (MS §15.4)       // CCD passes repeated with the tie filter at report time (hit storage cap, IS §12.3 item 15)
    double alpha_min = 1.0, alpha_mean = 1.0;
    double beta_final = 0.0;
    double mu = 0.0;
    int stall_adaptations = 0;
    double gpu_mem_mb = 0.0;    // device memory in use at the end of the step (cudaMemGetInfo, device-wide)
    double energy = 0.0;        // final incremental potential value (free DOFs)
    double max_penetration = 0.0; // debug sweep, 0 when disabled
    double time_ms = 0.0;
    std::map<std::string, double> stage_ms; // per-stage timings for the CSV
    bool hit_iteration_cap = false;
};

class ContactSystem;

class Stepper {
public:
    explicit Stepper(const Scene& scene);
    ~Stepper();
    Stepper(const Stepper&) = delete;
    Stepper& operator=(const Stepper&) = delete;

    // Advance one time step (MS §6). Prescribed targets are evaluated from the scene's
    // motion descriptions for the current frame.
    void step();

    int frame() const;
    const StepStats& last_stats() const;
    // The contact system (null on the elastic-only path); for tests and debug hooks.
    ContactSystem* contact_system();
    const Scene& scene() const;

    // Exported state (the anchor x^{t+1}), all vertices.
    void positions(Eigen::Matrix3Xd& X) const;
    void velocities(Eigen::Matrix3Xd& V) const;

    // Restart (IS §5.x): overwrite the state carried between steps. X and V hold all vertices in
    // the global numbering; the free range is taken from them, prescribed vertices take the
    // scene's own target of step frame-1 (and the velocity of that step). `frame` >= 1 becomes
    // the index of the next step. Returns the largest distance between X and the scene's target
    // over the prescribed vertices (rounding for a float32 file; large if the motion changed
    // before the restart step).
    double set_state(const Eigen::Matrix3Xd& X, const Eigen::Matrix3Xd& V, int frame);

    // Testing hooks: total incremental potential at the current x_hat, and the elastic energy.
    double incremental_potential() const;
    double elastic_energy() const;

    struct Impl;
    Impl& impl() { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

// Static column names of the per-step statistics CSV (io::CsvWriter), in row order.
const std::vector<std::string>& step_stats_columns();
std::vector<double> step_stats_row(const StepStats& s);

}  // namespace cs
