// Headless driver (doc/al-ipc-implementation-spec.md §5.1, §5.4).
#include "../../ext/cnpy/cnpy.h"
#include "solver/run.h"
#include <cstdlib>
#include <string>
#include "solver/stepper.h"
#include "scene/scene.h"
#include "io/mesh_io.h"
#include "core/log.h"
#include "core/timer.h"
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <filesystem>
#include <memory>
#include <fstream>
#include <stdexcept>

namespace cs {

namespace fs = std::filesystem;

namespace {

// Surface of the whole scene (bodies' boundary triangles + colliders) as one triangle soup
// over the global vertex numbering.
Eigen::Matrix3Xi scene_surface_faces(const Scene& scene) {
    Eigen::Matrix3Xi F(3, int(scene.surf_tris.size()));
    for (size_t i = 0; i < scene.surf_tris.size(); ++i) {
        F(0, int(i)) = scene.surf_tris[i][0];
        F(1, int(i)) = scene.surf_tris[i][1];
        F(2, int(i)) = scene.surf_tris[i][2];
    }
    return F;
}

Eigen::Matrix4Xi scene_tets(const Scene& scene) {
    Eigen::Matrix4Xi T(4, int(scene.tets.size()));
    for (size_t i = 0; i < scene.tets.size(); ++i)
        for (int k = 0; k < 4; ++k) T(k, int(i)) = scene.tets[i][k];
    return T;
}

}  // namespace

struct RunSession::Impl {
    SceneDesc desc;
    RunOptions options;
    fs::path out_dir;
    Scene scene;
    std::unique_ptr<Stepper> stepper;
    CsvWriter csv;
    Eigen::Matrix3Xi F;
    Eigen::Matrix4Xi T;
    Eigen::Matrix3Xd X;
    int first = 0, frame = 0, capped_run = 0, exit_code = 0;
    bool finished = false;

    void export_frame(int f) {
        if (!options.export_frames) return;
        if (!desc.simulation.save_bgeo && !desc.simulation.save_npy) return;
        if (desc.simulation.save_every <= 0 || f % desc.simulation.save_every != 0) return;
        stepper->positions(X);
        if (desc.simulation.save_npy) {
            const Eigen::Matrix<float, 3, Eigen::Dynamic> Xf = X.cast<float>();
            cnpy::npy_save((out_dir / ("x_" + std::to_string(f) + ".npy")).string(), Xf.data(), {size_t(Xf.cols()), size_t(3)});
        }
        if (!desc.simulation.save_bgeo) return;
        if (options.export_surfaces) write_bgeo_tris((out_dir / ("surface_" + std::to_string(f) + ".bgeo")).string(), X, F);
        if (options.export_tets && T.cols() > 0) write_bgeo_tets((out_dir / ("tets_" + std::to_string(f) + ".bgeo")).string(), X, T);
    }
    void checkpoint(int f) {
        if (!options.export_frames) return;
        if (desc.simulation.checkpoint_every <= 0 || f % desc.simulation.checkpoint_every != 0) return;
        Eigen::Matrix3Xd Xc, Vc;
        stepper->positions(Xc);
        stepper->velocities(Vc);
        cnpy::npy_save((out_dir / ("ckpt_x_" + std::to_string(f) + ".npy")).string(), Xc.data(), {size_t(Xc.cols()), size_t(3)});
        cnpy::npy_save((out_dir / ("ckpt_v_" + std::to_string(f) + ".npy")).string(), Vc.data(), {size_t(Vc.cols()), size_t(3)});
    }
};

RunSession::RunSession(const SceneDesc& desc_in, const RunOptions& options) : impl_(new Impl) {
    Impl& m = *impl_;
    try {
        m.desc = desc_in;
        m.options = options;
        SceneDesc& desc = m.desc;
        if (options.frames) desc.simulation.frames = *options.frames;
        if (options.output_dir) desc.simulation.output_dir = *options.output_dir;
        if (!options.quiet) set_log_level(desc.logging.level);
        // Stage timings only mean anything if each scope waits for the work it launched; see
        // TimerRegistry::set_sync_device. The syncs cost throughput, so they follow logging.timing.
        TimerRegistry::set_sync_device(desc.logging.timing);

        m.out_dir = fs::path(desc.simulation.output_dir).is_absolute() ? fs::path(desc.simulation.output_dir)
                                                                        : fs::path(CS_OUTPUT_PATH) / desc.simulation.output_dir;
        fs::create_directories(m.out_dir);
        {
            std::ofstream f(m.out_dir / "resolved_config.json");
            f << scene_desc_to_json(desc).dump(2) << "\n";
        }

        m.scene = build_scene(desc);
        log().info("{}", scene_summary(m.scene));
        m.stepper.reset(new Stepper(m.scene));
        Stepper& stepper = *m.stepper;
        const Scene& scene = m.scene;

        m.csv.open((m.out_dir / desc.logging.stats_csv).string(), step_stats_columns());
        m.F = scene_surface_faces(scene);
        m.T = scene_tets(scene);
        if (desc.simulation.save_npy && options.export_frames) {
            const Eigen::Matrix<int, 3, Eigen::Dynamic> Fi = m.F;   // column-major 3 x m == row-major [m, 3]
            cnpy::npy_save((m.out_dir / "surface_faces.npy").string(), Fi.data(), {size_t(Fi.cols()), size_t(3)});
        }

        // IS §5.x restart: continue from a saved state at an absolute step index.
        if (desc.simulation.restart_step >= 0) {
            auto resolve = [&](const std::string& p) {
                return fs::path(p).is_absolute() ? fs::path(p) : fs::path(CS_OUTPUT_PATH) / p;   // like output_dir
            };
            auto load = [&](const std::string& key, const std::string& p, Eigen::Matrix3Xd& M) {
                const fs::path file = resolve(p);
                if (!fs::exists(file)) throw std::runtime_error("simulation.restart." + key + ": no such file: " + file.string());
                const cnpy::NpyArray a = cnpy::npy_load(file.string());
                if (a.shape.size() != 2 || a.shape[1] != 3 || int(a.shape[0]) != scene.n_vertices || a.fortran_order)
                    throw std::runtime_error("simulation.restart." + key + ": " + file.string() + " must be a C-ordered [" +
                                             std::to_string(scene.n_vertices) + ", 3] array (all vertices, global numbering)");
                M.resize(3, scene.n_vertices);
                if (a.word_size == sizeof(float)) {
                    const float* d = a.data<float>();
                    for (int i = 0; i < scene.n_vertices; ++i) M.col(i) = Eigen::Vector3d(d[3 * i], d[3 * i + 1], d[3 * i + 2]);
                } else if (a.word_size == sizeof(double)) {
                    const double* d = a.data<double>();
                    for (int i = 0; i < scene.n_vertices; ++i) M.col(i) = Eigen::Vector3d(d[3 * i], d[3 * i + 1], d[3 * i + 2]);
                } else {
                    throw std::runtime_error("simulation.restart." + key + ": " + file.string() + " must be float32 or float64");
                }
            };
            Eigen::Matrix3Xd X0, V0;
            load("positions", desc.simulation.restart_positions, X0);
            if (desc.simulation.restart_velocities.empty()) V0 = Eigen::Matrix3Xd::Zero(3, scene.n_vertices);
            else load("velocities", desc.simulation.restart_velocities, V0);
            m.first = desc.simulation.restart_step;
            const double worst = stepper.set_state(X0, V0, m.first);
            log().info("restart: step {} from {}{} (prescribed vertices differ from the scene's targets by at most {:.3e} m)",
                       m.first, desc.simulation.restart_positions,
                       desc.simulation.restart_velocities.empty() ? ", at rest" : " with velocities", worst);
            if (worst > 1e-6)
                log().warn("restart: prescribed vertices in the file are up to {:.3e} m from the scene's targets at step {} "
                           "(the motion changed before the restart step?); the scene's targets are used",
                           worst, m.first - 1);
        }
        m.frame = m.first;
        m.export_frame(m.first);
        if (m.frame >= desc.simulation.frames) m.finished = true;
    } catch (...) {
        delete impl_;
        impl_ = nullptr;
        throw;
    }
}

RunSession::~RunSession() { delete impl_; }

bool RunSession::step() {
    Impl& m = *impl_;
    if (m.finished) return false;
    const SceneDesc& desc = m.desc;
    const int frame = m.frame;
    try {
        m.stepper->step();
    } catch (const std::exception& e) {
        // Log before anything is destroyed: after a sticky CUDA error the unwinding itself
        // aborts (Thrust device vectors throw from their destructors when cudaFree fails),
        // which used to lose the message that names the failing check.
        log().error("fatal in frame {}: {}", frame + 1, e.what());
        spdlog::default_logger()->flush();
        if (std::string(e.what()).find("CUDA") != std::string::npos ||
            std::string(e.what()).find("cuda") != std::string::npos)
            std::_Exit(3);   // the device state is unusable; skip the destructors
        throw;
    }
    const StepStats& s = m.stepper->last_stats();
    m.csv.row(step_stats_row(s));
    m.csv.flush();
    log().info("frame {:4d}  outer {:3d}  newton {:3d}  pcg/avg {:6.1f}  |C| {}  alpha_mean {:.3f}  beta {:.2e}  "
               "{:8.1f} ms",
               frame + 1, s.outer_iters, s.inner_newton, s.pcg_iters_avg, s.pairs_pt + s.pairs_ee + s.pairs_ph,
               s.alpha_mean, s.beta_final, s.time_ms);
    m.frame = frame + 1;
    m.export_frame(frame + 1);
    m.checkpoint(frame + 1);
    // IS §5.x: a sequence of capped steps is a prescribed squeeze with no feasible step; stop paying for it.
    m.capped_run = s.hit_iteration_cap ? m.capped_run + 1 : 0;
    if (desc.simulation.abort_after_capped_steps > 0 && m.capped_run >= desc.simulation.abort_after_capped_steps) {
        log().error("step {}: {} consecutive steps ended at the outer iteration cap ({}); stopping "
                    "(simulation.abort_after_capped_steps)", frame + 1, m.capped_run, desc.contact.max_outer_iters);
        spdlog::default_logger()->flush();
        m.exit_code = 4;
        m.finished = true;
        return false;
    }
    if (m.options.on_step && !m.options.on_step(frame + 1)) m.finished = true;
    if (m.frame >= desc.simulation.frames) m.finished = true;
    return !m.finished;
}

bool RunSession::finished() const { return impl_->finished; }
int RunSession::exit_code() const { return impl_->exit_code; }
int RunSession::frame() const { return impl_->frame; }
int RunSession::first_frame() const { return impl_->first; }
int RunSession::total_frames() const { return impl_->desc.simulation.frames; }
const SceneDesc& RunSession::desc() const { return impl_->desc; }
const Scene& RunSession::scene() const { return impl_->scene; }
Stepper& RunSession::stepper() { return *impl_->stepper; }
const StepStats& RunSession::last_stats() const { return impl_->stepper->last_stats(); }
std::string RunSession::output_dir() const { return impl_->out_dir.string(); }

int run_simulation(const SceneDesc& desc, const RunOptions& options) {
    RunSession session(desc, options);
    while (session.step()) {}
    return session.exit_code();
}

int run_simulation_file(const std::string& config_path, const RunOptions& options) {
    SceneDesc desc = load_scene_desc(config_path);
    return run_simulation(desc, options);
}

}  // namespace cs
