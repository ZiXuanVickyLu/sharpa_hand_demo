// Interactive viewer (IS §5.x "Interactive viewer cs_view"):
//   cs_view <config.json> [--play] [--frames N] [--output DIR] [--no-export] [--steps-per-frame K] [--record DIR]
// The same RunSession as cs_run, stepped from the polyscope UI callback: what is drawn is the
// solver's state after each step. One surface mesh per body and per collider.
#include <solver/run.h>
#include <solver/stepper.h>
#include <scene/scene.h>
#include <scene/scene_desc.h>
#include <core/version.h>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <imgui.h>
#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Group {                       // one drawn mesh: a body's or a collider's part of the scene surface
    std::string name;
    std::vector<int> global;         // local vertex -> global vertex
    std::vector<std::array<int, 3>> faces;
    std::vector<std::array<double, 3>> pos;
    polyscope::SurfaceMesh* mesh = nullptr;
    bool collider = false;
};

struct Viewer {
    std::unique_ptr<cs::RunSession> session;
    std::vector<Group> groups;
    std::vector<int> dirichlet;      // prescribed body vertices (global)
    polyscope::PointCloud* dirichlet_cloud = nullptr;
    Eigen::Matrix3Xd X;
    bool playing = false;
    int steps_per_frame = 1;
    int stop_at = 0;
    double total_ms = 0.0;
    long long total_newton = 0, total_pcg = 0;
    int steps_done = 0, capped_steps = 0;
    std::string record_dir;
    std::string message;

    void build_groups() {
        const cs::Scene& sc = session->scene();
        const int nb_all = int(sc.bodies.size());
        std::vector<std::vector<int>> local(static_cast<size_t>(nb_all));
        groups.assign(static_cast<size_t>(nb_all), Group{});
        for (int b = 0; b < nb_all; ++b) {
            groups[size_t(b)].name = sc.bodies[size_t(b)].name.empty() ? ("body_" + std::to_string(b)) : sc.bodies[size_t(b)].name;
            groups[size_t(b)].collider = sc.bodies[size_t(b)].kind != cs::BodyKind::Deformable;
            local[size_t(b)].assign(size_t(sc.n_vertices), -1);
        }
        for (const auto& t : sc.surf_tris) {
            const int b = sc.body_id[size_t(t[0])];
            if (b < 0 || b >= nb_all) continue;
            Group& g = groups[size_t(b)];
            std::array<int, 3> f{};
            for (int k = 0; k < 3; ++k) {
                int& l = local[size_t(b)][size_t(t[k])];
                if (l < 0) { l = int(g.global.size()); g.global.push_back(t[k]); }
                f[k] = l;
            }
            g.faces.push_back(f);
        }
        std::vector<Group> kept;
        for (auto& g : groups) if (!g.faces.empty()) kept.push_back(std::move(g));
        groups.swap(kept);
        for (int v = 0; v < sc.n_body_vertices; ++v)
            if (sc.is_prescribed[size_t(v)]) dirichlet.push_back(v);
    }

    void pull_positions() {
        session->stepper().positions(X);
        for (auto& g : groups) {
            g.pos.resize(g.global.size());
            for (size_t i = 0; i < g.global.size(); ++i) {
                const auto c = X.col(g.global[i]);
                g.pos[i] = {c[0], c[1], c[2]};
            }
        }
    }

    void register_all() {
        pull_positions();
        static const std::array<std::array<float, 3>, 6> palette = {{{0.90f, 0.62f, 0.20f}, {0.22f, 0.49f, 0.78f}, {0.35f, 0.66f, 0.38f},
                                                                     {0.78f, 0.30f, 0.32f}, {0.58f, 0.44f, 0.76f}, {0.85f, 0.78f, 0.30f}}};
        int k = 0;
        for (auto& g : groups) {
            g.mesh = polyscope::registerSurfaceMesh(g.name, g.pos, g.faces);
            if (g.collider) {
                g.mesh->setSurfaceColor({0.72f, 0.74f, 0.78f});
            } else {
                const auto& c = palette[size_t(k++) % palette.size()];
                g.mesh->setSurfaceColor({c[0], c[1], c[2]});
            }
            g.mesh->setSmoothShade(true);
        }
        if (!dirichlet.empty()) {
            dirichlet_cloud = polyscope::registerPointCloud("dirichlet", dirichlet_positions());
            dirichlet_cloud->setEnabled(false);
        }
    }

    std::vector<std::array<double, 3>> dirichlet_positions() const {
        std::vector<std::array<double, 3>> p(dirichlet.size());
        for (size_t i = 0; i < dirichlet.size(); ++i) p[i] = {X(0, dirichlet[i]), X(1, dirichlet[i]), X(2, dirichlet[i])};
        return p;
    }

    void push_positions() {
        pull_positions();
        for (auto& g : groups) g.mesh->updateVertexPositions(g.pos);
        if (dirichlet_cloud) dirichlet_cloud->updatePointPositions(dirichlet_positions());
    }

    // One solver step; false when the run cannot continue.
    bool step_once() {
        if (session->finished()) return false;
        const bool more = session->step();
        const cs::StepStats& s = session->last_stats();
        total_ms += s.time_ms;
        total_newton += s.inner_newton;
        total_pcg += s.pcg_iters_total;
        ++steps_done;
        if (s.hit_iteration_cap) {
            ++capped_steps;
            playing = false;
            message = "step " + std::to_string(session->frame()) + " ended at the outer iteration cap: paused";
        }
        if (!more && session->exit_code() == 4) message = "stopped by simulation.abort_after_capped_steps";
        return more;
    }

    void callback() {
        const cs::SceneDesc& d = session->desc();
        const double h = d.simulation.dt;
        ImGui::PushItemWidth(140);
        ImGui::Text("frame %d / %d   t = %.3f s", session->frame(), session->total_frames(), session->frame() * h);
        if (session->finished()) {
            ImGui::TextUnformatted(session->exit_code() == 0 ? "run finished" : "run stopped");
            playing = false;
        } else {
            if (ImGui::Button(playing ? "Pause" : "Play")) { playing = !playing; message.clear(); }
            ImGui::SameLine();
            const bool single = ImGui::Button("Step");
            ImGui::SameLine();
            ImGui::InputInt("steps / UI frame", &steps_per_frame);
            steps_per_frame = std::max(1, std::min(steps_per_frame, 1000));
            ImGui::InputInt("pause at frame", &stop_at);
            if (playing || single) {
                const int n = single ? 1 : steps_per_frame;
                bool moved = false;
                for (int k = 0; k < n; ++k) {
                    const bool more = step_once();
                    moved = true;
                    if (!more || !playing) break;
                    if (stop_at > 0 && session->frame() >= stop_at) { playing = false; break; }
                }
                if (moved) {
                    push_positions();
                    if (!record_dir.empty()) {
                        char name[64];
                        std::snprintf(name, sizeof(name), "frame_%05d.png", session->frame());
                        polyscope::screenshot((std::filesystem::path(record_dir) / name).string(), false);
                    }
                }
                if (stop_at > 0 && session->frame() >= stop_at) playing = false;
            }
        }
        if (!message.empty()) ImGui::TextColored(ImVec4(1.0f, 0.55f, 0.2f, 1.0f), "%s", message.c_str());
        ImGui::Separator();
        if (steps_done > 0) {
            const cs::StepStats& s = session->last_stats();
            ImGui::Text("last step: %.1f ms", s.time_ms);
            ImGui::Text("  outer %d   Newton %d   CG / solve %.1f", s.outer_iters, s.inner_newton, s.pcg_iters_avg);
            ImGui::Text("  pairs PT %d  EE %d  plane %d", s.pairs_pt, s.pairs_ee, s.pairs_ph);
            ImGui::Text("  alpha mean %.3f  min %.3f   beta %.1e", s.alpha_mean, s.alpha_min, s.beta_final);
            ImGui::Text("  energy %.6e   GPU %.0f MB", s.energy, s.gpu_mem_mb);
            ImGui::Separator();
            ImGui::Text("run: %d steps, %.1f s of solver time", steps_done, total_ms * 1e-3);
            ImGui::Text("  %.1f ms / step   %.2f Newton / step", total_ms / steps_done, double(total_newton) / steps_done);
            ImGui::Text("  %.1f CG / solve   capped steps %d", total_newton ? double(total_pcg) / double(total_newton) : 0.0, capped_steps);
        } else {
            ImGui::TextUnformatted("no step taken yet");
        }
        ImGui::Separator();
        const cs::Scene& sc = session->scene();
        ImGui::Text("%d vertices (%d free), %d tets, %zu surface tris", sc.n_vertices, sc.n_free, sc.n_tets, sc.surf_tris.size());
        ImGui::Text("h = %g s   d_hat = %.3e m   friction %s", h, d.contact.d_hat, d.contact.friction.enable ? "on" : "off");
        ImGui::TextWrapped("output: %s", session->output_dir().c_str());
        ImGui::PopItemWidth();
    }
};

Viewer* g_viewer = nullptr;

}  // namespace

int main(int argc, char** argv) {
    spdlog::info("contact_solver {} viewer", cs::version_string());
    if (argc < 2) {
        spdlog::error("usage: cs_view <config.json> [--play] [--frames N] [--output DIR] [--no-export] "
                      "[--steps-per-frame K] [--record DIR]");
        return 2;
    }
    Viewer viewer;
    cs::RunOptions opt;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--play") == 0) viewer.playing = true;
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) opt.frames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) opt.output_dir = argv[++i];
        else if (std::strcmp(argv[i], "--no-export") == 0) opt.export_frames = false;
        else if (std::strcmp(argv[i], "--steps-per-frame") == 0 && i + 1 < argc) viewer.steps_per_frame = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--record") == 0 && i + 1 < argc) viewer.record_dir = argv[++i];
        else {
            spdlog::error("unknown argument {}", argv[i]);
            return 2;
        }
    }
    try {
        const cs::SceneDesc desc = cs::load_scene_desc(argv[1]);
        viewer.session.reset(new cs::RunSession(desc, opt));
        if (!viewer.record_dir.empty()) std::filesystem::create_directories(viewer.record_dir);

        // up direction: against gravity when there is one, else the plane normal, else +y
        Eigen::Vector3d up(0, 1, 0);
        if (desc.simulation.gravity.norm() > 0) up = -desc.simulation.gravity.normalized();
        else if (!desc.planes.empty()) up = desc.planes[0].normal.normalized();
        int ax = 0;
        up.cwiseAbs().maxCoeff(&ax);
        polyscope::options::programName = std::string("cs_view - ") + std::filesystem::path(argv[1]).filename().string();
        polyscope::options::autocenterStructures = false;
        polyscope::options::autoscaleStructures = false;
        polyscope::options::groundPlaneMode = desc.planes.empty() ? polyscope::GroundPlaneMode::None
                                                                  : polyscope::GroundPlaneMode::TileReflection;
        polyscope::init();
        const bool neg = up[ax] < 0;
        polyscope::view::setUpDir(ax == 0 ? (neg ? polyscope::UpDir::NegXUp : polyscope::UpDir::XUp)
                                  : ax == 1 ? (neg ? polyscope::UpDir::NegYUp : polyscope::UpDir::YUp)
                                            : (neg ? polyscope::UpDir::NegZUp : polyscope::UpDir::ZUp));
        if (!desc.planes.empty()) {
            polyscope::options::groundPlaneHeightMode = polyscope::GroundPlaneHeightMode::Manual;
            polyscope::options::groundPlaneHeight = float(desc.planes[0].point[ax]);
        }
        viewer.build_groups();
        viewer.register_all();
        polyscope::view::resetCameraToHomeView();
        g_viewer = &viewer;
        polyscope::state::userCallback = [] { g_viewer->callback(); };
        polyscope::show();
        g_viewer = nullptr;
        return viewer.session->exit_code();
    } catch (const std::exception& e) {
        spdlog::error("fatal: {}", e.what());
        return 1;
    }
}
