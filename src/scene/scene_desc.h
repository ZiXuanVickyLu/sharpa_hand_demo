#pragma once
// Parsed, validated, defaults-filled configuration (doc/al-ipc-implementation-spec.md §5.3).
// Host-only, Eigen types; the device side never sees these structs.
#include <Eigen/Core>
#include <nlohmann/json_fwd.hpp>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cs {

using Vec3 = Eigen::Vector3d;

struct SimulationDesc {
    double dt = 0.01;
    int frames = 100;
    Vec3 gravity = Vec3(0, -9.81, 0);
    std::string output_dir = "output/run";
    bool save_bgeo = true;
    bool save_npy = false;              // x_<step>.npy positions (float32, global numbering) + surface_faces.npy
    int save_every = 1;
    // IS §5.x restart: checkpoints (float64 x and v every N steps, 0 = off), restart from a saved state
    // (restart_step >= 1 with a positions file; velocities optional, empty = from rest), and the guard that
    // stops a run after N consecutive steps at the outer iteration cap (0 = off).
    int checkpoint_every = 0;
    int restart_step = -1;
    std::string restart_positions;
    std::string restart_velocities;
    int abort_after_capped_steps = 0;
};

struct StallDesc {
    int iters = 50;
    double alpha = 1e-4;
    double mu_factor = 2.0;
    double d_hat_factor = 0.5;
    int max_adaptations = 10;
};

struct CcdDesc {
    double s = 0.1;                     // ACCD early-stop ratio
    int max_iter = 100;                 // ACCD iteration cap
    bool float_screen = false;          // certified float pre-filter (§12.4, off)
    double rebuild_quality_ratio = 1.5; // tree quality trigger for an extra rebuild
};

struct FrictionDesc {
    bool enable = true;
    double eps_v = 1e-3;                // m/s, IPC smoothing velocity
    std::string normal_force = "paper"; // "paper" (MS §13) | "lambda"
};

struct ContactDesc {
    bool enable = true;
    double d_hat = 1e-3;                // contact offset δ̂ (m)
    double epsilon = 1e-3;              // termination ε (MS §15)
    int K_min = 2;
    int max_outer_iters = 500;
    double decay_factor = 0.9;          // Γ
    double decay_remove_threshold = 0.01; // γ_min
    std::string mu_mode = "diag_max";   // "diag_max" | "fixed"
    double mu_scale = 0.1;              // C_μ
    std::optional<double> mu_fixed;     // kg, when mu_mode == "fixed"
    std::optional<double> mu_max;       // kg, upper bound of the estimate (MS §11.1); none = unbounded
    double alpha_lower_bound = 1e-6;
    double toi_tie_tolerance = 1e-6;
    StallDesc stall;
    CcdDesc ccd;
    std::string inversion_free = "auto"; // "auto" | "on" | "off"
    bool self_collision_default = true;
    bool exclude_one_ring_pairs = false;
    bool drop_parallel_edge_pairs = false;
    std::string toi_filter_domain = "paper"; // "paper" (new hits) | "all"
    FrictionDesc friction;
};

struct LineSearchDesc {
    int max_halvings = 30;
    double energy_tolerance = 1e-12;
    bool batched = true;   // IS §12.3 item 10: r, r/2, r/4 evaluated in one pass on small systems
};

struct NewtonDesc {
    int inner_max_iters = 8;
    LineSearchDesc line_search;
    double early_accept_velocity_tol = 0.0; // 0 = off
    // C-IPC-style Newton tolerance (m/s, 0 = off): an accepted full step ends the inner loop
    // only when the RMS increment velocity of the free vertices is below it (MS §7).
    double increment_velocity_tol = 0.0;
};

struct LinearSolverDesc {
    std::string type = "pcg";
    std::string preconditioner = "block_jacobi";
    double rel_tol = 1e-4;
    int max_iters = 2000;
    int check_interval = 8;                  // PCG iterations per captured graph replay (IS §6.6)
    bool graph = true;                       // false = plain launch loop with host reductions
    bool fused = true;                       // IS §12.3 item 14: one cooperative kernel per solve on small systems
    int fused_max_rows = 65536;
    bool rigid_coarse = false;               // IS §6.6 two-level preconditioner: per-body rigid-mode coarse correction
    int rigid_coarse_max_rows = 4096;        // bodies with more free rows get no correction
};

// The three cloth models are membranes (math spec §3.5); their integer values continue the tet
// models' so the device enum fem::Model matches by construction.
enum class MaterialModel { SNH, NH, COR, ClothStVK, ClothCOR, ClothSNH };
inline bool is_cloth_model(MaterialModel m) { return m >= MaterialModel::ClothStVK; }

struct MaterialDesc {
    std::string name;
    MaterialModel model = MaterialModel::SNH;
    double E = 1e6;
    double nu = 0.3;
    double density = 1000.0;
    bool snh_lambda_reparam = true;
    // Cloth only (MS §3.5, §3.7): membrane shear modulus (given directly as [Z25]'s mu_mem, or
    // derived from E and nu when absent), shell thickness, bending stiffness, bending Hessian.
    std::optional<double> mu_mem;
    double thickness = 1e-3;
    double k_bend = 0.0;
    std::string bending_hessian = "gauss_newton";
    // Derived Lamé parameters (filled by the parser): mu_L, lambda_L, and for SNH the
    // reparameterized lambda_hat and alpha_hat (MS §3.2).
    double mu_lame = 0.0;
    double lambda_lame = 0.0;
    double lambda_hat = 0.0;
    double alpha_hat = 1.0;
};

struct MotionDesc {
    enum class Type { None, Rotation, Translation, Scale, Keyframes };
    Type type = Type::None;
    Vec3 axis = Vec3(0, 1, 0);
    std::optional<Vec3> center;         // none = centroid of the selected vertices at capture
    double deg_per_second = 0.0;
    Vec3 velocity = Vec3::Zero();
    double rate = 0.0;                  // scale rate per second
    int start_frame = 0;
    int end_frame = -1;                 // -1 = never stop
    double ramp_time = 0.0;             // s; scripted types: the rate rises linearly from 0 over this time (IS §5.x)
    std::string keyframe_file;          // .npy [frames x n x 3] for Keyframes
    int substeps = 1;                   // keyframe playback: solver steps per keyframe
};

struct SelectionDesc {
    enum class Mode { All, Indices, Bbox };
    Mode mode = Mode::All;
    std::vector<int> indices;
    Vec3 bbox_min = Vec3::Zero();       // fractions of the placed bounding box
    Vec3 bbox_max = Vec3::Ones();
};

struct DirichletDesc {
    SelectionDesc select;
    MotionDesc motion;
    // IS §5.x: >= 0 makes this a held free region, projected onto its motion's targets until
    // that frame and an ordinary free part of the body from it on (the C-IPC card release).
    int release_frame = -1;
};

struct PlacementDesc {
    bool move_to_origin = false;
    Vec3 scale = Vec3::Ones();
    Vec3 rotation_deg = Vec3::Zero();   // Euler XYZ, degrees
    Vec3 translation = Vec3::Zero();
};

struct BodyDesc {
    std::string name;
    std::string type = "tet";           // "tet" | "cloth" (a triangle mesh, MS §3.5)
    std::string file;                   // resolved to an absolute path by the parser
    std::string material;
    PlacementDesc placement;
    Vec3 initial_velocity = Vec3::Zero();
    double thickness = 0.0;
    double friction = 0.0;
    std::optional<bool> self_collision; // none = contact.self_collision_default
    std::vector<DirichletDesc> dirichlet;
    int collision_group = -1;
};

struct ColliderDesc {
    std::string name;
    std::string type = "tri";
    std::string file;
    PlacementDesc placement;
    double thickness = 0.0;
    double friction = 0.0;
    MotionDesc motion;                  // None = static collider
};

struct PlaneDesc {
    Vec3 point = Vec3::Zero();
    Vec3 normal = Vec3(0, 1, 0);
    double friction = 0.0;
};

// Analytic sphere collider (math spec §4.1 PS, §12.3). inverted = the bodies live inside it.
// radius_keyframes are (time in seconds, radius) pairs, piecewise linear, held after the last;
// empty means the constant `radius`.
struct SphereDesc {
    Vec3 center = Vec3::Zero();
    double radius = 1.0;
    bool inverted = false;
    double friction = 0.0;
    std::vector<std::pair<double, double>> radius_keyframes;
};

struct ContactTableDesc {
    std::vector<std::pair<std::string, std::string>> exclude;
    struct Friction { std::string a, b; double mu; };
    std::vector<Friction> friction;
};

struct LoggingDesc {
    std::string level = "info";
    std::string stats_csv = "stats.csv";
    bool timing = true;
    bool debug_check_penetration = false;
};

struct SceneDesc {
    SimulationDesc simulation;
    ContactDesc contact;
    NewtonDesc newton;
    LinearSolverDesc linear_solver;
    std::map<std::string, MaterialDesc> materials;
    std::vector<BodyDesc> bodies;
    std::vector<ColliderDesc> colliders;
    std::vector<PlaneDesc> planes;
    std::vector<SphereDesc> spheres;
    ContactTableDesc contact_table;
    LoggingDesc logging;
    std::string base_dir;               // directory relative asset paths are resolved against
};

// Parse and validate. Unknown keys are errors. Relative "file" entries are resolved against
// `base_dir` first, then against CS_ASSET_PATH. Throws std::runtime_error with a path-like
// message ("bodies[1].material: unknown material 'x'").
SceneDesc parse_scene_desc(const nlohmann::json& j, const std::string& base_dir);

// Convenience: read the file, derive base_dir from it.
SceneDesc load_scene_desc(const std::string& json_path);

// The fully resolved configuration (all defaults filled), for the run's resolved_config.json.
nlohmann::json scene_desc_to_json(const SceneDesc& desc);

}  // namespace cs
