// Strict JSON -> SceneDesc parser, defaults, validation, path resolution and the inverse
// (scene_desc_to_json) for resolved_config.json. Schema: doc/al-ipc-implementation-spec.md §5.3.
//
// Every object is read through a Reader that records the keys it consumed; any key left over is
// reported as "<json.path>.<key>: unknown key". A JSON null is treated as "not given" (the default
// applies), which is also how the optional slots of the schema ("mu_fixed": null, "motion": null,
// "center": null) are expressed.
#include "scene/scene_desc.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#ifndef CS_ASSET_PATH
#define CS_ASSET_PATH ""
#endif

namespace cs {
namespace {

using json = nlohmann::json;
namespace fs = std::filesystem;

[[noreturn]] void fail(const std::string& path, const std::string& msg)
{
    throw std::runtime_error(path + ": " + msg);
}

std::string join(const std::string& path, const std::string& key)
{
    return path.empty() ? key : path + "." + key;
}

std::string indexed(const std::string& path, size_t i)
{
    return path + "[" + std::to_string(i) + "]";
}

// ---------------------------------------------------------------------------
// Typed access with path-qualified errors
// ---------------------------------------------------------------------------
double as_number(const json& v, const std::string& path)
{
    if (!v.is_number()) fail(path, "expected a number");
    return v.get<double>();
}

int as_int(const json& v, const std::string& path)
{
    if (v.is_number_integer()) {
        const auto x = v.get<long long>();
        if (x < -2147483648LL || x > 2147483647LL) fail(path, "integer out of range");
        return int(x);
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (std::floor(d) == d && std::fabs(d) < 2147483647.0) return int(d);
    }
    fail(path, "expected an integer");
}

bool as_bool(const json& v, const std::string& path)
{
    if (!v.is_boolean()) fail(path, "expected a boolean");
    return v.get<bool>();
}

std::string as_string(const json& v, const std::string& path)
{
    if (!v.is_string()) fail(path, "expected a string");
    return v.get<std::string>();
}

Vec3 as_vec3(const json& v, const std::string& path)
{
    if (!v.is_array() || v.size() != 3) fail(path, "expected an array of 3 numbers");
    Vec3 r;
    for (int i = 0; i < 3; ++i) r[i] = as_number(v[i], indexed(path, i));
    return r;
}

void check_array(const json& v, const std::string& path)
{
    if (!v.is_array()) fail(path, "expected an array");
}

void check(bool ok, const std::string& path, const std::string& msg)
{
    if (!ok) fail(path, msg);
}

void check_enum(const std::string& value, std::initializer_list<const char*> allowed, const std::string& path)
{
    for (const char* a : allowed)
        if (value == a) return;
    std::ostringstream os;
    os << "invalid value '" << value << "' (expected one of:";
    for (const char* a : allowed) os << " '" << a << "'";
    os << ")";
    fail(path, os.str());
}

// A JSON object read with strict key accounting: finish() rejects every key that was not asked for.
class Reader {
public:
    Reader(const json& j, std::string path) : j_(j), path_(std::move(path))
    {
        if (!j_.is_object()) fail(path_, "expected an object");
    }
    const std::string& path() const { return path_; }
    std::string key_path(const std::string& k) const { return join(path_, k); }

    // The value of `k` if present and not null (null counts as "use the default").
    const json* get(const std::string& k)
    {
        used_.insert(k);
        auto it = j_.find(k);
        if (it == j_.end() || it->is_null()) return nullptr;
        return &*it;
    }
    const json& require(const std::string& k)
    {
        const json* v = get(k);
        if (!v) fail(key_path(k), "missing required key");
        return *v;
    }

    double number(const std::string& k, double def)
    {
        const json* v = get(k);
        return v ? as_number(*v, key_path(k)) : def;
    }
    int integer(const std::string& k, int def)
    {
        const json* v = get(k);
        return v ? as_int(*v, key_path(k)) : def;
    }
    bool boolean(const std::string& k, bool def)
    {
        const json* v = get(k);
        return v ? as_bool(*v, key_path(k)) : def;
    }
    std::string string(const std::string& k, const std::string& def)
    {
        const json* v = get(k);
        return v ? as_string(*v, key_path(k)) : def;
    }
    Vec3 vec3(const std::string& k, const Vec3& def)
    {
        const json* v = get(k);
        return v ? as_vec3(*v, key_path(k)) : def;
    }
    // Sub-object; a missing/null one yields an empty object (all defaults).
    Reader child(const std::string& k)
    {
        const json* v = get(k);
        static const json empty = json::object();
        return Reader(v ? *v : empty, key_path(k));
    }

    // Keys whose name starts with '_' are comments and are ignored at every level.
    void finish() const
    {
        for (auto it = j_.begin(); it != j_.end(); ++it) {
            if (!it.key().empty() && it.key()[0] == '_') continue;
            if (!used_.count(it.key())) fail(key_path(it.key()), "unknown key '" + it.key() + "'");
        }
    }

private:
    const json& j_;
    std::string path_;
    std::set<std::string> used_;
};

// ---------------------------------------------------------------------------
// File resolution: absolute as-is; else base_dir/file, CS_ASSET_PATH/file, and — since configs
// write asset paths relative to the project root ("asset/basic/x.bgeo") — the asset directory
// with a leading "asset/" stripped and the project root itself.
// ---------------------------------------------------------------------------
std::string resolve_file(const std::string& file, const std::string& base_dir, const std::string& path)
{
    check(!file.empty(), path, "empty file name");
    const fs::path f(file);
    std::vector<fs::path> candidates;
    if (f.is_absolute()) {
        candidates.push_back(f);
    } else {
        candidates.push_back(fs::path(base_dir.empty() ? "." : base_dir) / f);
        std::string asset = CS_ASSET_PATH;
        while (!asset.empty() && (asset.back() == '/' || asset.back() == '\\')) asset.pop_back();
        if (!asset.empty()) {
            candidates.push_back(fs::path(asset) / f);
            if (file.rfind("asset/", 0) == 0) candidates.push_back(fs::path(asset) / file.substr(6));
            candidates.push_back(fs::path(asset).parent_path() / f);
        }
    }
    for (const fs::path& c : candidates) {
        std::error_code ec;
        if (fs::is_regular_file(c, ec)) return fs::absolute(c).lexically_normal().string();
    }
    // Not on disk: parsing a configuration must not depend on the assets being present (the big
    // ones are fetched on demand), so keep the most plausible candidate — the first one whose
    // directory exists — and let the mesh loader report the missing file when build_scene needs it.
    for (const fs::path& c : candidates) {
        std::error_code ec;
        if (fs::is_directory(c.parent_path(), ec)) return fs::absolute(c).lexically_normal().string();
    }
    return fs::absolute(candidates.front()).lexically_normal().string();
}

// ---------------------------------------------------------------------------
// Section parsers
// ---------------------------------------------------------------------------
SimulationDesc parse_simulation(Reader r)
{
    SimulationDesc d;
    d.dt = r.number("dt", d.dt);
    d.frames = r.integer("frames", d.frames);
    d.gravity = r.vec3("gravity", d.gravity);
    d.output_dir = r.string("output_dir", d.output_dir);
    d.save_bgeo = r.boolean("save_bgeo", d.save_bgeo);
    d.save_npy = r.boolean("save_npy", d.save_npy);
    d.save_every = r.integer("save_every", d.save_every);
    d.checkpoint_every = r.integer("checkpoint_every", d.checkpoint_every);
    d.abort_after_capped_steps = r.integer("abort_after_capped_steps", d.abort_after_capped_steps);
    if (r.get("restart") != nullptr) {   // IS §5.x restart
        Reader rr = r.child("restart");
        d.restart_step = as_int(rr.require("step"), rr.key_path("step"));
        d.restart_positions = as_string(rr.require("positions"), rr.key_path("positions"));
        d.restart_velocities = rr.string("velocities", "");
        rr.finish();
        check(d.restart_step >= 1, rr.key_path("step"), "must be >= 1 (step 0 is the scene's own initial state)");
        check(!d.restart_positions.empty(), rr.key_path("positions"), "must name an npy file");
    }
    // §5.3 lists "seed_velocity": null as a placeholder; per-body velocities live in bodies[].
    check(r.get("seed_velocity") == nullptr, r.key_path("seed_velocity"),
          "not supported; use bodies[].initial_velocity");
    r.finish();
    check(d.dt > 0, r.key_path("dt"), "must be > 0");
    check(d.frames >= 0, r.key_path("frames"), "must be >= 0");
    check(d.save_every >= 1, r.key_path("save_every"), "must be >= 1");
    check(d.checkpoint_every >= 0, r.key_path("checkpoint_every"), "must be >= 0");
    check(d.abort_after_capped_steps >= 0, r.key_path("abort_after_capped_steps"), "must be >= 0");
    check(d.restart_step < 0 || d.restart_step <= d.frames, r.key_path("restart"), "restart.step must be <= frames");
    return d;
}

StallDesc parse_stall(Reader r)
{
    StallDesc d;
    d.iters = r.integer("iters", d.iters);
    d.alpha = r.number("alpha", d.alpha);
    d.mu_factor = r.number("mu_factor", d.mu_factor);
    d.d_hat_factor = r.number("d_hat_factor", d.d_hat_factor);
    d.max_adaptations = r.integer("max_adaptations", d.max_adaptations);
    r.finish();
    check(d.iters >= 1, r.key_path("iters"), "must be >= 1");
    check(d.alpha > 0, r.key_path("alpha"), "must be > 0");
    check(d.mu_factor >= 1, r.key_path("mu_factor"), "must be >= 1");
    check(d.d_hat_factor > 0 && d.d_hat_factor <= 1, r.key_path("d_hat_factor"), "must be in (0, 1]");
    check(d.max_adaptations >= 0, r.key_path("max_adaptations"), "must be >= 0");
    return d;
}

CcdDesc parse_ccd(Reader r)
{
    CcdDesc d;
    d.s = r.number("s", d.s);
    d.max_iter = r.integer("max_iter", d.max_iter);
    d.float_screen = r.boolean("float_screen", d.float_screen);
    d.rebuild_quality_ratio = r.number("rebuild_quality_ratio", d.rebuild_quality_ratio);
    r.finish();
    check(d.s > 0 && d.s < 1, r.key_path("s"), "must be in (0, 1)");
    check(d.max_iter >= 1, r.key_path("max_iter"), "must be >= 1");
    check(d.rebuild_quality_ratio >= 1, r.key_path("rebuild_quality_ratio"), "must be >= 1");
    return d;
}

FrictionDesc parse_friction(Reader r)
{
    FrictionDesc d;
    d.enable = r.boolean("enable", d.enable);
    d.eps_v = r.number("eps_v", d.eps_v);
    d.normal_force = r.string("normal_force", d.normal_force);
    r.finish();
    check(d.eps_v > 0, r.key_path("eps_v"), "must be > 0");
    check_enum(d.normal_force, {"paper", "lambda"}, r.key_path("normal_force"));
    return d;
}

ContactDesc parse_contact(Reader r)
{
    ContactDesc d;
    d.enable = r.boolean("enable", d.enable);
    d.d_hat = r.number("d_hat", d.d_hat);
    d.epsilon = r.number("epsilon", d.epsilon);
    d.K_min = r.integer("K_min", d.K_min);
    d.max_outer_iters = r.integer("max_outer_iters", d.max_outer_iters);
    d.decay_factor = r.number("decay_factor", d.decay_factor);
    d.decay_remove_threshold = r.number("decay_remove_threshold", d.decay_remove_threshold);
    d.mu_mode = r.string("mu_mode", d.mu_mode);
    d.mu_scale = r.number("mu_scale", d.mu_scale);
    if (const json* v = r.get("mu_fixed")) d.mu_fixed = as_number(*v, r.key_path("mu_fixed"));
    if (const json* v = r.get("mu_max")) d.mu_max = as_number(*v, r.key_path("mu_max"));
    d.alpha_lower_bound = r.number("alpha_lower_bound", d.alpha_lower_bound);
    d.toi_tie_tolerance = r.number("toi_tie_tolerance", d.toi_tie_tolerance);
    d.stall = parse_stall(r.child("stall"));
    d.ccd = parse_ccd(r.child("ccd"));
    if (const json* v = r.get("inversion_free")) {
        // "auto" | "on" | "off"; §5.3 also allows JSON true/false to force it.
        if (v->is_boolean()) d.inversion_free = v->get<bool>() ? "on" : "off";
        else d.inversion_free = as_string(*v, r.key_path("inversion_free"));
    }
    d.self_collision_default = r.boolean("self_collision_default", d.self_collision_default);
    d.exclude_one_ring_pairs = r.boolean("exclude_one_ring_pairs", d.exclude_one_ring_pairs);
    d.drop_parallel_edge_pairs = r.boolean("drop_parallel_edge_pairs", d.drop_parallel_edge_pairs);
    d.toi_filter_domain = r.string("toi_filter_domain", d.toi_filter_domain);
    d.friction = parse_friction(r.child("friction"));
    r.finish();

    // MS §16 lists both of these as options on top of the paper's behaviour; only the paper's
    // setting (keep one-ring pairs, keep near-parallel edge-edge pairs with their sub-gradient)
    // is implemented, so refuse the other value rather than accepting it and ignoring it.
    check(!d.exclude_one_ring_pairs, r.key_path("exclude_one_ring_pairs"),
          "is not implemented; only false (the paper's behaviour) is supported");
    // drop_parallel_edge_pairs (MS §16): implemented in M12 for cloth self-contact, where the
    // near-parallel edge-edge pairs of a folded sheet have no well-defined separating direction.
    check(d.d_hat >= 0, r.key_path("d_hat"), "must be >= 0");
    check(d.epsilon > 0, r.key_path("epsilon"), "must be > 0");
    check(d.K_min >= 1, r.key_path("K_min"), "must be >= 1");
    check(d.max_outer_iters >= 1, r.key_path("max_outer_iters"), "must be >= 1");
    check(d.decay_factor > 0 && d.decay_factor < 1, r.key_path("decay_factor"), "must be in (0, 1)");
    check(d.decay_remove_threshold > 0 && d.decay_remove_threshold < 1, r.key_path("decay_remove_threshold"),
          "must be in (0, 1)");
    check_enum(d.mu_mode, {"diag_max", "fixed"}, r.key_path("mu_mode"));
    check(d.mu_scale > 0, r.key_path("mu_scale"), "must be > 0");
    if (d.mu_fixed) check(*d.mu_fixed > 0, r.key_path("mu_fixed"), "must be > 0");
    if (d.mu_max) check(*d.mu_max > 0, r.key_path("mu_max"), "must be > 0");
    if (d.mu_mode == "fixed") check(d.mu_fixed.has_value(), r.key_path("mu_fixed"), "required when mu_mode is 'fixed'");
    check(d.alpha_lower_bound > 0 && d.alpha_lower_bound < 1, r.key_path("alpha_lower_bound"), "must be in (0, 1)");
    check(d.toi_tie_tolerance >= 0, r.key_path("toi_tie_tolerance"), "must be >= 0");
    check_enum(d.inversion_free, {"auto", "on", "off"}, r.key_path("inversion_free"));
    check_enum(d.toi_filter_domain, {"paper", "all"}, r.key_path("toi_filter_domain"));
    return d;
}

NewtonDesc parse_newton(Reader r)
{
    NewtonDesc d;
    d.inner_max_iters = r.integer("inner_max_iters", d.inner_max_iters);
    {
        Reader ls = r.child("line_search");
        d.line_search.max_halvings = ls.integer("max_halvings", d.line_search.max_halvings);
        d.line_search.energy_tolerance = ls.number("energy_tolerance", d.line_search.energy_tolerance);
        d.line_search.batched = ls.boolean("batched", d.line_search.batched);
        ls.finish();
        check(d.line_search.max_halvings >= 0, ls.key_path("max_halvings"), "must be >= 0");
        check(d.line_search.energy_tolerance >= 0, ls.key_path("energy_tolerance"), "must be >= 0");
    }
    d.early_accept_velocity_tol = r.number("early_accept_velocity_tol", d.early_accept_velocity_tol);
    d.increment_velocity_tol = r.number("increment_velocity_tol", d.increment_velocity_tol);
    r.finish();
    check(d.inner_max_iters >= 1, r.key_path("inner_max_iters"), "must be >= 1");
    check(d.early_accept_velocity_tol >= 0, r.key_path("early_accept_velocity_tol"), "must be >= 0");
    check(d.increment_velocity_tol >= 0, r.key_path("increment_velocity_tol"), "must be >= 0");
    return d;
}

LinearSolverDesc parse_linear_solver(Reader r)
{
    LinearSolverDesc d;
    d.type = r.string("type", d.type);
    d.preconditioner = r.string("preconditioner", d.preconditioner);
    d.rel_tol = r.number("rel_tol", d.rel_tol);
    d.max_iters = r.integer("max_iters", d.max_iters);
    d.check_interval = r.integer("check_interval", d.check_interval);
    d.graph = r.boolean("graph", d.graph);
    d.fused = r.boolean("fused", d.fused);
    d.fused_max_rows = r.integer("fused_max_rows", d.fused_max_rows);
    d.rigid_coarse = r.boolean("rigid_coarse", d.rigid_coarse);
    d.rigid_coarse_max_rows = r.integer("rigid_coarse_max_rows", d.rigid_coarse_max_rows);
    r.finish();
    check_enum(d.type, {"pcg"}, r.key_path("type"));
    check_enum(d.preconditioner, {"block_jacobi", "jacobi", "none"}, r.key_path("preconditioner"));
    check(d.rel_tol > 0, r.key_path("rel_tol"), "must be > 0");
    check(d.max_iters >= 1, r.key_path("max_iters"), "must be >= 1");
    check(d.check_interval >= 1, r.key_path("check_interval"), "must be >= 1");
    check(d.fused_max_rows >= 0, r.key_path("fused_max_rows"), "must be >= 0");
    check(d.rigid_coarse_max_rows >= 1, r.key_path("rigid_coarse_max_rows"), "must be >= 1");
    return d;
}

MaterialModel parse_model(const std::string& s, const std::string& path)
{
    check_enum(s, {"SNH", "NH", "COR", "cloth_stvk", "cloth_cor", "cloth_snh"}, path);
    if (s == "SNH") return MaterialModel::SNH;
    if (s == "NH") return MaterialModel::NH;
    if (s == "COR") return MaterialModel::COR;
    if (s == "cloth_stvk") return MaterialModel::ClothStVK;
    if (s == "cloth_cor") return MaterialModel::ClothCOR;
    return MaterialModel::ClothSNH;
}

const char* model_name(MaterialModel m)
{
    switch (m) {
        case MaterialModel::SNH: return "SNH";
        case MaterialModel::NH: return "NH";
        case MaterialModel::COR: return "COR";
        case MaterialModel::ClothStVK: return "cloth_stvk";
        case MaterialModel::ClothCOR: return "cloth_cor";
        case MaterialModel::ClothSNH: return "cloth_snh";
    }
    return "SNH";
}

MaterialDesc parse_material(const std::string& name, Reader r)
{
    MaterialDesc d;
    d.name = name;
    d.model = parse_model(r.string("model", model_name(d.model)), r.key_path("model"));
    d.E = r.number("E", d.E);
    d.nu = r.number("nu", d.nu);
    d.density = r.number("density", d.density);
    d.snh_lambda_reparam = r.boolean("snh_lambda_reparam", d.snh_lambda_reparam);
    if (const json* mm = r.get("mu_mem")) d.mu_mem = as_number(*mm, r.key_path("mu_mem"));
    d.thickness = r.number("thickness", d.thickness);
    d.k_bend = r.number("k_bend", d.k_bend);
    d.bending_hessian = r.string("bending_hessian", d.bending_hessian);
    r.finish();
    check(!name.empty(), r.path(), "material name must not be empty");
    check(d.E > 0, r.key_path("E"), "must be > 0");
    check(d.nu > -1 && d.nu < 0.5, r.key_path("nu"), "must be in (-1, 0.5)");
    check(d.density > 0, r.key_path("density"), "must be > 0");
    check(d.thickness > 0, r.key_path("thickness"), "must be > 0");
    check(d.k_bend >= 0, r.key_path("k_bend"), "must be >= 0");
    check_enum(d.bending_hessian, {"gauss_newton", "full"}, r.key_path("bending_hessian"));
    // MS §3.7 specifies both; only the Gauss-Newton rank-one form is implemented (M11). Refuse
    // the other rather than run it as Gauss-Newton silently.
    check(d.bending_hessian == "gauss_newton", r.key_path("bending_hessian"),
          "\"full\" is not implemented; only \"gauss_newton\" is");
    if (is_cloth_model(d.model)) {
        // MS §3.5: mu_mem is [Z25]'s (and OGC's tri_ke) MEMBRANE stiffness, an areal modulus in
        // N/m, i.e. mu * t; the element energy is t A Psi(F) with Psi in mu, so mu = mu_mem / t.
        // Reading mu_mem as a Pa modulus makes the sheet 1000x too soft at t = 1 mm, and the AL
        // penalty (0.1 x the diagonal) then cannot separate a folded sheet: that is how the
        // twisting cloth stalled before this was settled. Without mu_mem, E and nu apply.
        d.mu_lame = d.mu_mem ? (*d.mu_mem / d.thickness) : d.E / (2.0 * (1.0 + d.nu));
        check(d.mu_lame > 0, r.key_path("mu_mem"), "must be > 0");
        d.lambda_lame = 2.0 * d.mu_lame * d.nu / (1.0 - d.nu);
        d.lambda_hat = (d.model == MaterialModel::ClothSNH && d.snh_lambda_reparam) ? d.lambda_lame + d.mu_lame
                                                                                     : d.lambda_lame;
        d.alpha_hat = d.lambda_hat > 0 ? 1.0 + d.mu_lame / d.lambda_hat : 1.0;
        return d;
    }
    check(!d.mu_mem, r.key_path("mu_mem"), "is a cloth parameter; tet materials take E");
    // Lamé parameters (MS §3.2) and the SNH re-parameterization.
    d.mu_lame = d.E / (2.0 * (1.0 + d.nu));
    d.lambda_lame = d.E * d.nu / ((1.0 + d.nu) * (1.0 - 2.0 * d.nu));
    d.lambda_hat = (d.model == MaterialModel::SNH && d.snh_lambda_reparam) ? d.lambda_lame + d.mu_lame : d.lambda_lame;
    d.alpha_hat = d.lambda_hat > 0 ? 1.0 + d.mu_lame / d.lambda_hat : 1.0;
    return d;
}

PlacementDesc parse_placement(Reader& r)
{
    // Placement keys are flat on the body/collider object (§5.3, coupled_solver convention).
    PlacementDesc p;
    p.move_to_origin = r.boolean("move_to_origin", p.move_to_origin);
    p.scale = r.vec3("scale", p.scale);
    p.rotation_deg = r.vec3("rotation", p.rotation_deg);
    p.translation = r.vec3("translation", p.translation);
    for (int i = 0; i < 3; ++i) check(p.scale[i] != 0, r.key_path("scale"), "components must be non-zero");
    return p;
}

MotionDesc::Type parse_motion_type(const std::string& s, const std::string& path)
{
    check_enum(s, {"rotation", "translation", "scale", "keyframes"}, path);
    if (s == "rotation") return MotionDesc::Type::Rotation;
    if (s == "translation") return MotionDesc::Type::Translation;
    if (s == "scale") return MotionDesc::Type::Scale;
    return MotionDesc::Type::Keyframes;
}

const char* motion_type_name(MotionDesc::Type t)
{
    switch (t) {
        case MotionDesc::Type::None: return "none";
        case MotionDesc::Type::Rotation: return "rotation";
        case MotionDesc::Type::Translation: return "translation";
        case MotionDesc::Type::Scale: return "scale";
        case MotionDesc::Type::Keyframes: return "keyframes";
    }
    return "none";
}

// `v` is a non-null motion object.
MotionDesc parse_motion(const json& v, const std::string& path, const std::string& base_dir)
{
    Reader r(v, path);
    MotionDesc m;
    m.type = parse_motion_type(as_string(r.require("type"), r.key_path("type")), r.key_path("type"));
    m.axis = r.vec3("axis", m.axis);
    if (const json* c = r.get("center")) m.center = as_vec3(*c, r.key_path("center"));
    m.deg_per_second = r.number("deg_per_second", m.deg_per_second);
    m.velocity = r.vec3("velocity", m.velocity);
    m.rate = r.number("rate", m.rate);
    m.start_frame = r.integer("start_frame", m.start_frame);
    m.end_frame = r.integer("end_frame", m.end_frame);
    m.ramp_time = r.number("ramp_time", m.ramp_time);
    m.substeps = r.integer("substeps", m.substeps);
    const json* file = r.get("file");
    r.finish();

    check(m.start_frame >= 0, r.key_path("start_frame"), "must be >= 0");
    check(m.end_frame >= -1, r.key_path("end_frame"), "must be >= -1 (-1 = never stop)");
    check(m.end_frame < 0 || m.end_frame >= m.start_frame, r.key_path("end_frame"), "must be >= start_frame");
    check(m.substeps >= 1, r.key_path("substeps"), "must be >= 1");
    check(m.ramp_time >= 0, r.key_path("ramp_time"), "must be >= 0");
    if (m.type == MotionDesc::Type::Keyframes) check(m.ramp_time == 0, r.key_path("ramp_time"), "applies to scripted motions only");
    if (m.type == MotionDesc::Type::Rotation) check(m.axis.norm() > 0, r.key_path("axis"), "must be non-zero");
    if (m.type == MotionDesc::Type::Keyframes) {
        check(file != nullptr, r.key_path("file"), "required for keyframe motion");
        m.keyframe_file = resolve_file(as_string(*file, r.key_path("file")), base_dir, r.key_path("file"));
    } else {
        check(file == nullptr, r.key_path("file"), "only valid for keyframe motion");
    }
    return m;
}

SelectionDesc parse_selection(const json* v, const std::string& path)
{
    SelectionDesc s;
    if (!v) return s;  // "all"
    if (v->is_string()) {
        check_enum(v->get<std::string>(), {"all"}, path);
        s.mode = SelectionDesc::Mode::All;
    } else if (v->is_array()) {
        s.mode = SelectionDesc::Mode::Indices;
        for (size_t i = 0; i < v->size(); ++i) {
            const int idx = as_int((*v)[i], indexed(path, i));
            check(idx >= 0, indexed(path, i), "vertex index must be >= 0");
            s.indices.push_back(idx);
        }
    } else if (v->is_object()) {
        s.mode = SelectionDesc::Mode::Bbox;
        Reader r(*v, path);
        s.bbox_min = r.vec3("bbox_min", s.bbox_min);
        s.bbox_max = r.vec3("bbox_max", s.bbox_max);
        r.finish();
        for (int i = 0; i < 3; ++i)
            check(s.bbox_min[i] <= s.bbox_max[i], path, "bbox_min must be <= bbox_max componentwise");
    } else {
        fail(path, "expected \"all\", an index array or {bbox_min, bbox_max}");
    }
    return s;
}

DirichletDesc parse_dirichlet(const json& v, const std::string& path, const std::string& base_dir)
{
    Reader r(v, path);
    DirichletDesc d;
    d.select = parse_selection(r.get("select"), r.key_path("select"));
    if (const json* m = r.get("motion")) d.motion = parse_motion(*m, r.key_path("motion"), base_dir);
    d.release_frame = r.integer("release_frame", d.release_frame);
    check(d.release_frame >= -1, r.key_path("release_frame"), "must be -1 (never) or a frame index >= 0");
    r.finish();
    return d;
}

BodyDesc parse_body(const json& v, const std::string& path, const std::string& base_dir,
                    const std::map<std::string, MaterialDesc>& materials)
{
    Reader r(v, path);
    BodyDesc b;
    b.name = as_string(r.require("name"), r.key_path("name"));
    b.type = r.string("type", b.type);
    b.file = resolve_file(as_string(r.require("file"), r.key_path("file")), base_dir, r.key_path("file"));
    b.material = as_string(r.require("material"), r.key_path("material"));
    b.placement = parse_placement(r);
    b.initial_velocity = r.vec3("initial_velocity", b.initial_velocity);
    b.thickness = r.number("thickness", b.thickness);
    b.friction = r.number("friction", b.friction);
    if (const json* sc = r.get("self_collision")) b.self_collision = as_bool(*sc, r.key_path("self_collision"));
    if (const json* d = r.get("dirichlet")) {
        const json& arr = *d;
        check_array(arr, r.key_path("dirichlet"));
        for (size_t i = 0; i < arr.size(); ++i)
            b.dirichlet.push_back(parse_dirichlet(arr[i], indexed(r.key_path("dirichlet"), i), base_dir));
    }
    b.collision_group = r.integer("collision_group", b.collision_group);
    r.finish();

    check(!b.name.empty(), r.key_path("name"), "must not be empty");
    check_enum(b.type, {"tet", "cloth"}, r.key_path("type"));
    check(materials.count(b.material) > 0, r.key_path("material"), "unknown material '" + b.material + "'");
    check(b.thickness >= 0, r.key_path("thickness"), "must be >= 0");
    check(b.friction >= 0, r.key_path("friction"), "must be >= 0");
    // No spec section defines collision groups and nothing downstream reads the value (the
    // contact table is built from primitive kind, per-body self_collision and
    // contact_table.exclude). Refuse anything but the default rather than accept and ignore it.
    check(b.collision_group == -1, r.key_path("collision_group"),
          "is not implemented; use contact_table.exclude to switch pairs of bodies off");
    return b;
}

ColliderDesc parse_collider(const json& v, const std::string& path, const std::string& base_dir)
{
    Reader r(v, path);
    ColliderDesc c;
    c.name = as_string(r.require("name"), r.key_path("name"));
    c.type = r.string("type", c.type);
    c.file = resolve_file(as_string(r.require("file"), r.key_path("file")), base_dir, r.key_path("file"));
    c.placement = parse_placement(r);
    c.thickness = r.number("thickness", c.thickness);
    c.friction = r.number("friction", c.friction);
    if (const json* m = r.get("motion")) c.motion = parse_motion(*m, r.key_path("motion"), base_dir);
    r.finish();

    check(!c.name.empty(), r.key_path("name"), "must not be empty");
    check_enum(c.type, {"tri"}, r.key_path("type"));
    check(c.thickness >= 0, r.key_path("thickness"), "must be >= 0");
    check(c.friction >= 0, r.key_path("friction"), "must be >= 0");
    return c;
}

SphereDesc parse_sphere(const json& v, const std::string& path)
{
    Reader r(v, path);
    SphereDesc q;
    q.center = r.vec3("center", q.center);
    q.radius = r.number("radius", q.radius);
    q.inverted = r.boolean("inverted", q.inverted);
    q.friction = r.number("friction", q.friction);
    if (const json* kf = r.get("radius_keyframes")) {
        check_array(*kf, r.key_path("radius_keyframes"));
        double t_prev = -1.0;
        for (size_t i = 0; i < kf->size(); ++i) {
            const json& e = (*kf)[i];
            const std::string where = r.key_path("radius_keyframes") + "[" + std::to_string(i) + "]";
            check(e.is_array() && e.size() == 2 && e[0].is_number() && e[1].is_number(), where,
                  "must be [time, radius]");
            const double t = e[0].get<double>(), rad = e[1].get<double>();
            check(t >= 0.0 && t > t_prev, where, "times must be >= 0 and strictly increasing");
            check(rad > 0.0, where, "radius must be > 0");
            q.radius_keyframes.emplace_back(t, rad);
            t_prev = t;
        }
        if (!q.radius_keyframes.empty()) q.radius = q.radius_keyframes.front().second;
    }
    r.finish();
    check(q.radius > 0, r.key_path("radius"), "must be > 0");
    check(q.friction >= 0, r.key_path("friction"), "must be >= 0");
    return q;
}

PlaneDesc parse_plane(const json& v, const std::string& path)
{
    Reader r(v, path);
    PlaneDesc p;
    p.point = r.vec3("point", p.point);
    p.normal = r.vec3("normal", p.normal);
    p.friction = r.number("friction", p.friction);
    r.finish();
    check(p.normal.norm() > 0, r.key_path("normal"), "must be non-zero");
    check(p.friction >= 0, r.key_path("friction"), "must be >= 0");
    return p;
}

ContactTableDesc parse_contact_table(Reader r, const std::set<std::string>& names)
{
    ContactTableDesc t;
    auto check_name = [&](const std::string& n, const std::string& path) {
        check(names.count(n) > 0, path, "unknown body or collider '" + n + "'");
    };
    if (const json* ex = r.get("exclude")) {
        const json& arr = *ex;
        check_array(arr, r.key_path("exclude"));
        for (size_t i = 0; i < arr.size(); ++i) {
            const std::string p = indexed(r.key_path("exclude"), i);
            const json& e = arr[i];
            check(e.is_array() && e.size() == 2, p, "expected [name_a, name_b]");
            std::pair<std::string, std::string> pr(as_string(e[0], indexed(p, 0)), as_string(e[1], indexed(p, 1)));
            check_name(pr.first, indexed(p, 0));
            check_name(pr.second, indexed(p, 1));
            t.exclude.push_back(std::move(pr));
        }
    }
    if (const json* fr = r.get("friction")) {
        const json& arr = *fr;
        check_array(arr, r.key_path("friction"));
        for (size_t i = 0; i < arr.size(); ++i) {
            const std::string p = indexed(r.key_path("friction"), i);
            const json& e = arr[i];
            check(e.is_array() && e.size() == 3, p, "expected [name_a, name_b, mu]");
            ContactTableDesc::Friction f;
            f.a = as_string(e[0], indexed(p, 0));
            f.b = as_string(e[1], indexed(p, 1));
            f.mu = as_number(e[2], indexed(p, 2));
            check_name(f.a, indexed(p, 0));
            check_name(f.b, indexed(p, 1));
            check(f.mu >= 0, indexed(p, 2), "must be >= 0");
            t.friction.push_back(std::move(f));
        }
    }
    r.finish();
    return t;
}

LoggingDesc parse_logging(Reader r)
{
    LoggingDesc d;
    d.level = r.string("level", d.level);
    d.stats_csv = r.string("stats_csv", d.stats_csv);
    d.timing = r.boolean("timing", d.timing);
    d.debug_check_penetration = r.boolean("debug_check_penetration", d.debug_check_penetration);
    r.finish();
    check_enum(d.level, {"trace", "debug", "info", "warn", "warning", "error", "critical", "off"}, r.key_path("level"));
    return d;
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------
json to_json(const Vec3& v)
{
    return json::array({v.x(), v.y(), v.z()});
}

json placement_keys(const PlacementDesc& p, json& j)
{
    j["move_to_origin"] = p.move_to_origin;
    j["scale"] = to_json(p.scale);
    j["rotation"] = to_json(p.rotation_deg);
    j["translation"] = to_json(p.translation);
    return j;
}

json motion_to_json(const MotionDesc& m)
{
    if (m.type == MotionDesc::Type::None) return nullptr;
    json j;
    j["type"] = motion_type_name(m.type);
    j["axis"] = to_json(m.axis);
    j["center"] = m.center ? to_json(*m.center) : json(nullptr);
    j["deg_per_second"] = m.deg_per_second;
    j["velocity"] = to_json(m.velocity);
    j["rate"] = m.rate;
    j["start_frame"] = m.start_frame;
    j["end_frame"] = m.end_frame;
    j["ramp_time"] = m.ramp_time;
    j["substeps"] = m.substeps;
    if (m.type == MotionDesc::Type::Keyframes) j["file"] = m.keyframe_file;
    return j;
}

json selection_to_json(const SelectionDesc& s)
{
    switch (s.mode) {
        case SelectionDesc::Mode::All: return "all";
        case SelectionDesc::Mode::Indices: return json(s.indices);
        case SelectionDesc::Mode::Bbox: {
            json j;
            j["bbox_min"] = to_json(s.bbox_min);
            j["bbox_max"] = to_json(s.bbox_max);
            return j;
        }
    }
    return "all";
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
SceneDesc parse_scene_desc(const json& j, const std::string& base_dir)
{
    Reader root(j, "");
    SceneDesc d;
    d.base_dir = base_dir;
    d.simulation = parse_simulation(root.child("simulation"));
    d.contact = parse_contact(root.child("contact"));
    d.newton = parse_newton(root.child("newton"));
    d.linear_solver = parse_linear_solver(root.child("linear_solver"));

    if (const json* mats = root.get("materials")) {
        Reader mr(*mats, "materials");
        for (auto it = mats->begin(); it != mats->end(); ++it) {
            if (!it.key().empty() && it.key()[0] == '_') continue;  // comment key
            const json* m = mr.get(it.key());
            static const json empty = json::object();
            d.materials.emplace(it.key(), parse_material(it.key(), Reader(m ? *m : empty, mr.key_path(it.key()))));
        }
        mr.finish();
    }

    std::set<std::string> names;
    auto unique_name = [&](const std::string& n, const std::string& path) {
        check(names.insert(n).second, path, "duplicate body/collider name '" + n + "'");
    };
    if (const json* bodies = root.get("bodies")) {
        const json& arr = *bodies;
        check_array(arr, "bodies");
        for (size_t i = 0; i < arr.size(); ++i) {
            d.bodies.push_back(parse_body(arr[i], indexed("bodies", i), base_dir, d.materials));
            unique_name(d.bodies.back().name, join(indexed("bodies", i), "name"));
        }
    }
    if (const json* colliders = root.get("colliders")) {
        const json& arr = *colliders;
        check_array(arr, "colliders");
        for (size_t i = 0; i < arr.size(); ++i) {
            d.colliders.push_back(parse_collider(arr[i], indexed("colliders", i), base_dir));
            unique_name(d.colliders.back().name, join(indexed("colliders", i), "name"));
        }
    }
    if (const json* planes = root.get("planes")) {
        const json& arr = *planes;
        check_array(arr, "planes");
        for (size_t i = 0; i < arr.size(); ++i) d.planes.push_back(parse_plane(arr[i], indexed("planes", i)));
    }
    if (const json* spheres = root.get("spheres")) {
        const json& arr = *spheres;
        check_array(arr, "spheres");
        for (size_t i = 0; i < arr.size(); ++i) d.spheres.push_back(parse_sphere(arr[i], indexed("spheres", i)));
    }
    d.contact_table = parse_contact_table(root.child("contact_table"), names);
    d.logging = parse_logging(root.child("logging"));
    root.finish();
    return d;
}

SceneDesc load_scene_desc(const std::string& json_path)
{
    std::ifstream in(json_path);
    if (!in) throw std::runtime_error("load_scene_desc: cannot open '" + json_path + "'");
    json j;
    try {
        j = json::parse(in, nullptr, true, true);  // allow comments
    } catch (const json::parse_error& e) {
        throw std::runtime_error("load_scene_desc: " + json_path + ": " + e.what());
    }
    std::string base_dir = fs::path(json_path).parent_path().string();
    if (base_dir.empty()) base_dir = ".";
    return parse_scene_desc(j, base_dir);
}

json scene_desc_to_json(const SceneDesc& d)
{
    json j;
    {
        json& s = j["simulation"];
        s["dt"] = d.simulation.dt;
        s["frames"] = d.simulation.frames;
        s["gravity"] = to_json(d.simulation.gravity);
        s["output_dir"] = d.simulation.output_dir;
        s["save_bgeo"] = d.simulation.save_bgeo;
        s["save_npy"] = d.simulation.save_npy;
        s["save_every"] = d.simulation.save_every;
        s["checkpoint_every"] = d.simulation.checkpoint_every;
        s["abort_after_capped_steps"] = d.simulation.abort_after_capped_steps;
        if (d.simulation.restart_step >= 0) {
            json& rs = s["restart"];
            rs["step"] = d.simulation.restart_step;
            rs["positions"] = d.simulation.restart_positions;
            if (d.simulation.restart_velocities.empty()) rs["velocities"] = nullptr;
            else rs["velocities"] = d.simulation.restart_velocities;
        } else {
            s["restart"] = nullptr;
        }
    }
    {
        const ContactDesc& c = d.contact;
        json& s = j["contact"];
        s["enable"] = c.enable;
        s["d_hat"] = c.d_hat;
        s["epsilon"] = c.epsilon;
        s["K_min"] = c.K_min;
        s["max_outer_iters"] = c.max_outer_iters;
        s["decay_factor"] = c.decay_factor;
        s["decay_remove_threshold"] = c.decay_remove_threshold;
        s["mu_mode"] = c.mu_mode;
        s["mu_scale"] = c.mu_scale;
        s["mu_fixed"] = c.mu_fixed ? json(*c.mu_fixed) : json(nullptr);
        s["mu_max"] = c.mu_max ? json(*c.mu_max) : json(nullptr);
        s["alpha_lower_bound"] = c.alpha_lower_bound;
        s["toi_tie_tolerance"] = c.toi_tie_tolerance;
        s["stall"] = {{"iters", c.stall.iters},
                      {"alpha", c.stall.alpha},
                      {"mu_factor", c.stall.mu_factor},
                      {"d_hat_factor", c.stall.d_hat_factor},
                      {"max_adaptations", c.stall.max_adaptations}};
        s["ccd"] = {{"s", c.ccd.s},
                    {"max_iter", c.ccd.max_iter},
                    {"float_screen", c.ccd.float_screen},
                    {"rebuild_quality_ratio", c.ccd.rebuild_quality_ratio}};
        s["inversion_free"] = c.inversion_free;
        s["self_collision_default"] = c.self_collision_default;
        s["exclude_one_ring_pairs"] = c.exclude_one_ring_pairs;
        s["drop_parallel_edge_pairs"] = c.drop_parallel_edge_pairs;
        s["toi_filter_domain"] = c.toi_filter_domain;
        s["friction"] = {{"enable", c.friction.enable},
                         {"eps_v", c.friction.eps_v},
                         {"normal_force", c.friction.normal_force}};
    }
    {
        json& s = j["newton"];
        s["inner_max_iters"] = d.newton.inner_max_iters;
        s["line_search"] = {{"max_halvings", d.newton.line_search.max_halvings},
                            {"energy_tolerance", d.newton.line_search.energy_tolerance},
                            {"batched", d.newton.line_search.batched}};
        s["early_accept_velocity_tol"] = d.newton.early_accept_velocity_tol;
        s["increment_velocity_tol"] = d.newton.increment_velocity_tol;
    }
    {
        json& s = j["linear_solver"];
        s["type"] = d.linear_solver.type;
        s["preconditioner"] = d.linear_solver.preconditioner;
        s["rel_tol"] = d.linear_solver.rel_tol;
        s["max_iters"] = d.linear_solver.max_iters;
        s["check_interval"] = d.linear_solver.check_interval;
        s["graph"] = d.linear_solver.graph;
        s["fused"] = d.linear_solver.fused;
        s["fused_max_rows"] = d.linear_solver.fused_max_rows;
        s["rigid_coarse"] = d.linear_solver.rigid_coarse;
        s["rigid_coarse_max_rows"] = d.linear_solver.rigid_coarse_max_rows;
    }
    {
        json mats = json::object();
        for (const auto& [name, m] : d.materials) {
            mats[name] = {{"model", model_name(m.model)},
                          {"E", m.E},
                          {"nu", m.nu},
                          {"density", m.density},
                          {"snh_lambda_reparam", m.snh_lambda_reparam},
                          {"mu_mem", m.mu_mem ? json(*m.mu_mem) : json(nullptr)},
                          {"thickness", m.thickness}, {"k_bend", m.k_bend},
                          {"bending_hessian", m.bending_hessian}};
        }
        j["materials"] = std::move(mats);
    }
    {
        json arr = json::array();
        for (const BodyDesc& b : d.bodies) {
            json bj;
            bj["name"] = b.name;
            bj["type"] = b.type;
            bj["file"] = b.file;
            bj["material"] = b.material;
            placement_keys(b.placement, bj);
            bj["initial_velocity"] = to_json(b.initial_velocity);
            bj["thickness"] = b.thickness;
            bj["friction"] = b.friction;
            bj["self_collision"] = b.self_collision ? json(*b.self_collision) : json(nullptr);
            json dir = json::array();
            for (const DirichletDesc& dd : b.dirichlet)
                dir.push_back({{"select", selection_to_json(dd.select)}, {"motion", motion_to_json(dd.motion)},
                               {"release_frame", dd.release_frame}});
            bj["dirichlet"] = std::move(dir);
            bj["collision_group"] = b.collision_group;
            arr.push_back(std::move(bj));
        }
        j["bodies"] = std::move(arr);
    }
    {
        json arr = json::array();
        for (const ColliderDesc& c : d.colliders) {
            json cj;
            cj["name"] = c.name;
            cj["type"] = c.type;
            cj["file"] = c.file;
            placement_keys(c.placement, cj);
            cj["thickness"] = c.thickness;
            cj["friction"] = c.friction;
            cj["motion"] = motion_to_json(c.motion);
            arr.push_back(std::move(cj));
        }
        j["colliders"] = std::move(arr);
    }
    {
        json arr = json::array();
        for (const PlaneDesc& p : d.planes)
            arr.push_back({{"point", to_json(p.point)}, {"normal", to_json(p.normal)}, {"friction", p.friction}});
        j["planes"] = std::move(arr);
    }
    {
        json arr = json::array();
        for (const SphereDesc& q : d.spheres) {
            json kf = json::array();
            for (const auto& [t, rad] : q.radius_keyframes) kf.push_back(json::array({t, rad}));
            arr.push_back({{"center", to_json(q.center)}, {"radius", q.radius}, {"inverted", q.inverted},
                           {"friction", q.friction}, {"radius_keyframes", std::move(kf)}});
        }
        j["spheres"] = std::move(arr);
    }
    {
        json ex = json::array();
        for (const auto& e : d.contact_table.exclude) ex.push_back(json::array({e.first, e.second}));
        json fr = json::array();
        for (const auto& f : d.contact_table.friction) fr.push_back(json::array({f.a, f.b, f.mu}));
        j["contact_table"] = {{"exclude", std::move(ex)}, {"friction", std::move(fr)}};
    }
    j["logging"] = {{"level", d.logging.level},
                    {"stats_csv", d.logging.stats_csv},
                    {"timing", d.logging.timing},
                    {"debug_check_penetration", d.logging.debug_check_penetration}};
    return j;
}

}  // namespace cs
