// Scene assembly (doc/al-ipc-implementation-spec.md §5.1) and prescribed-motion evaluation
// (math spec §12). Global numbering: free body vertices, then Dirichlet body vertices, then
// collider vertices. Contact ids: bodies, then colliders, then planes.
#include "scene/scene.h"
#include "core/log.h"

#include "geometry/mesh.h"

#include "../../ext/cnpy/cnpy.h"  // cnpy exports no include directory of its own

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs {
namespace {

constexpr double kPi = 3.14159265358979323846;

[[noreturn]] void fail(const std::string& msg)
{
    throw std::runtime_error("build_scene: " + msg);
}

std::vector<Eigen::Matrix3Xd> load_keyframes(const std::string& file, int n_expected, const std::string& where)
{
    cnpy::NpyArray arr;
    try {
        arr = cnpy::npy_load(file);
    } catch (const std::exception& e) {
        fail(where + ": cannot load keyframes '" + file + "': " + e.what());
    }
    if (arr.shape.size() != 3 || arr.shape[2] != 3)
        fail(where + ": keyframes '" + file + "' must have shape [frames, n, 3]");
    if (arr.fortran_order) fail(where + ": keyframes '" + file + "': Fortran-ordered arrays are not supported");
    const size_t frames = arr.shape[0], n = arr.shape[1];
    if (frames == 0) fail(where + ": keyframes '" + file + "' contain no frames");
    if (int(n) != n_expected)
        fail(where + ": keyframes '" + file + "' have " + std::to_string(n) + " vertices per frame, the region has " +
             std::to_string(n_expected));
    std::vector<Eigen::Matrix3Xd> out(frames);
    if (arr.word_size == 8) {
        const double* p = arr.data<double>();
        for (size_t f = 0; f < frames; ++f)
            out[f] = Eigen::Map<const Eigen::Matrix3Xd>(p + f * n * 3, 3, Eigen::Index(n));
    } else if (arr.word_size == 4) {
        const float* p = arr.data<float>();
        for (size_t f = 0; f < frames; ++f)
            out[f] = Eigen::Map<const Eigen::Matrix3Xf>(p + f * n * 3, 3, Eigen::Index(n)).cast<double>();
    } else {
        fail(where + ": keyframes '" + file + "': unsupported element size " + std::to_string(arr.word_size) +
             " (need float32 or float64)");
    }
    return out;
}

// Elapsed scripted time at the END of step `frame` (nominally (frame+1)*dt), with the frame
// index clamped to [start_frame, end_frame]: the motion has not started before start_frame and
// freezes after end_frame (end_frame < 0 = never stop).
double scripted_time(const MotionDesc& m, int frame, double dt)
{
    int f = frame;
    if (m.end_frame >= 0) f = std::min(f, m.end_frame);
    const double t = std::max(0, f - m.start_frame + 1) * dt;
    // IS §5.x ramp-up: constant acceleration to the scripted rate over ramp_time, then the rate itself
    if (m.ramp_time > 0.0) return t < m.ramp_time ? 0.5 * t * t / m.ramp_time : t - 0.5 * m.ramp_time;
    return t;
}

double scale_factor(const MotionDesc& m, double t)
{
    return std::max(1.0 + m.rate * t, 0.01);  // never collapse the region to a point
}

// Target positions of a region at the end of step `frame` (time (frame+1)*dt).
Eigen::Matrix3Xd region_target(const PrescribedRegion& r, int frame, double dt)
{
    const MotionDesc& m = r.motion;
    switch (m.type) {
        case MotionDesc::Type::None: return r.captured;
        case MotionDesc::Type::Keyframes: {
            const int nk = int(r.keyframes.size());
            if (nk == 0) return r.captured;
            const int s = std::max(1, m.substeps);
            const long long step = (long long)frame + 1;  // keyframe 0 is the initial state
            const long long k = step / s, rem = step % s;
            if (k >= nk - 1) return r.keyframes[size_t(nk - 1)];  // last frame held
            if (rem == 0) return r.keyframes[size_t(k)];
            const double w = double(rem) / double(s);  // linear blend inside a keyframe interval
            return r.keyframes[size_t(k)] + (r.keyframes[size_t(k + 1)] - r.keyframes[size_t(k)]) * w;
        }
        default: break;
    }
    const double t = scripted_time(m, frame, dt);
    Eigen::Matrix3Xd X = r.captured;
    if (t <= 0) return X;
    switch (m.type) {
        case MotionDesc::Type::Rotation: {
            const double angle = m.deg_per_second * kPi / 180.0 * t;
            const Eigen::Matrix3d R = Eigen::AngleAxisd(angle, m.axis.normalized()).toRotationMatrix();
            X = (R * (X.colwise() - r.center)).colwise() + r.center;
            break;
        }
        case MotionDesc::Type::Translation: X.colwise() += Vec3(m.velocity * t); break;
        case MotionDesc::Type::Scale: {
            const double s = scale_factor(m, t);
            X = (s * (X.colwise() - r.center)).colwise() + r.center;
            break;
        }
        default: break;
    }
    return X;
}

const char* kind_name(BodyKind k)
{
    return k == BodyKind::Deformable ? "body" : "collider";
}

}  // namespace

bool PrescribedRegion::evaluate(int frame, double dt, Eigen::Matrix3Xd& target) const
{
    target = region_target(*this, frame, dt);
    if (motion.type == MotionDesc::Type::None) return false;
    if (motion.type == MotionDesc::Type::Keyframes) {
        const Eigen::Matrix3Xd prev = frame > 0 ? region_target(*this, frame - 1, dt) : captured;
        return prev.cols() != target.cols() || (target.array() != prev.array()).any();
    }
    // Scripted motion: the region moves iff the elapsed time advanced (inside the active frame
    // window) and the motion parameter is non-zero.
    const double t = scripted_time(motion, frame, dt);
    const double t_prev = scripted_time(motion, frame - 1, dt);
    if (t == t_prev) return false;
    switch (motion.type) {
        case MotionDesc::Type::Rotation: return motion.deg_per_second != 0.0;
        case MotionDesc::Type::Translation: return motion.velocity.squaredNorm() > 0.0;
        case MotionDesc::Type::Scale: return scale_factor(motion, t) != scale_factor(motion, t_prev);
        default: return false;
    }
}

bool Scene::evaluate_targets(int frame, Eigen::Matrix3Xd& target) const
{
    if (target.cols() != n_vertices) target = x0;
    bool moved = false;
    Eigen::Matrix3Xd tmp;
    for (const PrescribedRegion& r : prescribed) {
        moved |= r.evaluate(frame, desc.simulation.dt, tmp);
        for (size_t i = 0; i < r.global_vertices.size(); ++i) target.col(r.global_vertices[i]) = tmp.col(Eigen::Index(i));
    }
    return moved;
}

bool Scene::has_held_regions() const
{
    for (const PrescribedRegion& r : prescribed)
        if (r.release_frame >= 0) return true;
    return false;
}

bool Scene::held_mask(int frame, std::vector<int>& held) const
{
    held.assign(size_t(std::max(n_free, 0)), 0);
    bool any = false;
    for (const PrescribedRegion& r : prescribed) {
        if (r.release_frame < 0 || frame >= r.release_frame) continue;
        for (int g : r.global_vertices)
            if (g >= 0 && g < n_free) { held[size_t(g)] = 1; any = true; }
    }
    return any;
}

Scene build_scene(const SceneDesc& desc)
{
    Scene s;
    s.desc = desc;

    // ---- materials, in material_id order ----
    std::map<std::string, int> material_id;
    for (const auto& [name, m] : desc.materials) {
        material_id[name] = int(s.materials.size());
        s.materials.push_back(m);
    }

    const int nb = int(desc.bodies.size());
    const int nc = int(desc.colliders.size());
    const int np = int(desc.planes.size());
    const int ns = int(desc.spheres.size());

    // ---- load and place meshes, resolve Dirichlet selections ----
    struct LoadedBody {
        TetMesh mesh;                           // tet bodies
        TriMesh tri;                            // cloth bodies (MS §3.5)
        bool cloth = false;
        std::vector<std::vector<int>> regions;  // local vertex ids per Dirichlet entry
        std::vector<unsigned char> prescribed;
        int n_vertices() const { return cloth ? tri.n_vertices() : mesh.n_vertices(); }
        const Eigen::Matrix3Xd& V() const { return cloth ? tri.V : mesh.V; }
    };
    std::vector<LoadedBody> bodies(static_cast<size_t>(nb));
    std::vector<TriMesh> colliders(static_cast<size_t>(nc));
    int n_body_vertices = 0, n_prescribed_body = 0, n_collider_vertices = 0;

    for (int b = 0; b < nb; ++b) {
        const BodyDesc& bd = desc.bodies[size_t(b)];
        const std::string where = "bodies[" + std::to_string(b) + "] '" + bd.name + "'";
        LoadedBody& L = bodies[size_t(b)];
        L.cloth = (bd.type == "cloth");
        {
            const auto mit = material_id.find(bd.material);
            if (mit == material_id.end()) fail(where + ": unknown material '" + bd.material + "'");
            const bool cloth_mat = is_cloth_model(s.materials[size_t(mit->second)].model);
            if (L.cloth != cloth_mat)
                fail(where + ": a " + bd.type + " body needs a " + (L.cloth ? "cloth" : "tet") +
                     " material, and '" + bd.material + "' is not one");
        }
        try {
            if (L.cloth) {
                L.tri = load_tri_mesh(bd.file);
                if (L.tri.n_faces() == 0) throw std::runtime_error("mesh has no triangles");
                apply_placement(L.tri.V, bd.placement);
                compute_edges(L.tri);
            } else {
                L.mesh = load_tet_mesh(bd.file);
                if (L.mesh.n_tets() == 0) throw std::runtime_error("mesh has no tetrahedra");
                apply_placement(L.mesh.V, bd.placement);
                orient_tets(L.mesh);
                extract_surface(L.mesh);
            }
        } catch (const std::exception& e) {
            fail(where + ": " + e.what());
        }
        const int n = L.n_vertices();
        L.prescribed.assign(size_t(n), 0);
        std::vector<int> owner(size_t(n), -1);
        for (size_t k = 0; k < bd.dirichlet.size(); ++k) {
            const std::string dwhere = where + ".dirichlet[" + std::to_string(k) + "]";
            std::vector<int> sel;
            try {
                sel = select_vertices(bd.dirichlet[k].select, L.V());
            } catch (const std::exception& e) {
                fail(dwhere + ": " + e.what());
            }
            if (sel.empty()) fail(dwhere + ": selection is empty");
            const bool held_free = bd.dirichlet[k].release_frame >= 0;   // IS §5.x: stays a free vertex
            for (int v : sel) {
                if (owner[size_t(v)] >= 0)
                    fail(dwhere + ": vertex " + std::to_string(v) + " is already selected by dirichlet[" +
                         std::to_string(owner[size_t(v)]) + "]");
                owner[size_t(v)] = int(k);
                if (!held_free) {
                    L.prescribed[size_t(v)] = 1;
                    ++n_prescribed_body;
                }
            }
            L.regions.push_back(std::move(sel));
        }
        n_body_vertices += n;
    }
    for (int c = 0; c < nc; ++c) {
        const ColliderDesc& cd = desc.colliders[size_t(c)];
        const std::string where = "colliders[" + std::to_string(c) + "] '" + cd.name + "'";
        TriMesh& M = colliders[size_t(c)];
        try {
            M = load_tri_mesh(cd.file);
            if (M.n_faces() == 0) throw std::runtime_error("mesh has no triangles");
            apply_placement(M.V, cd.placement);
            compute_edges(M);
        } catch (const std::exception& e) {
            fail(where + ": " + e.what());
        }
        n_collider_vertices += M.n_vertices();
    }

    // ---- global numbering ----
    s.n_body_vertices = n_body_vertices;
    s.n_free = n_body_vertices - n_prescribed_body;
    s.n_vertices = n_body_vertices + n_collider_vertices;
    int next_free = 0, next_prescribed = s.n_free, next_collider = s.n_body_vertices;
    s.bodies.resize(size_t(nb + nc));
    for (int b = 0; b < nb; ++b) {
        const BodyDesc& bd = desc.bodies[size_t(b)];
        const LoadedBody& L = bodies[size_t(b)];
        BodyInfo& bi = s.bodies[size_t(b)];
        bi.name = bd.name;
        bi.kind = BodyKind::Deformable;
        bi.contact_id = b;
        bi.self_collision = bd.self_collision.value_or(desc.contact.self_collision_default);
        bi.thickness = bd.thickness;
        bi.friction = bd.friction;
        {
            const auto mit = material_id.find(bd.material);
            if (mit == material_id.end())
                fail("bodies[" + std::to_string(b) + "] '" + bd.name + "': unknown material '" + bd.material + "'");
            bi.material_id = mit->second;
        }
        bi.collision_group = bd.collision_group;
        const int n = L.n_vertices();
        bi.local_to_global.resize(size_t(n));
        int lo = s.n_vertices, hi = -1;
        for (int v = 0; v < n; ++v) {
            const int g = L.prescribed[size_t(v)] ? next_prescribed++ : next_free++;
            bi.local_to_global[size_t(v)] = g;
            lo = std::min(lo, g);
            hi = std::max(hi, g);
        }
        bi.vertex_begin = n > 0 ? lo : 0;  // covering range; non-contiguous when prescribed vertices exist
        bi.vertex_end = n > 0 ? hi + 1 : 0;
    }
    for (int c = 0; c < nc; ++c) {
        const ColliderDesc& cd = desc.colliders[size_t(c)];
        const TriMesh& M = colliders[size_t(c)];
        BodyInfo& bi = s.bodies[size_t(nb + c)];
        bi.name = cd.name;
        bi.kind = BodyKind::Collider;
        bi.contact_id = nb + c;
        bi.self_collision = false;
        bi.thickness = cd.thickness;
        bi.friction = cd.friction;
        bi.material_id = -1;
        bi.collision_group = -1;
        const int n = M.n_vertices();
        bi.local_to_global.resize(size_t(n));
        bi.vertex_begin = next_collider;
        for (int v = 0; v < n; ++v) bi.local_to_global[size_t(v)] = next_collider++;
        bi.vertex_end = next_collider;
    }

    // ---- per-vertex arrays ----
    const int N = s.n_vertices;
    s.x0.resize(3, N);
    s.v0.setZero(3, N);
    s.mass.assign(size_t(N), 0.0);
    s.thickness.assign(size_t(N), 0.0);
    s.body_id.assign(size_t(N), -1);
    s.contact_id.assign(size_t(N), -1);
    s.is_prescribed.assign(size_t(N), 0);
    for (int b = 0; b < nb; ++b) {
        const BodyDesc& bd = desc.bodies[size_t(b)];
        const LoadedBody& L = bodies[size_t(b)];
        const BodyInfo& bi = s.bodies[size_t(b)];
        const MaterialDesc& mat = s.materials[size_t(bi.material_id)];
        const std::vector<double> masses =
            L.cloth ? shell_lumped_masses(L.tri, mat.density, mat.thickness) : lumped_masses(L.mesh, mat.density);
        // MS §4.1: a shell's contact primitives carry xi = t/2 unless the body overrides it.
        const double xi = (L.cloth && bd.thickness <= 0.0) ? 0.5 * mat.thickness : bd.thickness;
        for (int v = 0; v < L.n_vertices(); ++v) {
            const int g = bi.local_to_global[size_t(v)];
            s.x0.col(g) = L.V().col(v);
            if (!L.prescribed[size_t(v)]) {
                s.v0.col(g) = bd.initial_velocity;
                s.mass[size_t(g)] = masses[size_t(v)];
            }
            s.thickness[size_t(g)] = xi;
            s.body_id[size_t(g)] = b;
            s.contact_id[size_t(g)] = bi.contact_id;
            s.is_prescribed[size_t(g)] = L.prescribed[size_t(v)];
        }
    }
    for (int c = 0; c < nc; ++c) {
        const ColliderDesc& cd = desc.colliders[size_t(c)];
        const TriMesh& M = colliders[size_t(c)];
        const BodyInfo& bi = s.bodies[size_t(nb + c)];
        for (int v = 0; v < M.n_vertices(); ++v) {
            const int g = bi.local_to_global[size_t(v)];
            s.x0.col(g) = M.V.col(v);
            s.thickness[size_t(g)] = cd.thickness;
            s.body_id[size_t(g)] = nb + c;
            s.contact_id[size_t(g)] = bi.contact_id;
            s.is_prescribed[size_t(g)] = 1;
        }
    }

    // ---- tetrahedra ----
    for (int b = 0; b < nb; ++b) {
        const LoadedBody& L = bodies[size_t(b)];
        const BodyInfo& bi = s.bodies[size_t(b)];
        std::vector<double> DmInv, volume;
        try {
            if (L.cloth) shell_rest_data(L.tri, s.materials[size_t(bi.material_id)].thickness, DmInv, volume);
            else tet_rest_data(L.mesh, DmInv, volume);
        } catch (const std::exception& e) {
            fail("bodies[" + std::to_string(b) + "] '" + bi.name + "': " + e.what());
        }
        if (L.cloth) {
            // MS §3.6: a triangle is a three-vertex element in the tet layout, slot 3 = -1.
            for (int f = 0; f < L.tri.n_faces(); ++f) {
                std::array<int, 4> tet;
                for (int k = 0; k < 3; ++k) tet[size_t(k)] = bi.local_to_global[size_t(L.tri.F(k, f))];
                tet[3] = -1;
                s.tets.push_back(tet);
                s.tet_material.push_back(bi.material_id);
            }
            s.DmInv.insert(s.DmInv.end(), DmInv.begin(), DmInv.end());
            s.volume.insert(s.volume.end(), volume.begin(), volume.end());
            continue;
        }
        for (int t = 0; t < L.mesh.n_tets(); ++t) {
            std::array<int, 4> tet;
            for (int k = 0; k < 4; ++k) tet[size_t(k)] = bi.local_to_global[size_t(L.mesh.T(k, t))];
            s.tets.push_back(tet);
            s.tet_material.push_back(bi.material_id);
        }
        s.DmInv.insert(s.DmInv.end(), DmInv.begin(), DmInv.end());
        s.volume.insert(s.volume.end(), volume.begin(), volume.end());
    }
    s.n_tets = int(s.tets.size());

    // ---- contact surface ----
    auto add_surface = [&](int body, const std::vector<int>& l2g, const Eigen::VectorXi& verts,
                           const Eigen::Matrix2Xi& E, const Eigen::Matrix3Xi& F) {
        for (Eigen::Index i = 0; i < verts.size(); ++i) {
            s.surf_vertices.push_back(l2g[size_t(verts[i])]);
            s.surf_vertex_body.push_back(body);
        }
        for (Eigen::Index e = 0; e < E.cols(); ++e) {
            int a = l2g[size_t(E(0, e))], bb = l2g[size_t(E(1, e))];
            if (a > bb) std::swap(a, bb);
            s.surf_edges.push_back({a, bb});
            s.surf_edge_body.push_back(body);
        }
        for (Eigen::Index f = 0; f < F.cols(); ++f) {
            s.surf_tris.push_back({l2g[size_t(F(0, f))], l2g[size_t(F(1, f))], l2g[size_t(F(2, f))]});
            s.surf_tri_body.push_back(body);
        }
    };
    for (int b = 0; b < nb; ++b) {
        const LoadedBody& L = bodies[size_t(b)];
        if (L.cloth) {
            // MS §4.1: a shell exposes every vertex, edge and triangle it has
            Eigen::VectorXi all = Eigen::VectorXi::LinSpaced(L.tri.n_vertices(), 0, std::max(0, L.tri.n_vertices() - 1));
            if (L.tri.n_vertices() == 0) all.resize(0);
            add_surface(b, s.bodies[size_t(b)].local_to_global, all, L.tri.E, L.tri.F);
            continue;
        }
        add_surface(b, s.bodies[size_t(b)].local_to_global, L.mesh.surf_vertices, L.mesh.E, L.mesh.F);
    }
    for (int c = 0; c < nc; ++c) {
        const TriMesh& M = colliders[size_t(c)];
        Eigen::VectorXi all = Eigen::VectorXi::LinSpaced(M.n_vertices(), 0, std::max(0, M.n_vertices() - 1));
        if (M.n_vertices() == 0) all.resize(0);
        add_surface(nb + c, s.bodies[size_t(nb + c)].local_to_global, all, M.E, M.F);
    }

    // ---- planes ----
    for (int p = 0; p < np; ++p) {
        PlaneDesc q = desc.planes[size_t(p)];
        q.normal.normalize();
        s.planes.push_back(q);
        s.plane_contact_id.push_back(nb + nc + p);
    }

    // ---- spheres: contact ids follow the planes ----
    for (int q = 0; q < ns; ++q) {
        s.spheres.push_back(desc.spheres[size_t(q)]);
        s.sphere_contact_id.push_back(nb + nc + np + q);
    }

    // ---- contact table ----
    {
        const int n = nb + nc + np + ns;
        ContactTable& T = s.contact_table;
        T.n = n;
        T.enabled.assign(size_t(n) * size_t(n), 0);
        T.friction.assign(size_t(n) * size_t(n), 0.0);
        std::vector<double> mu(size_t(n), 0.0);
        std::vector<int> kind(size_t(n), 0);  // 0 body, 1 collider, 2 plane
        std::map<std::string, int> id_of;
        for (int b = 0; b < nb; ++b) {
            mu[size_t(b)] = desc.bodies[size_t(b)].friction;
            id_of[desc.bodies[size_t(b)].name] = b;
        }
        for (int c = 0; c < nc; ++c) {
            mu[size_t(nb + c)] = desc.colliders[size_t(c)].friction;
            kind[size_t(nb + c)] = 1;
            id_of[desc.colliders[size_t(c)].name] = nb + c;
        }
        for (int p = 0; p < np; ++p) {
            mu[size_t(nb + nc + p)] = desc.planes[size_t(p)].friction;
            kind[size_t(nb + nc + p)] = 2;
        }
        for (int q = 0; q < ns; ++q) {
            mu[size_t(nb + nc + np + q)] = desc.spheres[size_t(q)].friction;
            kind[size_t(nb + nc + np + q)] = 2;   // analytic, like a plane
        }
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                bool on;
                if (a == b) on = kind[size_t(a)] == 0 && s.bodies[size_t(a)].self_collision;  // self-contact
                else on = kind[size_t(a)] == 0 || kind[size_t(b)] == 0;  // at least one deformable
                T.enabled[size_t(a) * size_t(n) + size_t(b)] = on ? 1 : 0;
                T.friction[size_t(a) * size_t(n) + size_t(b)] = std::sqrt(mu[size_t(a)] * mu[size_t(b)]);
            }
        }
        auto lookup = [&](const std::string& name) {
            const auto it = id_of.find(name);
            if (it == id_of.end()) fail("contact_table: unknown body or collider '" + name + "'");
            return it->second;
        };
        for (const auto& [na, nbn] : desc.contact_table.exclude) {
            const int a = lookup(na), b = lookup(nbn);
            T.enabled[size_t(a) * size_t(n) + size_t(b)] = 0;
            T.enabled[size_t(b) * size_t(n) + size_t(a)] = 0;
        }
        for (const auto& f : desc.contact_table.friction) {
            const int a = lookup(f.a), b = lookup(f.b);
            T.friction[size_t(a) * size_t(n) + size_t(b)] = f.mu;
            T.friction[size_t(b) * size_t(n) + size_t(a)] = f.mu;
        }
    }

    // ---- prescribed regions: Dirichlet entries, then colliders ----
    auto make_region = [&](int body, const std::vector<int>& local, const std::vector<int>& l2g,
                           const Eigen::Matrix3Xd& V, const MotionDesc& motion, const std::string& where,
                           int release_frame = -1) {
        PrescribedRegion r;
        r.body = body;
        r.release_frame = release_frame;
        r.global_vertices.reserve(local.size());
        r.captured.resize(3, Eigen::Index(local.size()));
        for (size_t i = 0; i < local.size(); ++i) {
            r.global_vertices.push_back(l2g[size_t(local[i])]);
            r.captured.col(Eigen::Index(i)) = V.col(local[i]);
        }
        r.motion = motion;
        if (r.motion.type == MotionDesc::Type::Rotation) {
            if (r.motion.axis.norm() <= 0) fail(where + ": rotation axis must be non-zero");
            r.motion.axis.normalize();
        }
        r.center = motion.center ? *motion.center
                                 : (local.empty() ? Vec3::Zero() : Vec3(r.captured.rowwise().mean()));
        if (r.motion.type == MotionDesc::Type::Keyframes) {
            r.keyframes = load_keyframes(r.motion.keyframe_file, int(local.size()), where);
            // Keyframes past the array are held at the last frame (evaluate()); a short array is
            // almost always a scene-generation mistake, so say so once.
            const long long s_kf = std::max(1, r.motion.substeps);   // solver steps per keyframe (region_target: k = step / s)
            const long long need = (static_cast<long long>(desc.simulation.frames) + s_kf - 1) / s_kf + 1;
            if (static_cast<long long>(r.keyframes.size()) < need)
                log().warn("{}: keyframes '{}' hold {} of the {} keyframes that {} steps need; the last keyframe is held after that",
                           where, r.motion.keyframe_file, r.keyframes.size(), need, desc.simulation.frames);
        }
        s.prescribed.push_back(std::move(r));
    };
    for (int b = 0; b < nb; ++b) {
        const BodyDesc& bd = desc.bodies[size_t(b)];
        const LoadedBody& L = bodies[size_t(b)];
        for (size_t k = 0; k < L.regions.size(); ++k)
            make_region(b, L.regions[k], s.bodies[size_t(b)].local_to_global, L.V(), bd.dirichlet[k].motion,
                        "bodies[" + std::to_string(b) + "].dirichlet[" + std::to_string(k) + "]", bd.dirichlet[k].release_frame);
    }
    for (int c = 0; c < nc; ++c) {
        const TriMesh& M = colliders[size_t(c)];
        std::vector<int> all(size_t(M.n_vertices()));
        for (int v = 0; v < M.n_vertices(); ++v) all[size_t(v)] = v;
        make_region(nb + c, all, s.bodies[size_t(nb + c)].local_to_global, M.V, desc.colliders[size_t(c)].motion,
                    "colliders[" + std::to_string(c) + "].motion");
    }

    // ---- inversion guard (MS §10.4): NH bodies under "auto", all bodies under "on" ----
    s.body_needs_inversion_guard.assign(size_t(nb + nc), 0);
    for (int b = 0; b < nb; ++b) {
        const std::string& mode = desc.contact.inversion_free;
        bool guard = false;
        if (mode == "on") guard = true;
        else if (mode == "off") guard = false;
        else guard = s.materials[size_t(s.bodies[size_t(b)].material_id)].model == MaterialModel::NH;
        s.body_needs_inversion_guard[size_t(b)] = guard ? 1 : 0;
    }

    // ---- bounding box ----
    if (N > 0) {
        const Vec3 lo = s.x0.rowwise().minCoeff();
        const Vec3 hi = s.x0.rowwise().maxCoeff();
        s.bbox_diagonal = (hi - lo).norm();
    }
    return s;
}

std::string scene_summary(const Scene& scene)
{
    std::ostringstream os;
    os << "scene: " << scene.n_vertices << " vertices (" << scene.n_free << " free, "
       << (scene.n_body_vertices - scene.n_free) << " dirichlet, " << (scene.n_vertices - scene.n_body_vertices)
       << " collider), " << scene.n_tets << " tets, surface: " << scene.surf_vertices.size() << " vertices, "
       << scene.surf_edges.size() << " edges, " << scene.surf_tris.size() << " triangles; " << scene.planes.size()
       << " planes, " << scene.spheres.size() << " spheres, " << scene.prescribed.size() << " prescribed regions, " << scene.contact_table.n
       << " contact ids, bbox diagonal " << scene.bbox_diagonal << "\n";
    for (size_t b = 0; b < scene.bodies.size(); ++b) {
        const BodyInfo& bi = scene.bodies[b];
        int n_pres = 0;
        for (int g : bi.local_to_global) n_pres += scene.is_prescribed[size_t(g)];
        int n_tets = 0;
        for (size_t t = 0; t < scene.tets.size(); ++t)
            if (scene.body_id[size_t(scene.tets[t][0])] == int(b)) ++n_tets;
        int n_tris = 0;
        for (int tb : scene.surf_tri_body) n_tris += tb == int(b);
        os << "  [" << b << "] " << kind_name(bi.kind) << " '" << bi.name << "': " << bi.local_to_global.size()
           << " vertices (" << n_pres << " prescribed), " << n_tets << " tets, " << n_tris
           << " surface triangles, contact id " << bi.contact_id;
        if (bi.kind == BodyKind::Deformable) {
            const MaterialDesc& m = scene.materials[size_t(bi.material_id)];
            os << ", material '" << m.name << "' (E=" << m.E << ", nu=" << m.nu << ", rho=" << m.density << ")";
            if (scene.body_needs_inversion_guard[b]) os << ", inversion guard";
        }
        os << ", friction " << bi.friction << ", thickness " << bi.thickness << "\n";
    }
    return os.str();
}

double Scene::sphere_radius(int i, double t) const
{
    const SphereDesc& q = spheres[size_t(i)];
    const auto& kf = q.radius_keyframes;
    if (kf.empty()) return q.radius;
    if (t <= kf.front().first) return kf.front().second;
    for (size_t k = 1; k < kf.size(); ++k) {
        if (t <= kf[k].first) {
            const double t0 = kf[k - 1].first, t1 = kf[k].first;
            const double w = (t - t0) / (t1 - t0);
            return kf[k - 1].second + w * (kf[k].second - kf[k - 1].second);
        }
    }
    return kf.back().second;
}

}  // namespace cs
