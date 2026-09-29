// SceneBuffers::upload and the per-step host<->device transfers (doc/al-ipc-implementation-spec.md
// §4, §5.1). Everything here is host-side packing plus cudaMemcpy through DeviceArray; there are
// no kernels. The scene is validated before the first upload so that an inconsistent Scene is
// reported with a message instead of surfacing later as a device fault.
#include "gpu/scene_buffers.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("SceneBuffers::upload: " + what); }

template <class Container>
void expect_size(const Container& c, std::size_t n, const char* name)
{
    if (c.size() != n) {
        fail(std::string(name) + " has " + std::to_string(c.size()) + " entries, expected " + std::to_string(n));
    }
}

void check_vertex(int v, int n_vertices, const char* where)
{
    if (v < 0 || v >= n_vertices) {
        fail(std::string(where) + ": vertex id " + std::to_string(v) + " outside [0, " + std::to_string(n_vertices) + ")");
    }
}

MaterialParams to_material_params(const MaterialDesc& m)
{
    MaterialParams p{};
    // The device enum fem::Model continues these values: kClothStVK = 3, kClothCOR = 4,
    // kClothSNH = 5. A model missing here silently became SNH (p.model = 0) -- for the cloth
    // models that meant tet SNH evaluated on the padded F, a pre-tensioned zero-rest-length
    // membrane -- so the default is a hard failure.
    switch (m.model) {
        case MaterialModel::SNH: p.model = 0; break;
        case MaterialModel::NH: p.model = 1; break;
        case MaterialModel::COR: p.model = 2; break;
        case MaterialModel::ClothStVK: p.model = 3; break;
        case MaterialModel::ClothCOR: p.model = 4; break;
        case MaterialModel::ClothSNH: p.model = 5; break;
        default: fail("to_material_params: unmapped material model");
    }
    p.mu = real(m.mu_lame);
    p.lambda = real(m.lambda_lame);
    p.lambda_hat = real(m.lambda_hat);
    p.alpha_hat = real(m.alpha_hat);
    return p;
}

// Per-primitive info and thickness for one primitive kind. `verts` holds k global vertex ids
// per primitive, contiguous; `prim_body` is the scene's per-primitive body array (may be empty,
// in which case the body of the first vertex is used).
void pack_primitives(const Scene& scene, const int* verts, int k, std::size_t n_prims, const std::vector<int>& prim_body,
                     const char* name, std::vector<PrimInfo>& infos, std::vector<real>& thickness)
{
    if (!prim_body.empty()) expect_size(prim_body, n_prims, name);
    infos.resize(n_prims);
    thickness.resize(n_prims);
    for (std::size_t i = 0; i < n_prims; ++i) {
        const int* vs = verts + std::size_t(k) * i;
        bool all_prescribed = true;
        double xi = 0.0;
        for (int j = 0; j < k; ++j) {
            check_vertex(vs[j], scene.n_vertices, name);
            all_prescribed = all_prescribed && scene.is_prescribed[vs[j]] != 0;
            xi = j == 0 ? scene.thickness[vs[j]] : std::max(xi, scene.thickness[vs[j]]);
        }
        const int body = prim_body.empty() ? scene.body_id[vs[0]] : prim_body[i];
        if (body < 0 || body >= int(scene.bodies.size())) {
            fail(std::string(name) + "[" + std::to_string(i) + "]: body " + std::to_string(body) + " outside [0, " +
                 std::to_string(scene.bodies.size()) + ")");
        }
        const BodyInfo& b = scene.bodies[body];
        if (b.contact_id < 0 || b.contact_id > 0xFFFF) fail("contact id " + std::to_string(b.contact_id) + " does not fit 16 bits");
        if (body > 0xFFF) fail("body id " + std::to_string(body) + " does not fit 12 bits");
        infos[i].packed = PrimInfo::pack(b.contact_id, body, all_prescribed, b.self_collision);
        thickness[i] = real(xi);
    }
}

}  // namespace

void SceneBuffers::upload(const Scene& scene)
{
    // ---- validate the host scene ----
    const int n = scene.n_vertices;
    if (n < 0 || scene.n_free < 0 || scene.n_free > scene.n_body_vertices || scene.n_body_vertices > n) {
        fail("inconsistent vertex counts (n_free " + std::to_string(scene.n_free) + ", n_body_vertices " +
             std::to_string(scene.n_body_vertices) + ", n_vertices " + std::to_string(n) + ")");
    }
    if (scene.x0.cols() != n || scene.x0.rows() != 3) fail("x0 must be 3 x n_vertices");
    if (scene.v0.cols() != n || scene.v0.rows() != 3) fail("v0 must be 3 x n_vertices");
    expect_size(scene.mass, std::size_t(n), "mass");
    expect_size(scene.thickness, std::size_t(n), "thickness");
    expect_size(scene.body_id, std::size_t(n), "body_id");
    expect_size(scene.contact_id, std::size_t(n), "contact_id");
    expect_size(scene.is_prescribed, std::size_t(n), "is_prescribed");
    for (int i = 0; i < n; ++i) {
        const bool prescribed_by_range = i >= scene.n_free;
        if ((scene.is_prescribed[i] != 0) != prescribed_by_range) {
            fail("is_prescribed[" + std::to_string(i) + "] disagrees with the index range (n_free = " +
                 std::to_string(scene.n_free) + ")");
        }
        if (scene.body_id[i] < 0 || scene.body_id[i] >= int(scene.bodies.size())) {
            fail("body_id[" + std::to_string(i) + "] = " + std::to_string(scene.body_id[i]) + " outside [0, " +
                 std::to_string(scene.bodies.size()) + ")");
        }
        if (scene.contact_id[i] < 0 || scene.contact_id[i] >= scene.contact_table.n) {
            fail("contact_id[" + std::to_string(i) + "] = " + std::to_string(scene.contact_id[i]) + " outside [0, " +
                 std::to_string(scene.contact_table.n) + ")");
        }
    }
    if (scene.n_tets < 0) fail("negative n_tets");
    const std::size_t nt = std::size_t(scene.n_tets);
    expect_size(scene.tets, nt, "tets");
    expect_size(scene.DmInv, 9 * nt, "DmInv");
    expect_size(scene.volume, nt, "volume");
    expect_size(scene.tet_material, nt, "tet_material");
    for (std::size_t t = 0; t < nt; ++t) {
        // slot 3 = -1 marks a shell triangle padded to the tet layout (math spec §3.6)
        for (int s = 0; s < 4; ++s)
            if (!(s == 3 && scene.tets[t][3] == -1)) check_vertex(scene.tets[t][s], n, "tets");
        if (scene.tet_material[t] < 0 || scene.tet_material[t] >= int(scene.materials.size())) {
            fail("tet_material[" + std::to_string(t) + "] = " + std::to_string(scene.tet_material[t]) + " outside [0, " +
                 std::to_string(scene.materials.size()) + ")");
        }
    }
    const int nc = scene.contact_table.n;
    if (nc < 0) fail("negative contact table size");
    expect_size(scene.contact_table.enabled, std::size_t(nc) * nc, "contact_table.enabled");
    expect_size(scene.contact_table.friction, std::size_t(nc) * nc, "contact_table.friction");
    expect_size(scene.plane_contact_id, scene.planes.size(), "plane_contact_id");
    for (std::size_t p = 0; p < scene.planes.size(); ++p) {
        if (scene.plane_contact_id[p] < 0 || scene.plane_contact_id[p] >= nc) {
            fail("plane_contact_id[" + std::to_string(p) + "] = " + std::to_string(scene.plane_contact_id[p]) +
                 " outside [0, " + std::to_string(nc) + ")");
        }
        if (scene.planes[p].normal.norm() <= 0.0) fail("plane " + std::to_string(p) + " has a zero normal");
    }
    expect_size(scene.sphere_contact_id, scene.spheres.size(), "sphere_contact_id");
    for (std::size_t q = 0; q < scene.spheres.size(); ++q) {
        if (scene.sphere_contact_id[q] < 0 || scene.sphere_contact_id[q] >= nc) {
            fail("sphere_contact_id[" + std::to_string(q) + "] = " + std::to_string(scene.sphere_contact_id[q]) +
                 " outside [0, " + std::to_string(nc) + ")");
        }
        if (scene.spheres[q].radius <= 0.0) fail("sphere " + std::to_string(q) + " has a non-positive radius");
    }
    if (scene.surf_edges.size() > 0 && scene.surf_edge_body.size() != 0) expect_size(scene.surf_edge_body, scene.surf_edges.size(), "surf_edge_body");

    // ---- sizes ----
    n_vertices = n;
    n_free = scene.n_free;
    n_body_vertices = scene.n_body_vertices;
    n_tets = scene.n_tets;
    n_surf_v = int(scene.surf_vertices.size());
    n_surf_e = int(scene.surf_edges.size());
    n_surf_t = int(scene.surf_tris.size());
    n_planes = int(scene.planes.size());
    n_spheres = int(scene.spheres.size());
    n_contact_ids = nc;
    n_materials = int(scene.materials.size());

    // ---- vertex state ----
    {
        std::vector<real3> hx(n), hv(n);
        std::vector<real> hm(n), hxi(n);
        std::vector<int> hb(n), hc(n);
        std::vector<unsigned char> hp(n);
        for (int i = 0; i < n; ++i) {
            hx[i] = to_real3(scene.x0.col(i));
            hv[i] = to_real3(scene.v0.col(i));
            hp[i] = scene.is_prescribed[i] ? 1 : 0;
            hm[i] = hp[i] ? real(0) : real(scene.mass[i]);
            hxi[i] = real(scene.thickness[i]);
            hb[i] = scene.body_id[i];
            hc[i] = scene.contact_id[i];
        }
        x_prev.upload(hx);
        x_anchor.upload(hx);
        x_hat.upload(hx);
        x_tilde.upload(hx);
        x_ls0.upload(hx);
        x_target.upload(hx);
        v.upload(hv);
        mass.upload(hm);
        thickness.upload(hxi);
        body_id.upload(hb);
        contact_id.upload(hc);
        is_prescribed.upload(hp);
    }

    // ---- tets, rest data, materials ----
    {
        std::vector<int4> ht(nt);
        std::vector<real> hdm(9 * nt), hvol(nt);
        for (std::size_t t = 0; t < nt; ++t) {
            ht[t] = make_int4(scene.tets[t][0], scene.tets[t][1], scene.tets[t][2], scene.tets[t][3]);
            for (int k = 0; k < 9; ++k) hdm[9 * t + k] = real(scene.DmInv[9 * t + k]);
            hvol[t] = real(scene.volume[t]);
        }
        tet.upload(ht);
        DmInv.upload(hdm);
        volume.upload(hvol);
        tet_material.upload(scene.tet_material);

        std::vector<MaterialParams> hmat(scene.materials.size());
        for (std::size_t m = 0; m < scene.materials.size(); ++m) hmat[m] = to_material_params(scene.materials[m]);
        materials.upload(hmat);
    }

    // ---- vertex -> incident tets CSR over the free vertices (counting sort; tets and slots
    //      are visited in order, so every row is sorted by tet, then slot) ----
    {
        std::vector<int> ptr(std::size_t(n_free) + 1, 0);
        for (std::size_t t = 0; t < nt; ++t) {
            for (int s = 0; s < 4; ++s) {
                const int vid = scene.tets[t][s];
                if (vid >= 0 && vid < n_free) ++ptr[std::size_t(vid) + 1];
            }
        }
        for (int i = 0; i < n_free; ++i) ptr[std::size_t(i) + 1] += ptr[i];
        std::vector<int> next(ptr.begin(), ptr.end() - 1);
        const std::size_t n_entries = std::size_t(ptr[n_free]);
        std::vector<int> ht(n_entries), hs(n_entries);
        for (std::size_t t = 0; t < nt; ++t) {
            for (int s = 0; s < 4; ++s) {
                const int vid = scene.tets[t][s];
                if (vid >= 0 && vid < n_free) {
                    const int k = next[vid]++;
                    ht[k] = int(t);
                    hs[k] = s;
                }
            }
        }
        vt_ptr.upload(ptr);
        vt_tet.upload(ht);
        vt_slot.upload(hs);
    }

    // ---- contact surface ----
    {
        std::vector<int> hv(scene.surf_vertices.begin(), scene.surf_vertices.end());
        std::vector<int2> he(scene.surf_edges.size());
        std::vector<int> edge_verts(2 * scene.surf_edges.size());
        for (std::size_t e = 0; e < scene.surf_edges.size(); ++e) {
            const int a = scene.surf_edges[e][0], b = scene.surf_edges[e][1];
            he[e] = make_int2(a < b ? a : b, a < b ? b : a);
            edge_verts[2 * e] = he[e].x;
            edge_verts[2 * e + 1] = he[e].y;
        }
        std::vector<int3> htr(scene.surf_tris.size());
        std::vector<int> tri_verts(3 * scene.surf_tris.size());
        for (std::size_t f = 0; f < scene.surf_tris.size(); ++f) {
            htr[f] = make_int3(scene.surf_tris[f][0], scene.surf_tris[f][1], scene.surf_tris[f][2]);
            for (int j = 0; j < 3; ++j) tri_verts[3 * f + j] = scene.surf_tris[f][j];
        }

        std::vector<PrimInfo> info;
        std::vector<real> xi;
        pack_primitives(scene, hv.data(), 1, hv.size(), scene.surf_vertex_body, "surf_vertices", info, xi);
        surf_vert.upload(hv);
        surf_vert_info.upload(info);
        surf_vert_thickness.upload(xi);

        pack_primitives(scene, edge_verts.data(), 2, he.size(), scene.surf_edge_body, "surf_edges", info, xi);
        surf_edge.upload(he);
        surf_edge_info.upload(info);
        surf_edge_thickness.upload(xi);

        pack_primitives(scene, tri_verts.data(), 3, htr.size(), scene.surf_tri_body, "surf_tris", info, xi);
        surf_tri.upload(htr);
        surf_tri_info.upload(info);
        surf_tri_thickness.upload(xi);
    }

    // ---- planes ----
    {
        std::vector<real3> ho(scene.planes.size()), hn(scene.planes.size());
        for (std::size_t p = 0; p < scene.planes.size(); ++p) {
            ho[p] = to_real3(scene.planes[p].point);
            hn[p] = to_real3(Vec3(scene.planes[p].normal.normalized()));
        }
        plane_o.upload(ho);
        plane_n.upload(hn);
        plane_contact_id.upload(scene.plane_contact_id);
    }

    // ---- spheres: radii start at the schedule's t = 0 value; the stepper updates them ----
    {
        const std::size_t ns = scene.spheres.size();
        std::vector<real3> hc(ns);
        std::vector<real> hr(ns);
        std::vector<unsigned char> hi(ns);
        for (std::size_t q = 0; q < ns; ++q) {
            hc[q] = to_real3(scene.spheres[q].center);
            hr[q] = real(scene.sphere_radius(int(q), 0.0));
            hi[q] = scene.spheres[q].inverted ? 1 : 0;
        }
        sphere_c.upload(hc);
        sphere_r0.upload(hr);
        sphere_r1.upload(hr);
        sphere_inverted.upload(hi);
        sphere_contact_id.upload(scene.sphere_contact_id);
    }

    // ---- contact table ----
    {
        table_enabled.upload(scene.contact_table.enabled);
        std::vector<real> hf(scene.contact_table.friction.size());
        for (std::size_t i = 0; i < hf.size(); ++i) hf[i] = real(scene.contact_table.friction[i]);
        table_friction.upload(hf);
    }

    // ---- scalars ----
    dt = real(scene.desc.simulation.dt);
    gravity = to_real3(scene.desc.simulation.gravity);
}

void SceneBuffers::download_positions(std::vector<real3>& x) const { x_anchor.download(x); }

void SceneBuffers::download_velocities(std::vector<real3>& vel) const { v.download(vel); }

void SceneBuffers::upload_targets(const std::vector<real3>& target)
{
    if (target.size() != std::size_t(n_vertices)) {
        throw std::runtime_error("SceneBuffers::upload_targets: " + std::to_string(target.size()) + " targets for " +
                                 std::to_string(n_vertices) + " vertices");
    }
    x_target.upload(target);
}

void SceneBuffers::upload_held(const std::vector<int>& mask)
{
    held.upload(mask);
}

void SceneBuffers::set_sphere_radii(const std::vector<real>& r0, const std::vector<real>& r1)
{
    if (int(r0.size()) != n_spheres || int(r1.size()) != n_spheres) fail("set_sphere_radii: size mismatch");
    if (n_spheres == 0) return;
    sphere_r0.upload(r0);
    sphere_r1.upload(r1);
}

}  // namespace cs
