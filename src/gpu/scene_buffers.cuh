#pragma once
// Device mirror of cs::Scene (doc/al-ipc-implementation-spec.md §4). Plain SoA arrays sized
// once at upload; the contact stage owns its own capacity-managed buffers elsewhere.
// Vertex numbering is the scene's: [0, n_free) are the DOFs.
#include "core/typedef.cuh"
#include "core/device_buffer.cuh"
#include "scene/scene.h"
#include <vector>

namespace cs {

// Material parameters in the form the kernels use (MS §3.2). model: 0 = SNH, 1 = NH, 2 = COR.
struct MaterialParams {
    int model;
    real mu;         // Lamé mu
    real lambda;     // Lamé lambda (NH, COR)
    real lambda_hat; // SNH: reparameterized lambda (= lambda + mu when snh_lambda_reparam)
    real alpha_hat;  // SNH: 1 + mu / lambda_hat
};

// Packed per-surface-primitive info for leaf-side admissibility tests (§6.10 step 3):
// bits 0..15 contact id, bits 16..27 body id, bit 28 all-prescribed, bit 29 self-collision on.
struct PrimInfo {
    unsigned int packed;
    CUDA_INLINE_CALLABLE int contact_id() const { return int(packed & 0xFFFFu); }
    CUDA_INLINE_CALLABLE int body_id() const { return int((packed >> 16) & 0xFFFu); }
    CUDA_INLINE_CALLABLE bool all_prescribed() const { return (packed >> 28) & 1u; }
    CUDA_INLINE_CALLABLE bool self_collision() const { return (packed >> 29) & 1u; }
    static CUDA_INLINE_CALLABLE unsigned int pack(int contact_id, int body_id, bool all_prescribed, bool self_collision) {
        return (unsigned(contact_id) & 0xFFFFu) | ((unsigned(body_id) & 0xFFFu) << 16) |
               (all_prescribed ? 1u << 28 : 0u) | (self_collision ? 1u << 29 : 0u);
    }
};

struct SceneBuffers {
    // ---- sizes ----
    int n_vertices = 0;
    int n_free = 0;
    int n_body_vertices = 0;
    int n_tets = 0;
    int n_surf_v = 0, n_surf_e = 0, n_surf_t = 0;
    int n_planes = 0;
    int n_spheres = 0;
    int n_contact_ids = 0;
    int n_materials = 0;

    // ---- vertex state (size n_vertices, real3) ----
    DeviceArray<real3> x_prev;    // x^t
    DeviceArray<real3> x_anchor;  // x (intersection-free anchor)
    DeviceArray<real3> x_hat;     // x̂ (iterate; prescribed part = targets)
    DeviceArray<real3> x_tilde;   // inertia target (free part meaningful)
    DeviceArray<real3> x_ls0;     // line-search start point
    DeviceArray<real3> v;         // velocity
    DeviceArray<real3> x_target;  // per-step targets of prescribed vertices
    DeviceArray<int> held;        // n_free: 1 while a free vertex is held onto its target (IS §5.x); empty when unused
    DeviceArray<real> mass;       // lumped mass, 0 for prescribed
    DeviceArray<real> thickness;  // primitive thickness ξ per vertex
    DeviceArray<int> body_id;
    DeviceArray<int> contact_id;
    DeviceArray<unsigned char> is_prescribed;

    // ---- tets ----
    DeviceArray<int4> tet;            // global vertex ids
    DeviceArray<real> DmInv;          // 9 per tet, column-major
    DeviceArray<real> volume;         // rest volume
    DeviceArray<int> tet_material;    // index into materials
    DeviceArray<MaterialParams> materials;
    // vertex -> incident tets CSR over FREE vertices (gradient gather): for row v,
    // vt_tet[k] is the tet, vt_slot[k] the local vertex index (0..3) of v in it.
    DeviceArray<int> vt_ptr;          // n_free + 1
    DeviceArray<int> vt_tet;
    DeviceArray<int> vt_slot;

    // ---- contact surface ----
    DeviceArray<int> surf_vert;       // global ids
    DeviceArray<int2> surf_edge;      // global ids (x < y)
    DeviceArray<int3> surf_tri;       // global ids, outward
    DeviceArray<PrimInfo> surf_vert_info, surf_edge_info, surf_tri_info;
    DeviceArray<real> surf_vert_thickness, surf_edge_thickness, surf_tri_thickness; // ξ per primitive (max over vertices)

    // ---- planes ----
    DeviceArray<real3> plane_o;
    DeviceArray<real3> plane_n;       // unit
    DeviceArray<int> plane_contact_id;

    // ---- analytic spheres (MS §4.1 PS, §12.3) ----
    DeviceArray<real3> sphere_c;
    DeviceArray<real> sphere_r0;      // radius at the step's anchor (start)
    DeviceArray<real> sphere_r1;      // radius at the step's target (end); constraints use this
    DeviceArray<unsigned char> sphere_inverted;
    DeviceArray<int> sphere_contact_id;
    // Re-uploads the per-step radii (the stepper calls this before begin_step).
    void set_sphere_radii(const std::vector<real>& r0, const std::vector<real>& r1);

    // ---- contact table (n_contact_ids^2, row-major) ----
    DeviceArray<unsigned char> table_enabled;
    DeviceArray<real> table_friction;

    // ---- scalars ----
    real dt = real(0);
    real3 gravity = make_real3(real(0), real(0), real(0));

    // Upload everything from the host scene (initial state x0, v0). Also builds vt_* CSR and the
    // per-primitive infos. Throws on CUDA errors.
    void upload(const Scene& scene);

    // Download x_anchor (the exported state) into host vectors (n_vertices entries).
    void download_positions(std::vector<real3>& x) const;
    void download_velocities(std::vector<real3>& v) const;

    // Upload per-step prescribed targets (all n_vertices columns; only prescribed columns are
    // read on the device). Called once per step by the stepper after Scene::evaluate_targets.
    void upload_targets(const std::vector<real3>& target);
    void upload_held(const std::vector<int>& mask);   // size n_free
};

// Small host helpers shared by upload code and tests.
inline real3 to_real3(const Vec3& v) { return make_real3(real(v.x()), real(v.y()), real(v.z())); }

}  // namespace cs
