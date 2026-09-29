#pragma once
// The assembled host scene: every body, collider and plane merged into global arrays with
// the numbering the device side expects (doc/al-ipc-implementation-spec.md §5.1):
//   [0, n_free)            free vertices of all bodies (the DOFs)
//   [n_free, n_body_verts) prescribed (Dirichlet) vertices of bodies
//   [n_body_verts, n_vertices) collider vertices (always prescribed)
// Element and surface indices are global vertex ids. All host geometry is double.
#include "scene/scene_desc.h"
#include "geometry/mesh.h"
#include <Eigen/Core>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace cs {

enum class BodyKind { Deformable, Collider };

struct BodyInfo {
    std::string name;
    BodyKind kind = BodyKind::Deformable;
    int contact_id = 0;          // row in the contact table (one per body/collider/plane)
    int vertex_begin = 0;        // global vertex range of this body (may be non-contiguous for
    int vertex_end = 0;          //   deformables: free part and prescribed part; see maps below)
    bool self_collision = true;
    double thickness = 0.0;
    double friction = 0.0;
    int material_id = -1;        // deformables only
    int collision_group = -1;
    // Local -> global vertex map (size = the mesh's vertex count).
    std::vector<int> local_to_global;
};

// Prescribed-vertex driver: evaluates targets for one Dirichlet region or one kinematic
// collider at a given frame (MS §12). Captured positions are the placed rest positions.
struct PrescribedRegion {
    int body = 0;                       // index into Scene::bodies
    std::vector<int> global_vertices;   // vertices driven by this region
    Eigen::Matrix3Xd captured;          // positions at t = 0 (one column per vertex)
    MotionDesc motion;
    Vec3 center = Vec3::Zero();         // resolved rotation/scale center
    // Keyframes (MotionDesc::Type::Keyframes): frames x n x 3, row-major per frame.
    std::vector<Eigen::Matrix3Xd> keyframes;
    // IS §5.x held free region: its vertices are free (below n_free) and projected onto the
    // targets while frame < release_frame; -1 = an ordinary prescribed region.
    int release_frame = -1;
    // Target positions at the END of step `frame` (time (frame+1)*dt), written into
    // `target` columns in the order of global_vertices. Returns false when the region is
    // static for this frame (targets equal current positions).
    bool evaluate(int frame, double dt, Eigen::Matrix3Xd& target) const;
};

struct ContactTable {
    int n = 0;                                    // number of contact ids
    std::vector<unsigned char> enabled;           // n x n, row-major
    std::vector<double> friction;                 // n x n, pairwise Coulomb coefficient
    bool pair_enabled(int a, int b) const { return enabled[a * n + b] != 0; }
    double pair_friction(int a, int b) const { return friction[a * n + b]; }
};

struct Scene {
    SceneDesc desc;

    // ---- vertices (global numbering) ----
    int n_vertices = 0;
    int n_free = 0;               // DOFs: [0, n_free)
    int n_body_vertices = 0;      // bodies' vertices (free + prescribed): [0, n_body_vertices)
    Eigen::Matrix3Xd x0;          // initial positions
    Eigen::Matrix3Xd v0;          // initial velocities
    std::vector<double> mass;     // lumped; 0 for prescribed vertices
    std::vector<double> thickness;
    std::vector<int> body_id;     // index into `bodies`
    std::vector<int> contact_id;  // row in contact_table
    std::vector<unsigned char> is_prescribed;

    // ---- tetrahedra (deformable bodies) ----
    int n_tets = 0;
    std::vector<std::array<int, 4>> tets;   // global ids, positive orientation
    std::vector<double> DmInv;              // 9 per tet, column-major
    std::vector<double> volume;             // rest volume per tet
    std::vector<int> tet_material;          // index into materials
    std::vector<MaterialDesc> materials;    // in material_id order

    // ---- contact surface (bodies' boundaries + collider meshes) ----
    std::vector<int> surf_vertices;             // global ids
    std::vector<std::array<int, 2>> surf_edges; // global ids, i < j
    std::vector<std::array<int, 3>> surf_tris;  // global ids, outward orientation
    std::vector<int> surf_vertex_body, surf_edge_body, surf_tri_body; // body index per primitive

    // ---- planes ----
    std::vector<PlaneDesc> planes;
    std::vector<int> plane_contact_id;

    // ---- analytic spheres (MS §4.1 PS, §12.3) ----
    std::vector<SphereDesc> spheres;
    std::vector<int> sphere_contact_id;
    // Radius of sphere i at time t: piecewise linear in its keyframes, held after the last.
    double sphere_radius(int i, double t) const;

    // ---- bodies, prescribed regions, contact table ----
    std::vector<BodyInfo> bodies;
    std::vector<PrescribedRegion> prescribed;   // Dirichlet regions + moving colliders
    ContactTable contact_table;

    // Which vertices are free DOFs of a NH body (need the inversion guard), per body.
    std::vector<unsigned char> body_needs_inversion_guard;

    // Bounding box diagonal of the initial scene (for relative tolerances / reporting).
    double bbox_diagonal = 0.0;

    // Evaluate all prescribed targets for step `frame`; `target` has n_vertices columns and
    // only prescribed columns are written. Returns true if any prescribed vertex moves.
    bool evaluate_targets(int frame, Eigen::Matrix3Xd& target) const;
    // Held free vertices of step `frame` (IS §5.x): held[v] = 1 for every free vertex of a region
    // whose release_frame is beyond this frame. Returns false when nothing is held.
    bool held_mask(int frame, std::vector<int>& held) const;
    bool has_held_regions() const;
};

// Build the scene from a parsed description: loads meshes, applies placements, extracts
// surfaces, resolves Dirichlet selections and motions, numbers vertices, builds the contact
// table. Throws std::runtime_error with a precise message on any inconsistency.
Scene build_scene(const SceneDesc& desc);

// Human-readable summary (counts per body, DOFs, surface sizes) for the log.
std::string scene_summary(const Scene& scene);

}  // namespace cs
