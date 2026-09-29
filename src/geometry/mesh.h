#pragma once
// Host mesh containers and the operations the scene builder needs
// (doc/al-ipc-implementation-spec.md §5.1-5.2). Eigen, column-per-element storage
// (Matrix3Xd: one vertex per column) as in coupled_solver's libcwheels.
#include "scene/scene_desc.h"
#include <Eigen/Core>
#include <string>
#include <vector>

namespace cs {

struct TetMesh {
    Eigen::Matrix3Xd V;   // vertices (columns)
    Eigen::Matrix4Xi T;   // tets (columns), positively oriented after load (det > 0)
    // Boundary surface, filled by extract_surface():
    Eigen::Matrix3Xi F;   // boundary triangles, outward orientation
    Eigen::Matrix2Xi E;   // unique boundary edges (i < j)
    Eigen::VectorXi surf_vertices; // unique boundary vertex ids
    int n_vertices() const { return int(V.cols()); }
    int n_tets() const { return int(T.cols()); }
};

struct TriMesh {
    Eigen::Matrix3Xd V;
    Eigen::Matrix3Xi F;
    Eigen::Matrix2Xi E;   // unique edges (i < j), filled by compute_edges()
    int n_vertices() const { return int(V.cols()); }
    int n_faces() const { return int(F.cols()); }
};

// Loaders. Format by extension: .bgeo (HouGeoIO), .msh (gmsh 2.2 ASCII, tets = element type 4,
// triangles = type 2), .node/.ele (TetGen; pass the .node path), .obj (triangles; quads are
// split). Throw std::runtime_error on failure.
TetMesh load_tet_mesh(const std::string& path);
TriMesh load_tri_mesh(const std::string& path);

// Placement: optional recentering to the origin, then scale, rotation (Euler XYZ, degrees),
// translation — the coupled_solver convention.
void apply_placement(Eigen::Matrix3Xd& V, const PlacementDesc& placement);

// Fix tet orientation so every tet has positive signed volume (swap two vertices otherwise).
void orient_tets(TetMesh& mesh);

// Boundary faces (faces belonging to exactly one tet), oriented outward; unique edges and
// vertices of the boundary.
void extract_surface(TetMesh& mesh);

// Unique edges of a triangle mesh (i < j).
void compute_edges(TriMesh& mesh);

// Vertex selection for Dirichlet regions, evaluated on the placed positions
// (bbox fractions are relative to V's bounding box).
std::vector<int> select_vertices(const SelectionDesc& selection, const Eigen::Matrix3Xd& V);

// Lumped masses: rest volume of each tet times density, a quarter to each vertex.
std::vector<double> lumped_masses(const TetMesh& mesh, double density);

// Rest-shape data per tet: inverse of D_m (column-major 3x3) and rest volume.
void tet_rest_data(const TetMesh& mesh, std::vector<double>& DmInv, std::vector<double>& volume);

// Shells (math spec §3.5). Lumped masses of a triangle mesh: rho * thickness * area / 3 per vertex.
std::vector<double> shell_lumped_masses(const TriMesh& mesh, double density, double thickness);
// Rest-shape data per triangle in the PADDED tet layout the device kernels share: DmInv is the
// 2x2 inverse of the rest edge matrix in the triangle's own frame, stored in the upper-left of a
// column-major 3x3 whose third row and column are zero; volume is thickness * area. Appended to
// the given arrays.
void shell_rest_data(const TriMesh& mesh, double thickness, std::vector<double>& DmInv, std::vector<double>& volume);

}  // namespace cs
