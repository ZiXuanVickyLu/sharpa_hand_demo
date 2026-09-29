// Host mesh operations: placement, tet orientation, boundary-surface extraction, edges,
// vertex selection, lumped masses and rest-shape data (doc/al-ipc-implementation-spec.md §5.1).
#include "geometry/mesh.h"

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs {
namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

// Sorted unique (i < j) edges of a triangle list.
Eigen::Matrix2Xi unique_edges(const Eigen::Matrix3Xi& F)
{
    std::vector<std::array<int, 2>> edges;
    edges.reserve(size_t(3 * F.cols()));
    for (Eigen::Index f = 0; f < F.cols(); ++f) {
        for (int k = 0; k < 3; ++k) {
            int a = F(k, f), b = F((k + 1) % 3, f);
            if (a > b) std::swap(a, b);
            edges.push_back({a, b});
        }
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    Eigen::Matrix2Xi E;
    E.resize(2, Eigen::Index(edges.size()));
    for (size_t i = 0; i < edges.size(); ++i) {
        E(0, Eigen::Index(i)) = edges[i][0];
        E(1, Eigen::Index(i)) = edges[i][1];
    }
    return E;
}

double signed_volume6(const Eigen::Matrix3Xd& V, int a, int b, int c, int d)
{
    const Vec3 e1 = V.col(b) - V.col(a), e2 = V.col(c) - V.col(a), e3 = V.col(d) - V.col(a);
    return e1.dot(e2.cross(e3));
}

}  // namespace

void apply_placement(Eigen::Matrix3Xd& V, const PlacementDesc& placement)
{
    if (V.cols() == 0) return;
    if (placement.move_to_origin) {
        const Vec3 lo = V.rowwise().minCoeff();
        const Vec3 hi = V.rowwise().maxCoeff();
        V.colwise() -= Vec3(0.5 * (lo + hi));
    }
    for (int k = 0; k < 3; ++k) V.row(k) *= placement.scale[k];
    if (placement.rotation_deg.squaredNorm() > 0) {
        // Euler XYZ: rotate about X, then Y, then Z (fixed axes): R = Rz * Ry * Rx.
        const Vec3 r = placement.rotation_deg * kDegToRad;
        const Eigen::Matrix3d R = (Eigen::AngleAxisd(r.z(), Vec3::UnitZ()) * Eigen::AngleAxisd(r.y(), Vec3::UnitY()) *
                                   Eigen::AngleAxisd(r.x(), Vec3::UnitX()))
                                      .toRotationMatrix();
        V = R * V;
    }
    V.colwise() += placement.translation;
}

void orient_tets(TetMesh& mesh)
{
    for (Eigen::Index t = 0; t < mesh.T.cols(); ++t) {
        const double v6 = signed_volume6(mesh.V, mesh.T(0, t), mesh.T(1, t), mesh.T(2, t), mesh.T(3, t));
        if (v6 == 0.0)
            throw std::runtime_error("orient_tets: tet " + std::to_string(t) + " is degenerate (zero volume)");
        if (v6 < 0) std::swap(mesh.T(1, t), mesh.T(2, t));
    }
}

void extract_surface(TetMesh& mesh)
{
    struct Face {
        int a, b, c;  // sorted
        int opposite;
        bool operator<(const Face& o) const
        {
            return a != o.a ? a < o.a : (b != o.b ? b < o.b : c < o.c);
        }
        bool same(const Face& o) const { return a == o.a && b == o.b && c == o.c; }
    };
    static const int local[4][3] = {{1, 2, 3}, {0, 3, 2}, {0, 1, 3}, {0, 2, 1}};

    const Eigen::Index nt = mesh.T.cols();
    std::vector<Face> faces;
    faces.reserve(size_t(4 * nt));
    for (Eigen::Index t = 0; t < nt; ++t) {
        for (int f = 0; f < 4; ++f) {
            int i = mesh.T(local[f][0], t), j = mesh.T(local[f][1], t), k = mesh.T(local[f][2], t);
            if (i > j) std::swap(i, j);
            if (j > k) std::swap(j, k);
            if (i > j) std::swap(i, j);
            faces.push_back({i, j, k, mesh.T(f, t)});
        }
    }
    std::sort(faces.begin(), faces.end());

    std::vector<std::array<int, 3>> boundary;
    for (size_t s = 0; s < faces.size();) {
        size_t e = s + 1;
        while (e < faces.size() && faces[e].same(faces[s])) ++e;
        const size_t count = e - s;
        if (count == 1) {
            const Face& f = faces[s];
            int a = f.a, b = f.b, c = f.c;
            // Outward: the normal must point away from the tet's fourth vertex.
            const Vec3 n = (mesh.V.col(b) - mesh.V.col(a)).cross(mesh.V.col(c) - mesh.V.col(a));
            if (n.dot(mesh.V.col(f.opposite) - mesh.V.col(a)) > 0) std::swap(b, c);
            boundary.push_back({a, b, c});
        } else if (count > 2) {
            throw std::runtime_error("extract_surface: face (" + std::to_string(faces[s].a) + ", " +
                                     std::to_string(faces[s].b) + ", " + std::to_string(faces[s].c) +
                                     ") is shared by " + std::to_string(count) + " tets (non-manifold mesh)");
        }
        s = e;
    }

    mesh.F.resize(3, Eigen::Index(boundary.size()));
    for (size_t i = 0; i < boundary.size(); ++i)
        for (int k = 0; k < 3; ++k) mesh.F(k, Eigen::Index(i)) = boundary[i][size_t(k)];
    mesh.E = unique_edges(mesh.F);

    std::vector<int> verts(mesh.F.data(), mesh.F.data() + mesh.F.size());
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    mesh.surf_vertices.resize(Eigen::Index(verts.size()));
    for (size_t i = 0; i < verts.size(); ++i) mesh.surf_vertices[Eigen::Index(i)] = verts[i];
}

void compute_edges(TriMesh& mesh)
{
    mesh.E = unique_edges(mesh.F);
}

std::vector<int> select_vertices(const SelectionDesc& selection, const Eigen::Matrix3Xd& V)
{
    const int n = int(V.cols());
    std::vector<int> out;
    switch (selection.mode) {
        case SelectionDesc::Mode::All:
            out.resize(size_t(n));
            std::iota(out.begin(), out.end(), 0);
            break;
        case SelectionDesc::Mode::Indices:
            for (int idx : selection.indices) {
                if (idx < 0 || idx >= n)
                    throw std::runtime_error("select_vertices: index " + std::to_string(idx) + " outside [0, " +
                                             std::to_string(n) + ")");
                out.push_back(idx);
            }
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            break;
        case SelectionDesc::Mode::Bbox: {
            if (n == 0) break;
            const Vec3 lo = V.rowwise().minCoeff();
            const Vec3 hi = V.rowwise().maxCoeff();
            const Vec3 ext = hi - lo;
            // Tolerance so vertices exactly on the box boundary are included regardless of rounding.
            const double eps = 1e-9 * ext.maxCoeff() + 1e-300;
            const Vec3 bmin = lo + selection.bbox_min.cwiseProduct(ext);
            const Vec3 bmax = lo + selection.bbox_max.cwiseProduct(ext);
            for (int i = 0; i < n; ++i) {
                bool inside = true;
                for (int k = 0; k < 3 && inside; ++k)
                    inside = V(k, i) >= bmin[k] - eps && V(k, i) <= bmax[k] + eps;
                if (inside) out.push_back(i);
            }
            break;
        }
    }
    return out;
}

std::vector<double> lumped_masses(const TetMesh& mesh, double density)
{
    std::vector<double> mass(size_t(mesh.n_vertices()), 0.0);
    for (Eigen::Index t = 0; t < mesh.T.cols(); ++t) {
        const double vol =
            std::fabs(signed_volume6(mesh.V, mesh.T(0, t), mesh.T(1, t), mesh.T(2, t), mesh.T(3, t))) / 6.0;
        const double quarter = 0.25 * density * vol;
        for (int k = 0; k < 4; ++k) mass[size_t(mesh.T(k, t))] += quarter;
    }
    return mass;
}

void tet_rest_data(const TetMesh& mesh, std::vector<double>& DmInv, std::vector<double>& volume)
{
    const Eigen::Index nt = mesh.T.cols();
    DmInv.resize(size_t(9 * nt));
    volume.resize(size_t(nt));
    for (Eigen::Index t = 0; t < nt; ++t) {
        Eigen::Matrix3d Dm;
        const Vec3 x0 = mesh.V.col(mesh.T(0, t));
        for (int k = 0; k < 3; ++k) Dm.col(k) = mesh.V.col(mesh.T(k + 1, t)) - x0;
        const double det = Dm.determinant();
        if (!(det > 0))
            throw std::runtime_error("tet_rest_data: tet " + std::to_string(t) +
                                     " has non-positive volume (call orient_tets first)");
        const Eigen::Matrix3d inv = Dm.inverse();  // column-major storage
        std::copy(inv.data(), inv.data() + 9, DmInv.begin() + 9 * t);
        volume[size_t(t)] = det / 6.0;
    }
}

std::vector<double> shell_lumped_masses(const TriMesh& mesh, double density, double thickness)
{
    std::vector<double> mass(size_t(mesh.n_vertices()), 0.0);
    for (Eigen::Index f = 0; f < mesh.F.cols(); ++f) {
        const Vec3 a = mesh.V.col(mesh.F(0, f)), b = mesh.V.col(mesh.F(1, f)), c = mesh.V.col(mesh.F(2, f));
        const double area = 0.5 * ((b - a).cross(c - a)).norm();
        const double third = density * thickness * area / 3.0;
        for (int k = 0; k < 3; ++k) mass[size_t(mesh.F(k, f))] += third;
    }
    return mass;
}

void shell_rest_data(const TriMesh& mesh, double thickness, std::vector<double>& DmInv, std::vector<double>& volume)
{
    for (Eigen::Index f = 0; f < mesh.F.cols(); ++f) {
        const Vec3 X0 = mesh.V.col(mesh.F(0, f)), X1 = mesh.V.col(mesh.F(1, f)), X2 = mesh.V.col(mesh.F(2, f));
        const Vec3 d1 = X1 - X0, d2 = X2 - X0;
        const double l1 = d1.norm();
        const Vec3 n = d1.cross(d2);
        const double area = 0.5 * n.norm();
        if (!(l1 > 0.0) || !(area > 0.0))
            throw std::runtime_error("shell_rest_data: triangle " + std::to_string(f) + " is degenerate");
        // rest frame: e1 along the first edge, e2 = n x e1 in the plane (MS §3.5)
        const Vec3 e1 = d1 / l1;
        const Vec3 e2 = n.normalized().cross(e1);
        Eigen::Matrix2d Dm;
        Dm << e1.dot(d1), e1.dot(d2),
              0.0,        e2.dot(d2);
        const Eigen::Matrix2d inv = Dm.inverse();
        double pad[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   // column-major 3x3
        pad[0] = inv(0, 0); pad[1] = inv(1, 0);          // column 0
        pad[3] = inv(0, 1); pad[4] = inv(1, 1);          // column 1
        DmInv.insert(DmInv.end(), pad, pad + 9);
        volume.push_back(thickness * area);
    }
}

}  // namespace cs
