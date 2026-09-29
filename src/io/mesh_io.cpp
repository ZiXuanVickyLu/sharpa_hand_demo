// Output writers (doc/al-ipc-implementation-spec.md §5.4 / §6.11). bgeo goes through the
// vendored HouGeoIO (HTetrahedra / HTriangleMesh); positions and attributes are written in
// float, which is what Houdini reads by default and enough for export. Numbers in the text
// formats (obj, csv) use the shortest round-trip representation.
#include "io/mesh_io.h"

#include <HouGeo.h>

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace cs {
namespace {

void append_number(std::string& out, double value)
{
    char buf[64];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), value);
    if (r.ec != std::errc()) throw std::runtime_error("mesh_io: number formatting failed");
    out.append(buf, r.ptr);
}

void create_parent_directories(const std::string& path)
{
    const std::filesystem::path p(path);
    if (!p.has_parent_path()) return;
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
    if (ec) throw std::runtime_error("mesh_io: cannot create directory '" + p.parent_path().string() + "': " + ec.message());
}

// HouIO::ExportHouGeo terminates the process (a bare `throw;`) when it cannot open the file, so
// the path is probed here first and any failure surfaces as an exception.
void probe_writable(const std::string& path)
{
    create_parent_directories(path);
    std::ofstream probe(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!probe) throw std::runtime_error("mesh_io: cannot open '" + path + "' for writing");
}

template <class Index>
void check_indices(const Index& I, Eigen::Index n_points, const char* what)
{
    for (Eigen::Index c = 0; c < I.cols(); ++c) {
        for (Eigen::Index r = 0; r < I.rows(); ++r) {
            const int v = I(r, c);
            if (v < 0 || v >= n_points) {
                throw std::runtime_error(std::string("mesh_io: ") + what + " " + std::to_string(c) + " references vertex " +
                                         std::to_string(v) + " outside [0, " + std::to_string(n_points) + ")");
            }
        }
    }
}

template <class HouMesh>
void set_point_data(HouMesh& mesh, const Eigen::Matrix3Xd& V, const std::map<std::string, Eigen::VectorXd>& point_scalars,
                    const std::map<std::string, Eigen::Matrix3Xd>& point_vectors)
{
    const Eigen::Index n = V.cols();
    const Eigen::Matrix3Xf P = V.cast<float>();
    mesh.SetPointAttributeT("P", P);
    for (const auto& [name, values] : point_scalars) {
        if (name == "P") throw std::runtime_error("mesh_io: point attribute name 'P' is reserved for positions");
        if (values.size() != n) {
            throw std::runtime_error("mesh_io: point scalar '" + name + "' has " + std::to_string(values.size()) +
                                     " values for " + std::to_string(n) + " points");
        }
        const Eigen::VectorXf f = values.cast<float>();
        mesh.SetPointAttribute(name, f);
    }
    for (const auto& [name, values] : point_vectors) {
        if (name == "P") throw std::runtime_error("mesh_io: point attribute name 'P' is reserved for positions");
        if (values.cols() != n) {
            throw std::runtime_error("mesh_io: point vector '" + name + "' has " + std::to_string(values.cols()) +
                                     " columns for " + std::to_string(n) + " points");
        }
        const Eigen::Matrix3Xf f = values.cast<float>();
        mesh.SetPointAttributeT(name, f);
    }
}

void export_geo(const std::string& path, const hou::HouGeo& geo)
{
    probe_writable(path);
    if (!hou::HouIO::ExportHouGeo(path, geo)) throw std::runtime_error("mesh_io: HouGeoIO failed to write '" + path + "'");
}

}  // namespace

void write_bgeo_tets(const std::string& path, const Eigen::Matrix3Xd& V, const Eigen::Matrix4Xi& T,
                     const std::map<std::string, Eigen::VectorXd>& point_scalars,
                     const std::map<std::string, Eigen::Matrix3Xd>& point_vectors)
{
    check_indices(T, V.cols(), "tet");
    hou::HTetrahedra mesh;
    set_point_data(mesh, V, point_scalars, point_vectors);
    if (!mesh.SetTetTopology(T)) throw std::runtime_error("mesh_io: HouGeoIO rejected the tet topology");
    export_geo(path, mesh);
}

void write_bgeo_tris(const std::string& path, const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F,
                     const std::map<std::string, Eigen::VectorXd>& point_scalars,
                     const std::map<std::string, Eigen::Matrix3Xd>& point_vectors)
{
    check_indices(F, V.cols(), "triangle");
    hou::HTriangleMesh mesh;
    set_point_data(mesh, V, point_scalars, point_vectors);
    if (!mesh.SetTriangleTopology(F)) throw std::runtime_error("mesh_io: HouGeoIO rejected the triangle topology");
    export_geo(path, mesh);
}

void write_obj_tris(const std::string& path, const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F)
{
    check_indices(F, V.cols(), "triangle");
    create_parent_directories(path);
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("mesh_io: cannot open '" + path + "' for writing");
    std::string text;
    text.reserve(std::size_t(V.cols()) * 40 + std::size_t(F.cols()) * 24);
    for (Eigen::Index i = 0; i < V.cols(); ++i) {
        text += "v ";
        append_number(text, V(0, i));
        text += ' ';
        append_number(text, V(1, i));
        text += ' ';
        append_number(text, V(2, i));
        text += '\n';
    }
    for (Eigen::Index f = 0; f < F.cols(); ++f) {
        text += "f " + std::to_string(F(0, f) + 1) + ' ' + std::to_string(F(1, f) + 1) + ' ' + std::to_string(F(2, f) + 1) + '\n';
    }
    out << text;
    if (!out) throw std::runtime_error("mesh_io: write to '" + path + "' failed");
}

// ---------------------------------------------------------------------------------------------

void CsvWriter::open(const std::string& path, const std::vector<std::string>& columns)
{
    if (columns.empty()) throw std::runtime_error("CsvWriter::open: no columns");
    if (out_.is_open()) out_.close();
    out_.clear();
    create_parent_directories(path);
    out_.open(path, std::ios::out | std::ios::trunc);
    if (!out_) throw std::runtime_error("CsvWriter::open: cannot open '" + path + "' for writing");
    path_ = path;
    columns_ = columns;
    std::string header;
    for (std::size_t i = 0; i < columns_.size(); ++i) {
        if (i) header += ',';
        header += columns_[i];
    }
    header += '\n';
    out_ << header;
    out_.flush();
    header_written_ = true;
}

void CsvWriter::row(const std::vector<double>& values)
{
    if (!out_.is_open() || !header_written_) throw std::runtime_error("CsvWriter::row: writer is not open");
    if (values.size() != columns_.size()) {
        throw std::runtime_error("CsvWriter::row: " + std::to_string(values.size()) + " values for " +
                                 std::to_string(columns_.size()) + " columns");
    }
    std::string line;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) line += ',';
        append_number(line, values[i]);
    }
    line += '\n';
    out_ << line;
    if (!out_) throw std::runtime_error("CsvWriter::row: write to '" + path_ + "' failed");
}

void CsvWriter::flush()
{
    if (out_.is_open()) out_.flush();
}

}  // namespace cs
