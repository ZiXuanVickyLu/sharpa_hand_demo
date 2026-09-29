#pragma once
// Output writers (doc/al-ipc-implementation-spec.md §5.4 / §6.11): bgeo via HouGeoIO for
// Houdini, obj as a lowest-common-denominator surface format, CSV for per-step statistics.
#include <Eigen/Core>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace cs {

// Tetrahedral mesh with optional per-point scalar/vector attributes.
void write_bgeo_tets(const std::string& path, const Eigen::Matrix3Xd& V, const Eigen::Matrix4Xi& T,
                     const std::map<std::string, Eigen::VectorXd>& point_scalars = {},
                     const std::map<std::string, Eigen::Matrix3Xd>& point_vectors = {});

// Triangle mesh (surface export) with optional attributes.
void write_bgeo_tris(const std::string& path, const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F,
                     const std::map<std::string, Eigen::VectorXd>& point_scalars = {},
                     const std::map<std::string, Eigen::Matrix3Xd>& point_vectors = {});

void write_obj_tris(const std::string& path, const Eigen::Matrix3Xd& V, const Eigen::Matrix3Xi& F);

// Append-only CSV with a fixed header written on first use. open() truncates the file and
// writes the header; row() appends one line (values.size() must equal the column count);
// flush() pushes buffered rows to disk (call once per step so a crash loses at most one row).
class CsvWriter {
public:
    CsvWriter() = default;
    void open(const std::string& path, const std::vector<std::string>& columns);
    void row(const std::vector<double>& values);
    void flush();
    bool is_open() const { return out_.is_open(); }
    const std::string& path() const { return path_; }
private:
    std::string path_;
    std::vector<std::string> columns_;
    bool header_written_ = false;
    std::ofstream out_;
};

}  // namespace cs
