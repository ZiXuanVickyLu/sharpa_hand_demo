// Mesh loaders (doc/al-ipc-implementation-spec.md §5.2): bgeo through HouGeoIO, gmsh 2.2 ASCII,
// TetGen .node/.ele, Wavefront obj. Every failure is a std::runtime_error naming the file.
#include "geometry/mesh.h"

#include <HouGeo.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cs {
namespace {

[[noreturn]] void fail(const std::string& path, const std::string& msg)
{
    throw std::runtime_error("mesh load: " + path + ": " + msg);
}

std::string lower_extension(const std::string& path)
{
    const auto dot = path.find_last_of('.');
    const auto slash = path.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return "";
    std::string ext = path.substr(dot);
    for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
    return ext;
}

std::string replace_extension(const std::string& path, const std::string& ext)
{
    const auto dot = path.find_last_of('.');
    const auto slash = path.find_last_of("/\\");
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return path + ext;
    return path.substr(0, dot) + ext;
}

std::string read_whole_file(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(path, "cannot open file");
    in.seekg(0, std::ios::end);
    const std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::string buf;
    buf.resize(size_t(std::max<std::streamoff>(size, 0)));
    if (size > 0) in.read(buf.data(), size);
    return buf;
}

// Whitespace-separated token scanner over an in-memory buffer; optionally treats '#' as a
// comment-to-end-of-line marker (TetGen).
class Scanner {
public:
    Scanner(const std::string& buf, const std::string& path, bool hash_comments)
        : p_(buf.data()), end_(buf.data() + buf.size()), path_(path), hash_comments_(hash_comments)
    {
    }

    bool next(std::string_view& tok)
    {
        skip();
        if (p_ == end_) return false;
        const char* s = p_;
        while (p_ != end_ && !is_space(*p_)) ++p_;
        tok = std::string_view(s, size_t(p_ - s));
        return true;
    }

    long long integer(const char* what)
    {
        std::string_view t;
        if (!next(t)) fail(path_, std::string("unexpected end of file while reading ") + what);
        long long v = 0;
        const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
        if (r.ec != std::errc() || r.ptr != t.data() + t.size())
            fail(path_, std::string("expected an integer for ") + what + ", got '" + std::string(t) + "'");
        return v;
    }

    double real(const char* what)
    {
        std::string_view t;
        if (!next(t)) fail(path_, std::string("unexpected end of file while reading ") + what);
        double v = 0;
        const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
        if (r.ec != std::errc() || r.ptr != t.data() + t.size())
            fail(path_, std::string("expected a number for ") + what + ", got '" + std::string(t) + "'");
        return v;
    }

    // True when another token follows on the current line (spaces and tabs skipped, newlines
    // not). MEDIT writers put a record's reference number on the record's line, so a record
    // whose line ends after its fixed fields has no reference (Arm13K.mesh of [Z25]'s assets).
    bool more_on_line() const
    {
        const char* q = p_;
        while (q != end_ && (*q == ' ' || *q == '\t')) ++q;
        return q != end_ && *q != '\n' && *q != '\r';
    }

    void expect(const char* keyword)
    {
        std::string_view t;
        if (!next(t) || t != keyword) fail(path_, std::string("expected '") + keyword + "'");
    }

    // Skips tokens up to and including `keyword`.
    void skip_until(const std::string& keyword)
    {
        std::string_view t;
        while (next(t))
            if (t == keyword) return;
        fail(path_, "missing '" + keyword + "'");
    }

private:
    static bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
    void skip()
    {
        for (;;) {
            while (p_ != end_ && is_space(*p_)) ++p_;
            if (hash_comments_ && p_ != end_ && *p_ == '#') {
                while (p_ != end_ && *p_ != '\n') ++p_;
                continue;
            }
            return;
        }
    }
    const char* p_;
    const char* end_;
    std::string path_;
    bool hash_comments_;
};

void check_indices(const Eigen::Ref<const Eigen::MatrixXi>& idx, int n_vertices, const std::string& path,
                   const char* what)
{
    for (Eigen::Index c = 0; c < idx.cols(); ++c)
        for (Eigen::Index r = 0; r < idx.rows(); ++r)
            if (idx(r, c) < 0 || idx(r, c) >= n_vertices)
                fail(path, std::string(what) + " " + std::to_string(c) + " references vertex " +
                               std::to_string(idx(r, c)) + " outside [0, " + std::to_string(n_vertices) + ")");
}

// ---------------------------------------------------------------------------
// bgeo (HouGeoIO)
// ---------------------------------------------------------------------------
template <class Geo>
void import_bgeo(const std::string& path, Geo& geo)
{
    bool ok = false;
    try {
        ok = hou::HouIO::ImportHouGeo(path, &geo);
    } catch (const std::exception& e) {
        fail(path, std::string("bgeo import failed: ") + e.what());
    }
    if (!ok) fail(path, "bgeo import failed");
}

TetMesh load_bgeo_tets(const std::string& path)
{
    hou::HTetrahedra geo;
    import_bgeo(path, geo);
    TetMesh mesh;
    if (!geo.GetPointAttributeT("P", &mesh.V)) fail(path, "no point attribute 'P'");
    Eigen::VectorXi prim;
    if (geo.GetPrimitive(&prim) && prim.size() > 0 && (prim.array() != 4).any())
        fail(path, "not a tetrahedral mesh (a primitive does not have 4 vertices)");
    if (!geo.GetTetTopology(&mesh.T)) fail(path, "no tetrahedra topology");
    check_indices(mesh.T, mesh.n_vertices(), path, "tet");
    return mesh;
}

TriMesh load_bgeo_tris(const std::string& path)
{
    hou::HTriangleMesh geo;
    import_bgeo(path, geo);
    TriMesh mesh;
    if (!geo.GetPointAttributeT("P", &mesh.V)) fail(path, "no point attribute 'P'");
    Eigen::VectorXi prim;
    if (geo.GetPrimitive(&prim) && prim.size() > 0 && (prim.array() != 3).any())
        fail(path, "not a triangle mesh (a primitive does not have 3 vertices); triangulate it first");
    if (!geo.GetTriangleTopology(&mesh.F)) fail(path, "no triangle topology");
    check_indices(mesh.F, mesh.n_vertices(), path, "triangle");
    return mesh;
}

// ---------------------------------------------------------------------------
// gmsh 2.2 ASCII
// ---------------------------------------------------------------------------
int gmsh_nodes_per_element(int type)
{
    switch (type) {
        case 1: return 2;    // line
        case 2: return 3;    // triangle
        case 3: return 4;    // quad
        case 4: return 4;    // tet
        case 5: return 8;    // hex
        case 6: return 6;    // prism
        case 7: return 5;    // pyramid
        case 8: return 3;    // 3-node line
        case 9: return 6;    // 6-node triangle
        case 10: return 9;   // 9-node quad
        case 11: return 10;  // 10-node tet
        case 12: return 27;  // 27-node hex
        case 13: return 18;  // 18-node prism
        case 14: return 14;  // 14-node pyramid
        case 15: return 1;   // point
        case 16: return 8;   // 8-node quad
        case 17: return 20;  // 20-node hex
        case 18: return 15;  // 15-node prism
        case 19: return 13;  // 13-node pyramid
        default: return -1;
    }
}

TetMesh load_msh(const std::string& path)
{
    const std::string buf = read_whole_file(path);
    Scanner sc(buf, path, false);

    std::vector<long long> node_ids;
    std::vector<double> xyz;
    std::vector<long long> tet_nodes;  // 4 per tet, gmsh node numbers
    bool have_format = false, have_nodes = false;
    bool v4 = false;                   // MSH 4.1: entity blocks in $Nodes and $Elements (IS §5.2)

    std::string_view tok;
    while (sc.next(tok)) {
        if (tok == "$MeshFormat") {
            const double version = sc.real("version");
            const long long file_type = sc.integer("file-type");
            sc.integer("data-size");
            v4 = version >= 4.1 && version < 5.0;
            if (!v4 && (version < 2.0 || version >= 3.0))
                fail(path, "unsupported gmsh format version " + std::to_string(version) + " (need 2.x or 4.1)");
            if (file_type != 0) fail(path, "binary gmsh files are not supported");
            sc.expect("$EndMeshFormat");
            have_format = true;
        } else if (tok == "$Nodes" && v4) {
            // numEntityBlocks numNodes minNodeTag maxNodeTag; per block: entityDim entityTag parametric
            // numNodesInBlock, then the block's node tags, then its coordinates (+ entityDim parametric
            // coordinates per node when parametric is set).
            const long long blocks = sc.integer("entity block count");
            const long long n = sc.integer("node count");
            sc.integer("min node tag");
            sc.integer("max node tag");
            if (n < 0 || blocks < 0) fail(path, "negative node count");
            node_ids.reserve(size_t(n));
            xyz.reserve(size_t(3 * n));
            for (long long b = 0; b < blocks; ++b) {
                const long long dim = sc.integer("entity dimension");
                sc.integer("entity tag");
                const long long parametric = sc.integer("parametric flag");
                const long long nb = sc.integer("nodes in block");
                if (nb < 0) fail(path, "negative node count in an entity block");
                for (long long i = 0; i < nb; ++i) node_ids.push_back(sc.integer("node tag"));
                for (long long i = 0; i < nb; ++i) {
                    xyz.push_back(sc.real("node x"));
                    xyz.push_back(sc.real("node y"));
                    xyz.push_back(sc.real("node z"));
                    if (parametric) for (long long k = 0; k < dim; ++k) sc.real("parametric coordinate");
                }
            }
            if ((long long)node_ids.size() != n) fail(path, "$Nodes: the entity blocks hold " + std::to_string(node_ids.size()) +
                                                                " nodes, the header says " + std::to_string(n));
            sc.expect("$EndNodes");
            have_nodes = true;
        } else if (tok == "$Elements" && v4) {
            // numEntityBlocks numElements minElementTag maxElementTag; per block: entityDim entityTag
            // elementType numElementsInBlock, then elementTag node1 ... nodeN per element.
            const long long blocks = sc.integer("entity block count");
            sc.integer("element count");
            sc.integer("min element tag");
            sc.integer("max element tag");
            for (long long b = 0; b < blocks; ++b) {
                sc.integer("entity dimension");
                sc.integer("entity tag");
                const int type = int(sc.integer("element type"));
                const long long mb = sc.integer("elements in block");
                const int nn = gmsh_nodes_per_element(type);
                if (nn < 0) fail(path, "unsupported element type " + std::to_string(type));
                for (long long i = 0; i < mb; ++i) {
                    sc.integer("element tag");
                    if (type == 4 || type == 11) {
                        for (int k = 0; k < 4; ++k) tet_nodes.push_back(sc.integer("tet node"));
                        for (int k = 4; k < nn; ++k) sc.integer("tet node");
                    } else {
                        for (int k = 0; k < nn; ++k) sc.integer("element node");
                    }
                }
            }
            sc.expect("$EndElements");
        } else if (tok == "$Nodes") {
            const long long n = sc.integer("node count");
            if (n < 0) fail(path, "negative node count");
            node_ids.resize(size_t(n));
            xyz.resize(size_t(3 * n));
            for (long long i = 0; i < n; ++i) {
                node_ids[size_t(i)] = sc.integer("node number");
                xyz[size_t(3 * i + 0)] = sc.real("node x");
                xyz[size_t(3 * i + 1)] = sc.real("node y");
                xyz[size_t(3 * i + 2)] = sc.real("node z");
            }
            sc.expect("$EndNodes");
            have_nodes = true;
        } else if (tok == "$Elements") {
            const long long m = sc.integer("element count");
            if (m < 0) fail(path, "negative element count");
            tet_nodes.reserve(size_t(4 * m));
            for (long long i = 0; i < m; ++i) {
                sc.integer("element number");
                const int type = int(sc.integer("element type"));
                const long long ntags = sc.integer("tag count");
                for (long long t = 0; t < ntags; ++t) sc.integer("tag");
                const int nn = gmsh_nodes_per_element(type);
                if (nn < 0) fail(path, "unsupported element type " + std::to_string(type));
                if (type == 4 || type == 11) {
                    for (int k = 0; k < 4; ++k) tet_nodes.push_back(sc.integer("tet node"));
                    for (int k = 4; k < nn; ++k) sc.integer("tet node");
                } else {
                    for (int k = 0; k < nn; ++k) sc.integer("element node");
                }
            }
            sc.expect("$EndElements");
        } else if (tok.size() > 1 && tok[0] == '$' && tok.rfind("$End", 0) != 0) {
            sc.skip_until("$End" + std::string(tok.substr(1)));
        } else {
            fail(path, "unexpected token '" + std::string(tok) + "'");
        }
    }
    if (!have_format) fail(path, "missing $MeshFormat section");
    if (!have_nodes) fail(path, "missing $Nodes section");
    if (tet_nodes.empty()) fail(path, "no tetrahedra (element type 4) found");

    // Node number -> index. gmsh numbers are usually 1..n; fall back to a map otherwise.
    const size_t n = node_ids.size();
    bool contiguous = true;
    for (size_t i = 0; i < n && contiguous; ++i) contiguous = node_ids[i] == (long long)i + 1;
    std::unordered_map<long long, int> id_map;
    if (!contiguous) {
        id_map.reserve(n * 2);
        for (size_t i = 0; i < n; ++i)
            if (!id_map.emplace(node_ids[i], int(i)).second)
                fail(path, "duplicate node number " + std::to_string(node_ids[i]));
    }
    auto to_index = [&](long long id) -> int {
        if (contiguous) {
            if (id < 1 || id > (long long)n) fail(path, "element references unknown node " + std::to_string(id));
            return int(id - 1);
        }
        const auto it = id_map.find(id);
        if (it == id_map.end()) fail(path, "element references unknown node " + std::to_string(id));
        return it->second;
    };

    TetMesh mesh;
    mesh.V.resize(3, Eigen::Index(n));
    if (n > 0) std::copy(xyz.begin(), xyz.end(), mesh.V.data());
    const size_t nt = tet_nodes.size() / 4;
    mesh.T.resize(4, Eigen::Index(nt));
    for (size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) mesh.T(k, Eigen::Index(t)) = to_index(tet_nodes[4 * t + size_t(k)]);
    return mesh;
}

// ---------------------------------------------------------------------------
// MEDIT / INRIA .mesh (ASCII): the format the [Z25] supplementary assets ship in
// ---------------------------------------------------------------------------
// Sections used: `Vertices` (n, then x y z [ref]), `Tetrahedra` (n, then four 1-based ids [ref]);
// the reference numbers and the MeshVersionFormatted/Dimension header are optional.
// `Triangles`, `Edges`, `Corners` and the rest are skipped: the contact surface is extracted
// from the tets. Node ids are 1-based and contiguous by the format's definition.
TetMesh load_medit(const std::string& path)
{
    // MEDIT files carry `# ...` comment lines (the [Z25] assets do); the scanner tokenises by
    // whitespace, so blank them out before scanning rather than teach it about line ends.
    std::string buf = read_whole_file(path);
    for (std::size_t i = 0; i < buf.size();) {
        std::size_t eol = buf.find('\n', i);
        if (eol == std::string::npos) eol = buf.size();
        std::size_t first = i;
        while (first < eol && (buf[first] == ' ' || buf[first] == '\t' || buf[first] == '\r')) ++first;
        if (first < eol && buf[first] == '#') std::fill(buf.begin() + long(i), buf.begin() + long(eol), ' ');
        i = eol + 1;
    }
    Scanner sc(buf, path, false);
    std::vector<double> xyz;
    std::vector<long long> tet_nodes;
    bool have_vertices = false;
    std::string_view tok;
    while (sc.next(tok)) {
        if (tok == "MeshVersionFormatted") {
            sc.integer("version");
        } else if (tok == "Dimension") {
            const long long dim = sc.integer("dimension");
            if (dim != 3) fail(path, "MEDIT dimension " + std::to_string(dim) + " (need 3)");
        } else if (tok == "Vertices") {
            const long long n = sc.integer("vertex count");
            if (n < 0) fail(path, "negative vertex count");
            xyz.resize(size_t(3 * n));
            for (long long i = 0; i < n; ++i) {
                xyz[size_t(3 * i + 0)] = sc.real("vertex x");
                xyz[size_t(3 * i + 1)] = sc.real("vertex y");
                xyz[size_t(3 * i + 2)] = sc.real("vertex z");
                if (sc.more_on_line()) sc.integer("vertex ref");
            }
            have_vertices = true;
        } else if (tok == "Tetrahedra") {
            const long long m = sc.integer("tet count");
            if (m < 0) fail(path, "negative tet count");
            tet_nodes.reserve(size_t(4 * m));
            for (long long i = 0; i < m; ++i) {
                for (int k = 0; k < 4; ++k) tet_nodes.push_back(sc.integer("tet node"));
                if (sc.more_on_line()) sc.integer("tet ref");
            }
        } else if (tok == "Triangles" || tok == "Edges" || tok == "Quadrilaterals" || tok == "Hexahedra") {
            const long long m = sc.integer("element count");
            const int per = (tok == "Edges") ? 2 : (tok == "Triangles") ? 3 : (tok == "Quadrilaterals") ? 4 : 8;
            for (long long i = 0; i < m; ++i) {
                for (int k = 0; k < per; ++k) sc.integer("element node");
                if (sc.more_on_line()) sc.integer("element ref");
            }
        } else if (tok == "Corners" || tok == "RequiredVertices" || tok == "Ridges" || tok == "RequiredEdges") {
            const long long m = sc.integer("count");
            for (long long i = 0; i < m; ++i) sc.integer("id");
        } else if (tok == "End") {
            break;
        } else {
            fail(path, "unexpected MEDIT keyword '" + std::string(tok) + "'");
        }
    }
    if (!have_vertices) fail(path, "missing Vertices section");
    if (tet_nodes.empty()) fail(path, "no Tetrahedra section");
    const size_t n = xyz.size() / 3;
    TetMesh mesh;
    mesh.V.resize(3, Eigen::Index(n));
    if (n > 0) std::copy(xyz.begin(), xyz.end(), mesh.V.data());
    const size_t nt = tet_nodes.size() / 4;
    mesh.T.resize(4, Eigen::Index(nt));
    for (size_t t = 0; t < nt; ++t)
        for (int k = 0; k < 4; ++k) {
            const long long id = tet_nodes[4 * t + size_t(k)];
            if (id < 1 || id > (long long)n) fail(path, "tet references unknown vertex " + std::to_string(id));
            mesh.T(k, Eigen::Index(t)) = int(id - 1);
        }
    return mesh;
}

// ---------------------------------------------------------------------------
// TetGen .node / .ele
// ---------------------------------------------------------------------------
TetMesh load_tetgen(const std::string& given_path)
{
    const std::string node_path = replace_extension(given_path, ".node");
    const std::string ele_path = replace_extension(given_path, ".ele");
    TetMesh mesh;
    long long base = 0;
    {
        const std::string buf = read_whole_file(node_path);
        Scanner sc(buf, node_path, true);
        const long long n = sc.integer("point count");
        const long long dim = sc.integer("dimension");
        const long long nattr = sc.integer("attribute count");
        const long long nmark = sc.integer("boundary marker flag");
        if (n < 0) fail(node_path, "negative point count");
        if (dim != 3) fail(node_path, "dimension must be 3");
        if (nattr < 0 || nmark < 0) fail(node_path, "malformed header");
        mesh.V.resize(3, Eigen::Index(n));
        for (long long i = 0; i < n; ++i) {
            const long long idx = sc.integer("point index");
            if (i == 0) {
                if (idx != 0 && idx != 1) fail(node_path, "first point index must be 0 or 1");
                base = idx;
            }
            if (idx != i + base)
                fail(node_path, "point indices must be consecutive (expected " + std::to_string(i + base) + ", got " +
                                    std::to_string(idx) + ")");
            for (int k = 0; k < 3; ++k) mesh.V(k, Eigen::Index(i)) = sc.real("point coordinate");
            for (long long a = 0; a < nattr; ++a) sc.real("point attribute");
            for (long long m = 0; m < nmark; ++m) sc.integer("boundary marker");
        }
    }
    {
        const std::string buf = read_whole_file(ele_path);
        Scanner sc(buf, ele_path, true);
        const long long m = sc.integer("tet count");
        const long long npt = sc.integer("nodes per tet");
        const long long nattr = sc.integer("attribute count");
        if (m < 0) fail(ele_path, "negative tetrahedron count");
        if (npt != 4 && npt != 10) fail(ele_path, "nodes per tetrahedron must be 4 or 10");
        if (nattr < 0) fail(ele_path, "malformed header");
        mesh.T.resize(4, Eigen::Index(m));
        for (long long t = 0; t < m; ++t) {
            sc.integer("tet index");
            for (long long k = 0; k < npt; ++k) {
                const long long v = sc.integer("tet node") - base;
                if (k < 4) mesh.T(int(k), Eigen::Index(t)) = int(v);
            }
            for (long long a = 0; a < nattr; ++a) sc.real("tet attribute");
        }
    }
    check_indices(mesh.T, mesh.n_vertices(), ele_path, "tet");
    return mesh;
}

// ---------------------------------------------------------------------------
// Wavefront obj (positions and faces only; polygons are fan-triangulated)
// ---------------------------------------------------------------------------
TriMesh load_obj(const std::string& path)
{
    std::ifstream in(path);
    if (!in) fail(path, "cannot open file");
    std::vector<double> xyz;
    std::vector<int> tris;
    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key) || key.empty() || key[0] == '#') continue;
        if (key == "v") {
            double x, y, z;
            if (!(ls >> x >> y >> z)) fail(path, "line " + std::to_string(line_no) + ": malformed vertex");
            xyz.push_back(x);
            xyz.push_back(y);
            xyz.push_back(z);
        } else if (key == "f") {
            const int nv = int(xyz.size() / 3);
            std::vector<int> poly;
            std::string tok;
            while (ls >> tok) {
                const std::string first = tok.substr(0, tok.find('/'));
                int idx = 0;
                const auto r = std::from_chars(first.data(), first.data() + first.size(), idx);
                if (r.ec != std::errc() || r.ptr != first.data() + first.size() || idx == 0)
                    fail(path, "line " + std::to_string(line_no) + ": malformed face index '" + tok + "'");
                idx = idx < 0 ? nv + idx : idx - 1;
                if (idx < 0 || idx >= nv)
                    fail(path, "line " + std::to_string(line_no) + ": face index out of range");
                poly.push_back(idx);
            }
            if (poly.size() < 3) fail(path, "line " + std::to_string(line_no) + ": face with fewer than 3 vertices");
            for (size_t k = 1; k + 1 < poly.size(); ++k) {
                tris.push_back(poly[0]);
                tris.push_back(poly[k]);
                tris.push_back(poly[k + 1]);
            }
        }
        // vn, vt, o, g, s, usemtl, mtllib, l, p: ignored
    }
    TriMesh mesh;
    mesh.V.resize(3, Eigen::Index(xyz.size() / 3));
    std::copy(xyz.begin(), xyz.end(), mesh.V.data());
    mesh.F.resize(3, Eigen::Index(tris.size() / 3));
    std::copy(tris.begin(), tris.end(), mesh.F.data());
    return mesh;
}

}  // namespace

TetMesh load_tet_mesh(const std::string& path)
{
    const std::string ext = lower_extension(path);
    if (ext == ".bgeo") return load_bgeo_tets(path);
    if (ext == ".msh") return load_msh(path);
    if (ext == ".mesh") return load_medit(path);
    if (ext == ".node" || ext == ".ele") return load_tetgen(path);
    fail(path, "unsupported tetrahedral mesh format '" + ext + "' (expected .bgeo, .msh, .mesh, .node/.ele)");
}

TriMesh load_tri_mesh(const std::string& path)
{
    const std::string ext = lower_extension(path);
    if (ext == ".bgeo") return load_bgeo_tris(path);
    if (ext == ".obj") return load_obj(path);
    fail(path, "unsupported triangle mesh format '" + ext + "' (expected .bgeo, .obj)");
}

}  // namespace cs
