#pragma once
#include <map>
#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <houjson/json.h>
#include <Eigen/Core>

namespace hou {

class HouIO;

class HouGeo
{
protected:
    struct Attribute
    {
        typedef std::shared_ptr<Attribute> Ptr;
        typedef std::shared_ptr<const Attribute> CPtr;
        typedef std::variant<Eigen::MatrixXf, Eigen::MatrixXd, Eigen::MatrixXi> Data;

        const int &GetNumElment() const { return num_element_; };
        const int &GetTupleSize() const { return tuple_size_; };
        const std::string &GetName() const { return name_; };
        const std::string &GetStorage() const { return storage_; };
        const std::string &GetType() const { return type_; };

        const void *GetRawPointer() const;
        void *GetRawPointer();

        int num_element_ = 1; // for global attributes, num_element should be 1
        int tuple_size_;
        std::string name_;
        std::string storage_ = "fpreal32"; // fpreal32, fpreal64, int32
        std::string type_ = "numeric";     // numeric string
        Data data_;

        friend class HouIO;
    };
    typedef Attribute PointAttribute;
    typedef Attribute VertexAttribute;
    typedef Attribute PrimitiveAttribute;
    typedef Attribute GlobalAttribute;

    struct Primitive
    {
        typedef std::shared_ptr<Primitive> Ptr;
        const std::string &GetType() const { return type_; };
        const int &GetStartVertex() const { return s_v_; };
        const int &GetNumPrimitive() const { return n_p_; };
        const int *GetNumVerticesArrayRawPointer() const { return nverticesPerPrim_.data(); };
        int *GetNumVerticesArrayRawPointer() { return nverticesPerPrim_.data(); };

        // We follow the name in .bgeo
        Eigen::VectorXi
            nverticesPerPrim_; // nvertices in .geo. Actually, we have nvertices_rle model(r_v) and nvertice model in
                               // houdini, but when we import, we always transfer it to nvertices model(n_v).
        std::string type_;     // PolygonCurve_run = c_r, Polygon = p_r
        int s_v_ = 0;          // startvertex
        int n_p_ = 1;          // nprimitives

        friend class HouIO;
    };

    struct Topology
    {
        typedef std::shared_ptr<Topology> Ptr;
        const Eigen::VectorXi &GetTopology() const { return indices_; }
        Eigen::VectorXi &GetTopology() { return indices_; }

        Eigen::VectorXi indices_;

        friend class HouIO;
    };

protected:
    std::map<std::string, PointAttribute::Ptr> point_attributes_;
    std::map<std::string, VertexAttribute::Ptr> vertex_attributes_;
    std::map<std::string, PrimitiveAttribute::Ptr> primitive_attributes_;
    std::map<std::string, GlobalAttribute::Ptr> global_attributes_;

    Primitive::Ptr primitive_;
    Topology::Ptr topology_;

    friend class HouIO;

    template <typename T>

    struct ExtractBase
    {
    private:
        template <typename T1>
        static auto helper() -> typename T1::Scalar;

        template <typename T1>
        static auto helper() -> typename std::enable_if<std::is_fundamental_v<T1>, T1>::type;

    public:
        using type = decltype(helper<T>());
    };

    template <typename T>
    constexpr char *GetType()
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return "fpreal32";
        }
        else if (std::is_same_v<T, double>)
        {
            return "fpreal64";
        }
        else if (std::is_same_v<T, int>)
        {
            return "int32";
        }
    }
};

class HouIO
{
public:
    static bool ImportHouGeo(const std::string &file, HouGeo *geo);
    static bool ExportHouGeo(const std::string &file, const HouGeo &geo);

private:
    static HouGeo::Attribute::Ptr LoadAttribute(hou::json::ArrayPtr attribute, const int32_t elementCount);
    static HouGeo::Topology::Ptr LoadTopology(json::ObjectPtr o);
    static HouGeo::Primitive::Ptr LoadPolyPrimitive(hou::json::ArrayPtr primitive);
    static void ExportAttribute(json::BinaryWriter *g_writer, HouGeo::Attribute::CPtr attr);
};

// Frontend:
// Particle, Point Cloud
// for the point attribute, we only support column major matrix
// default: point attribute should be Nxd matrix, where d is the dimension, N is the number of points
// For Transpose matrix (d, N), we have SetPointAttributeT(name, data).
// This is useful for setting point attribute like position, velocity, where the dimension is 3 and the number of points is the number of vertices. (3, #V) attribute without transpose in meaningless.
// The user should take care of the matrix format.

class HParticle : public HouGeo
{
public:
    HParticle();

    // Normal version for attribute shape (N, d)
    template <typename T>
    void SetPointAttribute(const std::string &name, const T &data);

    template <typename T>
    void SetGlobalAttribute(const std::string &name, const T &data);

    template <typename T>
    bool GetPointAttribute(const std::string &name, T *data) const;

    template <typename T>
    bool GetGlobalAttribute(const std::string &name, T *data) const;

    // Transpose version for attribute shape (d, N)
    template <typename T>
    void SetPointAttributeT(const std::string &name, const T &data);

    template <typename T>
    void SetGlobalAttributeT(const std::string &name, const T &data);

    template <typename T>
    bool GetPointAttributeT(const std::string &name, T *data) const;

    template <typename T>
    bool GetGlobalAttributeT(const std::string &name, T *data) const;
};

// Surface Mesh, Curve, Volume Mesh

class HMesh : public HParticle
{
public:
    template <typename T>
    void SetVertexAttribute(const std::string &name, const T &data);

    template <typename T>
    void SetPrimitiveAttribute(const std::string &name, const T &data);

    template <typename T>
    bool GetVertexAttribute(const std::string &name, T *data) const;

    template <typename T>
    bool GetPrimitiveAttribute(const std::string &name, T *data) const;

    // Transpose version for attribute shape (d, N)

    template <typename T>
    void SetVertexAttributeT(const std::string &name, const T &data);

    template <typename T>
    void SetPrimitiveAttributeT(const std::string &name, const T &data);

    template <typename T>
    bool GetVertexAttributeT(const std::string &name, T *data) const;

    template <typename T>
    bool GetPrimitiveAttributeT(const std::string &name, T *data) const;
    

    bool GetTopologyInRunLengthEncoding(Eigen::VectorXi *topo) const;

    bool GetPrimitive(Eigen::VectorXi *prim) const;
};

// Triangle Mesh
// topology should be 3xM matrix, column major matrix, where M is the number of faces
// the user should take care of the matrix format.
class HTriangleMesh : public HMesh
{
public:
    bool GetTriangleTopology(Eigen::MatrixXi *F) const;
    bool SetTriangleTopology(const Eigen::MatrixXi &F);
    bool GetTriangleTopology(Eigen::Matrix3Xi *F) const;
    bool SetTriangleTopology(const Eigen::Matrix3Xi &F);
};

class HCurve : public HMesh
{
public:
    bool GetCurveTopology(Eigen::VectorXi *F) const;
    bool SetCurveTopology(const Eigen::VectorXi &F);

    bool GetCurvePrimitive(Eigen::VectorXi *prim) const;
    bool SetCurvePrimitive(const Eigen::VectorXi &prim);
};

// Segment (Line Segments)
// Segments are line primitives with 2 vertices each
// topology should be 2xN matrix, column major matrix, where N is the number of segments
// the user should take care of the matrix format.
class HSegment : public HMesh
{
public:
    bool GetSegmentTopology(Eigen::MatrixXi *S) const;
    bool SetSegmentTopology(const Eigen::MatrixXi &S);
    bool GetSegmentTopology(Eigen::Matrix2Xi *S) const;
    bool SetSegmentTopology(const Eigen::Matrix2Xi &S);
};

// Cube
// default: cube topology should be 8xN matrix, column major matrix, where N is the number of cubes
// the user should take care of the matrix format.
class HCube : public HMesh
{
public:
    bool GetCubeTopology(Eigen::MatrixXi *V) const;
    bool SetCubeTopology(const Eigen::MatrixXi &V);
};

// Tetrahedra
// default: tetrahedra topology should be 4xN matrix, column major matrix, where N is the number of tetrahedra
// the user should take care of the matrix format.
class HTetrahedra : public HMesh
{
public:
    bool GetTetTopology(Eigen::MatrixXi *V) const;
    bool SetTetTopology(const Eigen::MatrixXi &V);
    bool GetTetTopology(Eigen::Matrix4Xi *V) const;
    bool SetTetTopology(const Eigen::Matrix4Xi &V);
};

} // namespace hou