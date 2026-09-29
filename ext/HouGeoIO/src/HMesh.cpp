#include <HouGeo.h>

using namespace hou;
using namespace Eigen;

bool HMesh::GetTopologyInRunLengthEncoding(Eigen::VectorXi *topo) const
{
    *topo = topology_->GetTopology();

    return true;
}

bool HMesh::GetPrimitive(Eigen::VectorXi *prim) const
{
    *prim = primitive_->nverticesPerPrim_;
    return true;
}

template <typename T>
void HMesh::SetVertexAttribute(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    typedef MatrixX<Base> Target;
    MatrixX<Base> r_data = data.transpose();

    auto attr_iter = vertex_attributes_.find(name);
    if (attr_iter == vertex_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();
        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        ptr->type_ = "numeric";

        vertex_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->data_ = r_data;
    }
}


template <typename T>
bool HMesh::GetVertexAttribute(const std::string &name, T *data) const
{
    typedef typename ExtractBase<T>::type Base;

    if (vertex_attributes_.find(name) == vertex_attributes_.end())
    {
        return false;
    }

    Attribute::CPtr ptr = vertex_attributes_.at(name);

    std::visit([data](const auto& stored_matrix) {
        using StoredType = typename std::decay_t<decltype(stored_matrix)>::Scalar;
        if constexpr (std::is_same_v<Base, StoredType>) {
            *data = stored_matrix.transpose();
        } else {
            *data = stored_matrix.template cast<Base>().transpose();
        }
    }, ptr->data_);

    return true;
}

template void HMesh::SetVertexAttribute<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HMesh::SetVertexAttribute<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HMesh::SetVertexAttribute<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HMesh::SetVertexAttribute<VectorXi>(const std::string &name, const VectorXi &data);
template void HMesh::SetVertexAttribute<VectorXf>(const std::string &name, const VectorXf &data);
template void HMesh::SetVertexAttribute<VectorXd>(const std::string &name, const VectorXd &data);

template bool HMesh::GetVertexAttribute<MatrixXf>(const std::string &name, MatrixXf *data) const;
template bool HMesh::GetVertexAttribute<MatrixXd>(const std::string &name, MatrixXd *data) const;
template bool HMesh::GetVertexAttribute<MatrixXi>(const std::string &name, MatrixXi *data) const;
template bool HMesh::GetVertexAttribute<VectorXf>(const std::string &name, VectorXf *data) const;
template bool HMesh::GetVertexAttribute<VectorXd>(const std::string &name, VectorXd *data) const;
template bool HMesh::GetVertexAttribute<VectorXi>(const std::string &name, VectorXi *data) const;

template <typename T>
void HMesh::SetPrimitiveAttribute(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    typedef MatrixX<Base> Target;
    MatrixX<Base> r_data = data.transpose();

    auto attr_iter = primitive_attributes_.find(name);
    if (primitive_attributes_.find(name) == primitive_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();
        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        ptr->type_ = "numeric";

        primitive_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->data_ = r_data;
    }
}

template <typename T>
bool HMesh::GetPrimitiveAttribute(const std::string &name, T *data) const
{
    typedef typename ExtractBase<T>::type Base;
    typedef MatrixX<Base> Target;

    if (primitive_attributes_.find(name) == primitive_attributes_.end())
    {
        return false;
    }

    Attribute::CPtr ptr = primitive_attributes_.at(name);

    std::visit([data](const auto& stored_matrix) {
       using StoredType = typename std::decay_t<decltype(stored_matrix)>::Scalar;
        // In Houdini, rows is tupeSize, cols is element count
       if constexpr (std::is_same_v<Base, StoredType>) {
           *data = stored_matrix.transpose();
       } else {
           *data = stored_matrix.template cast<Base>().transpose();
       }
   }, ptr->data_);
    return true;
}

template void HMesh::SetPrimitiveAttribute<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HMesh::SetPrimitiveAttribute<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HMesh::SetPrimitiveAttribute<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HMesh::SetPrimitiveAttribute<VectorXi>(const std::string &name, const VectorXi &data);
template void HMesh::SetPrimitiveAttribute<VectorXf>(const std::string &name, const VectorXf &data);
template void HMesh::SetPrimitiveAttribute<VectorXd>(const std::string &name, const VectorXd &data);


template bool HMesh::GetPrimitiveAttribute<MatrixXf>(const std::string &name, MatrixXf *data) const;
template bool HMesh::GetPrimitiveAttribute<MatrixXd>(const std::string &name, MatrixXd *data) const;
template bool HMesh::GetPrimitiveAttribute<MatrixXi>(const std::string &name, MatrixXi *data) const;
template bool HMesh::GetPrimitiveAttribute<VectorXf>(const std::string &name, VectorXf *data) const;
template bool HMesh::GetPrimitiveAttribute<VectorXd>(const std::string &name, VectorXd *data) const;
template bool HMesh::GetPrimitiveAttribute<VectorXi>(const std::string &name, VectorXi *data) const;

// Transpose version for attribute shape (N, d)
template <typename T>
void HMesh::SetVertexAttributeT(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    typedef MatrixX<Base> Target;
    MatrixX<Base> r_data = data;

    auto attr_iter = vertex_attributes_.find(name);
    if (attr_iter == vertex_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();
        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        ptr->type_ = "numeric";

        vertex_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->data_ = r_data;
    }
}

template <typename T>
bool HMesh::GetVertexAttributeT(const std::string &name, T *data) const
{
    typedef typename ExtractBase<T>::type Base;
    typedef MatrixX<Base> Target;

    if (vertex_attributes_.find(name) == vertex_attributes_.end())
    {
        return false;
    }

    Attribute::CPtr ptr = vertex_attributes_.at(name);
    std::visit([data](const auto& stored_matrix) {
       using StoredType = typename std::decay_t<decltype(stored_matrix)>::Scalar;
        // In Houdini, rows is tupeSize, cols is element count
       if constexpr (std::is_same_v<Base, StoredType>) {
           *data = stored_matrix;
       } else {
           *data = stored_matrix.template cast<Base>();
       }
   }, ptr->data_);
    return true;
}

template void HMesh::SetVertexAttributeT<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HMesh::SetVertexAttributeT<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HMesh::SetVertexAttributeT<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HMesh::SetVertexAttributeT<Matrix3Xf>(const std::string &name, const Matrix3Xf &data);
template void HMesh::SetVertexAttributeT<Matrix3Xd>(const std::string &name, const Matrix3Xd &data);
template void HMesh::SetVertexAttributeT<Matrix3Xi>(const std::string &name, const Matrix3Xi &data);

template bool HMesh::GetVertexAttributeT<MatrixXf>(const std::string &name, MatrixXf *data) const;
template bool HMesh::GetVertexAttributeT<MatrixXd>(const std::string &name, MatrixXd *data) const;
template bool HMesh::GetVertexAttributeT<MatrixXi>(const std::string &name, MatrixXi *data) const;
template bool HMesh::GetVertexAttributeT<Matrix3Xf>(const std::string &name, Matrix3Xf *data) const;
template bool HMesh::GetVertexAttributeT<Matrix3Xd>(const std::string &name, Matrix3Xd *data) const;
template bool HMesh::GetVertexAttributeT<Matrix3Xi>(const std::string &name, Matrix3Xi *data) const;

template <typename T>
void HMesh::SetPrimitiveAttributeT(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    typedef MatrixX<Base> Target;
    MatrixX<Base> r_data = data;

    auto attr_iter = primitive_attributes_.find(name);
    if (primitive_attributes_.find(name) == primitive_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();
        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        ptr->type_ = "numeric";

        primitive_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->data_ = r_data;
    }
}

template <typename T>
bool HMesh::GetPrimitiveAttributeT(const std::string &name, T *data) const
{
    typedef typename ExtractBase<T>::type Base;
    typedef MatrixX<Base> Target;

    if (primitive_attributes_.find(name) == primitive_attributes_.end())
    {
        return false;
    }

    Attribute::CPtr ptr = primitive_attributes_.at(name);
    std::visit([data](const auto& stored_matrix) {
       using StoredType = typename std::decay_t<decltype(stored_matrix)>::Scalar;
        // In Houdini, rows is tupeSize, cols is element count
       if constexpr (std::is_same_v<Base, StoredType>) {
           *data = stored_matrix;
       } else {
           *data = stored_matrix.template cast<Base>();
       }
   }, ptr->data_);
    return true;
}

template void HMesh::SetPrimitiveAttributeT<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HMesh::SetPrimitiveAttributeT<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HMesh::SetPrimitiveAttributeT<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HMesh::SetPrimitiveAttributeT<Matrix3Xf>(const std::string &name, const Matrix3Xf &data);
template void HMesh::SetPrimitiveAttributeT<Matrix3Xd>(const std::string &name, const Matrix3Xd &data);
template void HMesh::SetPrimitiveAttributeT<Matrix3Xi>(const std::string &name, const Matrix3Xi &data);

template bool HMesh::GetPrimitiveAttributeT<MatrixXf>(const std::string &name, MatrixXf *data) const;
template bool HMesh::GetPrimitiveAttributeT<MatrixXd>(const std::string &name, MatrixXd *data) const;
template bool HMesh::GetPrimitiveAttributeT<MatrixXi>(const std::string &name, MatrixXi *data) const;
template bool HMesh::GetPrimitiveAttributeT<Matrix3Xf>(const std::string &name, Matrix3Xf *data) const;
template bool HMesh::GetPrimitiveAttributeT<Matrix3Xd>(const std::string &name, Matrix3Xd *data) const;
template bool HMesh::GetPrimitiveAttributeT<Matrix3Xi>(const std::string &name, Matrix3Xi *data) const;



//topo of triangle mesh will be 3xM matrix, column major matrix, where M is the number of faces
bool HTriangleMesh::GetTriangleTopology(MatrixXi *F) const

{
    if (primitive_ == nullptr)
    {
        return false;
    }
    if (topology_ == nullptr)
    {
        return false;
    }
    int num_triangles = primitive_->GetNumPrimitive();

    *F = topology_->GetTopology().reshaped(3, num_triangles);

    return true;
}

bool HTriangleMesh::SetTriangleTopology(const MatrixXi &F)
{
    topology_ = std::make_shared<Topology>();

    int num_triangles = F.cols();

    int element_dim = F.rows();

    if (element_dim != 3)
    {
        std::cerr
            << "The triangle topology matrix should be a 3xn matrix, your rows not 3, go back to check your matrix"
            << std::endl;
        return false;
    }

    topology_->indices_ = F.reshaped(num_triangles * element_dim, 1);

    primitive_ = std::make_shared<Primitive>();

    primitive_->nverticesPerPrim_.resize(num_triangles);
    primitive_->nverticesPerPrim_.setConstant(3);

    primitive_->type_ = "p_r";
    primitive_->s_v_ = 0;
    primitive_->n_p_ = num_triangles;

    return true;
}

//topo of triangle mesh will be 3xM matrix, column major matrix, where M is the number of faces
bool HTriangleMesh::GetTriangleTopology(Matrix3Xi *F) const

{
    if (primitive_ == nullptr)
    {
        return false;
    }
    if (topology_ == nullptr)
    {
        return false;
    }
    int num_triangles = primitive_->GetNumPrimitive();

    *F = topology_->GetTopology().reshaped(3, num_triangles);

    return true;
}

bool HTriangleMesh::SetTriangleTopology(const Matrix3Xi &F)
{
    topology_ = std::make_shared<Topology>();

    int num_triangles = F.cols();

    int element_dim = F.rows();

    if (element_dim != 3)
    {
        std::cerr
            << "The triangle topology matrix should be a 3xn matrix, your rows not 3, go back to check your matrix"
            << std::endl;
        return false;
    }

    topology_->indices_ = F.reshaped(num_triangles * element_dim, 1);

    primitive_ = std::make_shared<Primitive>();

    primitive_->nverticesPerPrim_.resize(num_triangles);
    primitive_->nverticesPerPrim_.setConstant(3);

    primitive_->type_ = "p_r";
    primitive_->s_v_ = 0;
    primitive_->n_p_ = num_triangles;

    return true;
}


bool HCube::GetCubeTopology(MatrixXi *V) const
{
    if (primitive_ == nullptr)
    {
        return false;
    }
    if (topology_ == nullptr)
    {
        return false;
    }
    int num_cubes = primitive_->GetNumPrimitive();

    *V = topology_->GetTopology().reshaped(8, num_cubes);

    return true;
}

bool HCube::SetCubeTopology(const Eigen::MatrixXi &V)
{
    topology_ = std::make_shared<Topology>();

    int num_cubes = V.cols();

    int element_dim = V.rows();

    if (element_dim != 8)
    {
        std::cerr << "The cube topology matrix should be a 8xn matrix, your rows not 8, go back to check your matrix"
                  << std::endl;
        return false;
    }

    topology_->indices_ = V.reshaped(num_cubes * element_dim, 1);

    primitive_ = std::make_shared<Primitive>();

    primitive_->type_ = "h_r";

    primitive_->s_v_ = 0;

    primitive_->n_p_ = num_cubes;
    return true;
}

bool HTetrahedra::GetTetTopology(MatrixXi *V) const
{
    if (primitive_ == nullptr)
    {
        return false;
    }
    if (topology_ == nullptr)
    {
        return false;
    }
    int num_tets = primitive_->GetNumPrimitive();

    *V = topology_->GetTopology().reshaped(4, num_tets);

    return true;
}

bool HTetrahedra::SetTetTopology(const Eigen::MatrixXi &V)
{
    topology_ = std::make_shared<Topology>();

    int num_tets = V.cols();

    int element_dim = V.rows();

    if (element_dim != 4)
    {
        std::cerr << "The tet topology matrix should be a 4xn matrix, your rows not 4, go back to check your matrix"
                  << std::endl;
        return false;
    }

    topology_->indices_ = V.reshaped(num_tets * element_dim, 1);

    primitive_ = std::make_shared<Primitive>();

    primitive_->type_ = "t_r";

    primitive_->s_v_ = 0;

    primitive_->n_p_ = num_tets;

    return true;
}

bool HTetrahedra::GetTetTopology(Matrix4Xi *V) const
{
    if (primitive_ == nullptr)
    {
        return false;
    }
    if (topology_ == nullptr)
    {
        return false;
    }
    int num_tets = primitive_->GetNumPrimitive();

    *V = topology_->GetTopology().reshaped(4, num_tets);

    return true;
}

bool HTetrahedra::SetTetTopology(const Eigen::Matrix4Xi &V)
{
    topology_ = std::make_shared<Topology>();

    int num_tets = V.cols();

    int element_dim = V.rows();

    if (element_dim != 4)
    {
        std::cerr << "The tet topology matrix should be a 4xn matrix, your rows not 4, go back to check your matrix"
                  << std::endl;
        return false;
    }

    topology_->indices_ = V.reshaped(num_tets * element_dim, 1);

    primitive_ = std::make_shared<Primitive>();

    primitive_->type_ = "t_r";

    primitive_->s_v_ = 0;

    primitive_->n_p_ = num_tets;

    return true;
}