#include <HouGeo.h>

using namespace hou;
using namespace Eigen;

HParticle::HParticle()
{
    topology_ = std::make_shared<HouGeo::Topology>();
    primitive_ = nullptr;
}

template <typename T>
void HParticle::SetPointAttribute(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    MatrixX<Base> r_data = data.transpose();

    auto attr_iter = point_attributes_.find(name);
    if (attr_iter == point_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();
        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        ptr->type_ = "numeric";

        point_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->data_ = r_data;
    }
}

template void HParticle::SetPointAttribute<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HParticle::SetPointAttribute<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HParticle::SetPointAttribute<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HParticle::SetPointAttribute<VectorXi>(const std::string &name, const VectorXi &data);
template void HParticle::SetPointAttribute<VectorXf>(const std::string &name, const VectorXf &data);
template void HParticle::SetPointAttribute<VectorXd>(const std::string &name, const VectorXd &data);

template <typename T>
void HParticle::SetGlobalAttribute(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    MatrixX<Base> r_data;

    if constexpr (std::is_fundamental_v<T>)
    {
        r_data.resize(1, 1);
        r_data(0, 0) = data;
    }
    else
    {
        r_data = data.transpose();
    }

    auto attr_iter = global_attributes_.find(name);
    if (attr_iter == global_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();

        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        if (ptr->tuple_size_ == 1 && !std::is_fundamental_v<T>)
        {
            ptr->type_ = "arraydata";
        }
        else
        {
            ptr->type_ = "numeric";
        }
        global_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->num_element_ = r_data.cols();
        attr_iter->second->storage_ = GetType<Base>();
        attr_iter->second->tuple_size_ = r_data.rows();
        attr_iter->second->data_ = r_data;
    }
}

template void HParticle::SetGlobalAttribute<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HParticle::SetGlobalAttribute<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HParticle::SetGlobalAttribute<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HParticle::SetGlobalAttribute<VectorXf>(const std::string &name, const VectorXf &data);
template void HParticle::SetGlobalAttribute<VectorXi>(const std::string &name, const VectorXi &data);
template void HParticle::SetGlobalAttribute<VectorXd>(const std::string &name, const VectorXd &data);
template void HParticle::SetGlobalAttribute<float>(const std::string &name, const float &data);
template void HParticle::SetGlobalAttribute<int>(const std::string &name, const int &data);
template void HParticle::SetGlobalAttribute<double>(const std::string &name, const double &data);

template <typename T>
bool HParticle::GetPointAttribute(const std::string &name, T *data) const
{
    typedef typename ExtractBase<T>::type Base;

    if (point_attributes_.find(name) == point_attributes_.end())
    {
        return false;
    }

    Attribute::CPtr ptr = point_attributes_.at(name);

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

template bool HParticle::GetPointAttribute<Eigen::MatrixXf>(const std::string &name, Eigen::MatrixXf *data) const;
template bool HParticle::GetPointAttribute<Eigen::MatrixXd>(const std::string &name, Eigen::MatrixXd *data) const;
template bool HParticle::GetPointAttribute<Eigen::MatrixXi>(const std::string &name, Eigen::MatrixXi *data) const;
template bool HParticle::GetPointAttribute<Eigen::VectorXf>(const std::string &name, Eigen::VectorXf *data) const;
template bool HParticle::GetPointAttribute<Eigen::VectorXd>(const std::string &name, Eigen::VectorXd *data) const;
template bool HParticle::GetPointAttribute<Eigen::VectorXi>(const std::string &name, Eigen::VectorXi *data) const;

template <typename T>
bool HParticle::GetGlobalAttribute(const std::string &name, T *data) const
{
    typedef typename ExtractBase<T>::type Base;

    if (global_attributes_.find(name) == global_attributes_.end())
    {
        return false;
    }

    Attribute::CPtr ptr = global_attributes_.at(name);

    if constexpr (std::is_fundamental<T>::value)
    {
        *data = *reinterpret_cast<const T *>(ptr->GetRawPointer());
    }
    else
    {
        std::visit([data, ptr](const auto& stored_matrix) {
            using StoredType = typename std::decay_t<decltype(stored_matrix)>::Scalar;
            if constexpr (std::is_same_v<Base, StoredType>) {
                if (ptr->GetType() == "arraydata") {
                    *data = stored_matrix; // VectorX case
                } else {
                    *data = stored_matrix.transpose(); // Houdini format conversion, In Houdini, rows is tupeSize, cols is element count
                }
            } else {
                // Different type, cast then apply logic
                if (ptr->GetType() == "arraydata") {
                    *data = stored_matrix.template cast<Base>(); // VectorX case
                } else {
                    *data = stored_matrix.template cast<Base>().transpose(); // Houdini format conversion, In Houdini, rows is tupeSize, cols is element count
                }
            }
        }, ptr->data_);
    }

    return true;
}


template bool HParticle::GetGlobalAttribute<Eigen::MatrixXf>(const std::string &name, Eigen::MatrixXf *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::MatrixXd>(const std::string &name, Eigen::MatrixXd *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::MatrixXi>(const std::string &name, Eigen::MatrixXi *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::VectorXf>(const std::string &name, Eigen::VectorXf *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::VectorXd>(const std::string &name, Eigen::VectorXd *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::VectorXi>(const std::string &name, Eigen::VectorXi *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::Vector3f>(const std::string &name, Eigen::Vector3f *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::Vector3i>(const std::string &name, Eigen::Vector3i *data) const;
template bool HParticle::GetGlobalAttribute<Eigen::Vector3d>(const std::string &name, Eigen::Vector3d *data) const;
template bool HParticle::GetGlobalAttribute<float>(const std::string &name, float *data) const;
template bool HParticle::GetGlobalAttribute<double>(const std::string &name, double *data) const;
template bool HParticle::GetGlobalAttribute<int>(const std::string &name, int *data) const;

// Transpose version
template <typename T>
void HParticle::SetPointAttributeT(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    MatrixX<Base> r_data = data;

    auto attr_iter = point_attributes_.find(name);
    if (attr_iter == point_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();
        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        ptr->type_ = "numeric";

        point_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->data_ = r_data;
    }
}

template void HParticle::SetPointAttributeT<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HParticle::SetPointAttributeT<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HParticle::SetPointAttributeT<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HParticle::SetPointAttributeT<Matrix3Xf>(const std::string &name, const Matrix3Xf &data);
template void HParticle::SetPointAttributeT<Matrix3Xd>(const std::string &name, const Matrix3Xd &data);
template void HParticle::SetPointAttributeT<Matrix3Xi>(const std::string &name, const Matrix3Xi &data);



template <typename T>
void HParticle::SetGlobalAttributeT(const std::string &name, const T &data)
{
    typedef typename ExtractBase<T>::type Base;
    MatrixX<Base> r_data;

    if constexpr (std::is_fundamental_v<T>)
    {
        r_data.resize(1, 1);
        r_data(0, 0) = data;
    }
    else
    {
        r_data = data; // In Houdini, rows is tupeSize, cols is element count
    }

    auto attr_iter = global_attributes_.find(name);
    if (attr_iter == global_attributes_.end())
    {
        Attribute::Ptr ptr = std::make_shared<Attribute>();

        ptr->data_ = r_data;
        ptr->name_ = name;
        ptr->num_element_ = r_data.cols();
        ptr->storage_ = GetType<Base>();
        ptr->tuple_size_ = r_data.rows();
        if (ptr->tuple_size_ == 1 && !std::is_fundamental_v<T>)
        {
            ptr->type_ = "arraydata";
        }
        else
        {
            ptr->type_ = "numeric";
        }
        global_attributes_.insert({name, ptr});
    }
    else
    {
        attr_iter->second->num_element_ = r_data.cols();
        attr_iter->second->storage_ = GetType<Base>();
        attr_iter->second->tuple_size_ = r_data.rows();
        attr_iter->second->data_ = r_data;
    }
}

template void HParticle::SetGlobalAttributeT<MatrixXf>(const std::string &name, const MatrixXf &data);
template void HParticle::SetGlobalAttributeT<MatrixXi>(const std::string &name, const MatrixXi &data);
template void HParticle::SetGlobalAttributeT<MatrixXd>(const std::string &name, const MatrixXd &data);
template void HParticle::SetGlobalAttributeT<Matrix3Xf>(const std::string &name, const Matrix3Xf &data);
template void HParticle::SetGlobalAttributeT<Matrix3Xd>(const std::string &name, const Matrix3Xd &data);
template void HParticle::SetGlobalAttributeT<Matrix3Xi>(const std::string &name, const Matrix3Xi &data);
template void HParticle::SetGlobalAttributeT<VectorXf>(const std::string &name, const VectorXf &data);
template void HParticle::SetGlobalAttributeT<VectorXd>(const std::string &name, const VectorXd &data);
template void HParticle::SetGlobalAttributeT<VectorXi>(const std::string &name, const VectorXi &data);
template void HParticle::SetGlobalAttributeT<Vector3f>(const std::string &name, const Vector3f &data);
template void HParticle::SetGlobalAttributeT<Vector3d>(const std::string &name, const Vector3d &data);
template void HParticle::SetGlobalAttributeT<Vector3i>(const std::string &name, const Vector3i &data);
template void HParticle::SetGlobalAttributeT<float>(const std::string &name, const float &data);
template void HParticle::SetGlobalAttributeT<int>(const std::string &name, const int &data);
template void HParticle::SetGlobalAttributeT<double>(const std::string &name, const double &data);


template <typename T>
bool HParticle::GetPointAttributeT(const std::string &name, T *data) const {
    typedef typename ExtractBase<T>::type Base;

    if (point_attributes_.find(name) == point_attributes_.end()) {
        return false;
    }

    Attribute::CPtr ptr = point_attributes_.at(name);
    std::visit([data](const auto& stored_matrix) {
        using StoredType = typename std::decay_t<decltype(stored_matrix)>::Scalar;
        if constexpr (std::is_same_v<Base, StoredType>) {
            *data = stored_matrix;
        } else {
            *data = stored_matrix.template cast<Base>();
        }
    }, ptr->data_);

    return true;
}

template bool HParticle::GetPointAttributeT<Eigen::MatrixXf>(const std::string &name, Eigen::MatrixXf *data) const;
template bool HParticle::GetPointAttributeT<Eigen::MatrixXd>(const std::string &name, Eigen::MatrixXd *data) const;
template bool HParticle::GetPointAttributeT<Eigen::MatrixXi>(const std::string &name, Eigen::MatrixXi *data) const;
template bool HParticle::GetPointAttributeT<Eigen::Matrix3Xf>(const std::string &name, Eigen::Matrix3Xf *data) const;
template bool HParticle::GetPointAttributeT<Eigen::Matrix3Xd>(const std::string &name, Eigen::Matrix3Xd *data) const;
template bool HParticle::GetPointAttributeT<Eigen::Matrix3Xi>(const std::string &name, Eigen::Matrix3Xi *data) const;


template <typename T>
bool HParticle::GetGlobalAttributeT(const std::string &name, T *data) const
{
    typedef typename ExtractBase<T>::type Base;

    if (global_attributes_.find(name) == global_attributes_.end())
    {
        return false;
    }

    Attribute::CPtr ptr = global_attributes_.at(name);

    if constexpr (std::is_fundamental<T>::value)
    {
        *data = *reinterpret_cast<const T *>(ptr->GetRawPointer());
    }
    else
    {
        std::visit([data](const auto& stored_matrix) {
            using StoredType = typename std::decay_t<decltype(stored_matrix)>::Scalar;
            if constexpr (std::is_same_v<Base, StoredType>) {
                *data = stored_matrix;
            } else {
                *data = stored_matrix.template cast<Base>();
            }
        }, ptr->data_);
    }

    return true;
}

template bool HParticle::GetGlobalAttributeT<Eigen::MatrixXf>(const std::string &name, Eigen::MatrixXf *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::MatrixXd>(const std::string &name, Eigen::MatrixXd *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::MatrixXi>(const std::string &name, Eigen::MatrixXi *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::Matrix3Xf>(const std::string &name, Eigen::Matrix3Xf *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::Matrix3Xd>(const std::string &name, Eigen::Matrix3Xd *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::Matrix3Xi>(const std::string &name, Eigen::Matrix3Xi *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::VectorXf>(const std::string &name, Eigen::VectorXf *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::Vector3f>(const std::string &name, Eigen::Vector3f *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::VectorXd>(const std::string &name, Eigen::VectorXd *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::Vector3d>(const std::string &name, Eigen::Vector3d *data) const;
template bool HParticle::GetGlobalAttributeT<Eigen::VectorXi>(const std::string &name, Eigen::VectorXi *data) const;
template bool HParticle::GetGlobalAttributeT<float>(const std::string &name, float *data) const;
template bool HParticle::GetGlobalAttributeT<double>(const std::string &name, double *data) const;
template bool HParticle::GetGlobalAttributeT<int>(const std::string &name, int *data) const;
