#include <HouGeo.h>

using namespace hou;

using namespace Eigen;

bool HCurve::GetCurveTopology(VectorXi *F) const
{
    if (topology_ == nullptr)
    {
        return false;
    }
    *F = topology_->GetTopology();
    return true;
}

bool HCurve::SetCurveTopology(const Eigen::VectorXi &F)
{
    topology_ = std::make_shared<Topology>();

    topology_->indices_ = F;

    return true;
}

bool HCurve::GetCurvePrimitive(VectorXi *prim) const
{
    if (primitive_ == nullptr)
    {
        return false;
    }
    *prim = primitive_->nverticesPerPrim_;
    return true;
}

bool HCurve::SetCurvePrimitive(const Eigen::VectorXi &prim)
{
    primitive_ = std::make_shared<Primitive>();

    primitive_->nverticesPerPrim_ = prim;

    primitive_->type_ = "c_r";

    primitive_->s_v_ = 0;

    primitive_->n_p_ = prim.size();

    return true;
}

// HSegment implementation
bool HSegment::GetSegmentTopology(MatrixXi *S) const
{
    if (primitive_ == nullptr)
    {
        return false;
    }
    if (topology_ == nullptr)
    {
        return false;
    }
    int num_segments = primitive_->GetNumPrimitive();

    *S = topology_->GetTopology().reshaped(2, num_segments);

    return true;
}

bool HSegment::SetSegmentTopology(const MatrixXi &S)
{
    topology_ = std::make_shared<Topology>();

    int num_segments = S.cols();

    int element_dim = S.rows();

    if (element_dim != 2)
    {
        std::cerr
            << "The segment topology matrix should be a 2xn matrix, your rows not 2, go back to check your matrix"
            << std::endl;
        return false;
    }

    topology_->indices_ = S.reshaped(num_segments * element_dim, 1);

    primitive_ = std::make_shared<Primitive>();

    primitive_->nverticesPerPrim_.resize(num_segments);
    primitive_->nverticesPerPrim_.setConstant(2);

    primitive_->type_ = "p_r";  // polygon run type for line segments
    primitive_->s_v_ = 0;
    primitive_->n_p_ = num_segments;

    return true;
}

bool HSegment::GetSegmentTopology(Matrix2Xi *S) const
{
    if (primitive_ == nullptr)
    {
        return false;
    }
    if (topology_ == nullptr)
    {
        return false;
    }
    int num_segments = primitive_->GetNumPrimitive();

    *S = topology_->GetTopology().reshaped(2, num_segments);

    return true;
}

bool HSegment::SetSegmentTopology(const Matrix2Xi &S)
{
    topology_ = std::make_shared<Topology>();

    int num_segments = S.cols();

    int element_dim = S.rows();

    if (element_dim != 2)
    {
        std::cerr
            << "The segment topology matrix should be a 2xn matrix, your rows not 2, go back to check your matrix"
            << std::endl;
        return false;
    }

    topology_->indices_ = S.reshaped(num_segments * element_dim, 1);

    primitive_ = std::make_shared<Primitive>();

    primitive_->nverticesPerPrim_.resize(num_segments);
    primitive_->nverticesPerPrim_.setConstant(2);

    primitive_->type_ = "p_r";  // polygon run type for line segments
    primitive_->s_v_ = 0;
    primitive_->n_p_ = num_segments;

    return true;
}