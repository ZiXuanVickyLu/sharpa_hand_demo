namespace hou {
    // ... other classes ...

    class HTetrahedralMesh : public HMesh {
    public:
        HTetrahedralMesh() = default;
        virtual ~HTetrahedralMesh() = default;

        bool GetTetrahedralTopology(Eigen::MatrixXi* T) const;
        bool SetTetrahedralTopology(const Eigen::MatrixXi& T);
    };

} // namespace hou 