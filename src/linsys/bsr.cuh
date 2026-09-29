#pragma once
// 3x3 block-sparse-row matrix over the free vertices (doc/al-ipc-implementation-spec.md §4.3,
// §6.4, §6.6). Both triangles are stored so the SpMV is a pure gather (no atomics); the
// elasticity assembly writes each upper slot and its mirror. Blocks are row-major 3x3
// (b[0..2] = first row) in `real`.
#include "core/typedef.cuh"
#include "core/device_buffer.cuh"
#include "scene/scene.h"
#include <vector>

namespace cs {

struct BsrPattern {
    int n_rows = 0;      // = n_free
    int n_blocks = 0;    // stored blocks (both triangles + diagonal)
    int n_upper = 0;     // slots with col >= row

    DeviceArray<int> row_ptr;      // n_rows + 1
    DeviceArray<int> col_idx;      // n_blocks, sorted within each row
    DeviceArray<int> diag_slot;    // per row: slot of the (v, v) block
    DeviceArray<int> slot_mirror;  // per slot: slot of the transposed block (self for diagonal)

    // Elasticity gather map (Appendix B): for every upper slot, the (tet, a, b) contributions,
    // a = local index (0..3) of the ROW vertex in the tet, b = local index of the COL vertex.
    DeviceArray<int> upper_slot;            // n_upper -> slot
    DeviceArray<int> contrib_ptr;           // n_upper + 1
    DeviceArray<int> contrib_tet;           // tet index
    DeviceArray<unsigned char> contrib_ab;  // a * 4 + b

    // Host copies (pattern only) for tests and diagnostics.
    std::vector<int> h_row_ptr, h_col_idx;

    // Build from the scene's tets restricted to free vertices (prescribed vertices are not
    // rows/cols). Deterministic ordering: rows ascending, cols ascending within a row,
    // contributions ordered by (tet, a, b).
    static BsrPattern build(const Scene& scene);
};

struct BsrMatrix {
    const BsrPattern* pattern = nullptr;
    DeviceArray<real> blocks;    // 9 * n_blocks
    DeviceArray<real> diag_inv;  // 9 * n_rows, inverse of the (possibly augmented) diagonal block
    // Scratch of the two-phase elasticity assembly (fem_assemble_bsr, IS §12.3 item 11): the
    // ten 3x3 blocks of every tet of the current chunk, 90 reals per tet.
    DeviceArray<real> assembly_stage;

    void init(const BsrPattern& p);
    void zero();
    int n_rows() const { return pattern ? pattern->n_rows : 0; }
};

// blocks[diag(v)] += s[v] * I  (lumped mass on the diagonal).
void bsr_add_scaled_identity(BsrMatrix& A, const real* per_row_scalar);

// blocks[diag(v)] += D[v] (9 per row, row-major), e.g. the contact diagonal (§6.5).
void bsr_add_diagonal_blocks(BsrMatrix& A, const real* per_row_block9);

// y = A x over the free vertices (gather SpMV, 16 lanes per block-row, fixed summation order).
// `stream` is the launch stream (the PCG graph captures on its own stream, §6.6); `skip`, when
// given, is a device flag that makes the kernel return at once when non-zero (the graph's early
// exit past convergence).
void bsr_spmv(const BsrMatrix& A, const real3* x, real3* y, cudaStream_t stream = 0, const int* skip = nullptr);

// diag_inv[v] = inverse(blocks[diag(v)] + extra[v]) where extra (9 per row) may be null.
// Symmetrizes the inverse ((B^-1 + B^-T)/2) so the preconditioner is exactly symmetric.
void bsr_build_block_jacobi(BsrMatrix& A, const real* extra_diag_block9);

// z = diag_inv * r.
void bsr_apply_block_jacobi(const BsrMatrix& A, const real3* r, real3* z);

// Largest diagonal entry over all rows and the three components (for the μ estimate). Always
// double: the μ estimate and every global reduction are double in every build (spec §11.1).
// `scratch` holds the n per-row maxima and is kept between calls by the owner.
double bsr_max_diagonal_entry(const BsrMatrix& A, DeviceArray<real>& scratch);

// Dense copy of the matrix (3 n_rows square, row-major doubles) for tests on small systems.
void bsr_to_dense(const BsrMatrix& A, std::vector<double>& dense);

}  // namespace cs
