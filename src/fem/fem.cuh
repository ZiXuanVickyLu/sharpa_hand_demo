#pragma once
// Elasticity kernels (doc/al-ipc-implementation-spec.md §6.4, math spec §3 and Appendix B).
// All functions launch on the default stream and check errors; positions are the FULL vertex
// array (prescribed vertices hold their targets), outputs are over free vertices only.
#include "core/typedef.cuh"
#include "core/device_buffer.cuh"
#include "gpu/scene_buffers.cuh"

namespace cs {

// Per-tet cache written by fem_element_cache (sizes n_tets * stride).
//   U, V     : 9 each, column-major rotations of the signed SVD of F
//   eig      : 9 = { a0+, a1+, a2+ (scaling), tau01+, tau02+, tau12+ (twist), phi01+, phi02+, phi12+ (flip) }
//              PSD-projected (max(., 0)) eigenvalues of the F-Hessian (MS §3.3)
//   q        : 9 = the three scaling eigenvectors q_s of the 3x3 matrix A, column-major
//   P        : 9 = first Piola-Kirchhoff stress, column-major
//   energy   : Ψ(F) V_e (NOT scaled by h^2)
struct ElementCache {
    DeviceArray<real> U, V, eig, q, P, energy;
    void resize(int n_tets);
};

// Fills the cache at positions x. `need_hessian` = false skips the eigen-system (gradient/energy only).
void fem_element_cache(const SceneBuffers& sb, const real3* x, ElementCache& cache, bool need_hessian);

// Elastic energy Σ_e V_e Ψ(F_e(x)) (NOT scaled by h^2), evaluated directly from positions
// (no SVD for SNH/NH; COR uses the SVD). Fixed-order two-level reduction in double.
real fem_elastic_energy(const SceneBuffers& sb, const real3* x, DeviceArray<real>& scratch);
// The same sum written to the device double at d_out, no host synchronisation (IS §12.3 item 10).
void fem_elastic_energy_to(const SceneBuffers& sb, const real3* x, DeviceArray<real>& scratch, double* d_out);

// G_v += h2 * Σ_{e ∋ v} V_e P_e a_{a(v)} for free vertices (gather over vt_* CSR).
// `cache.P` must be current for x.
void fem_add_gradient(const SceneBuffers& sb, const ElementCache& cache, real h2, real3* G);

// Per-free-vertex diagonal 3x3 blocks of h2 * Σ_e (H_e^+)_{aa} (Appendix B, a == b), used for
// the μ estimate before any matrix exists. out: 9 per free vertex (row-major within block).
void fem_diagonal_blocks(const SceneBuffers& sb, const ElementCache& cache, real h2, real* out);

// Max over free vertices and components of ( mass_v + diag(h2 H)_{vv} ) — MS §11.1.
real fem_max_diagonal(const SceneBuffers& sb, const ElementCache& cache, real h2, DeviceArray<real>& scratch);

// Inversion guard for NH bodies (MS §10.4): smallest θ ∈ (0,1] with det D_s(x + θ dx) = 0 over
// tets whose vertices move, scaled by 0.9; returns 1 if none. Only tets of NH materials are tested.
// all_bodies = true guards every tet (contact.inversion_free = "on"); false guards NH tets only
// ("auto", the paper's rule: the guard exists for non-invertible energies). "off" skips the call.
real fem_inversion_toi(const SceneBuffers& sb, const real3* x, const real3* dx, DeviceArray<real>& scratch,
                       bool all_bodies = false);

// MS §6 guarded Dirichlet advance: the largest theta in (0, 1] such that no guarded, moving tet falls below
// eta * f(0) on the way, f(t) = oriented det D_s(x + t dx) (first root of f(t) = eta f(0) per tet); 1 if none.
// With x_ref the bound is eta * f at x_ref (the start-of-step state) instead of eta * f(0), so repeated
// advances cannot compound; a tet already below the bound at theta = 0 does not constrain the step.
real fem_volume_loss_toi(const SceneBuffers& sb, const real3* x, const real3* dx, double eta, DeviceArray<real>& scratch,
                         bool all_bodies = false, const real3* x_ref = nullptr);

// MS §10.4 end-state rule for the anchor blend: min over the guarded, moving tets with f(0) > 0 of
// f(theta) / min(f(0), f(1)) (f(1) <= 0: / f(0)), f(t) = oriented det D_s(x + t dx). The state
// x + theta dx is accepted when this margin is >= eta_J. Returns +inf when no tet is tested.
double fem_inversion_margin(const SceneBuffers& sb, const real3* x, const real3* dx, double theta,
                            DeviceArray<real>& scratch, bool all_bodies = false);

// A.blocks[slot] += h2 * Σ_contrib V_e (H_e^+)_{ab} for every upper slot of the pattern's gather
// map, and the transpose into the mirror slot (Appendix B closed form from the cache's U, V,
// eig, q). Deterministic (fixed contribution order), no atomics. Does NOT zero A first.
void fem_assemble_bsr(const SceneBuffers& sb, const ElementCache& cache, real h2, struct BsrMatrix& A);
// The M1 kernel (one thread per upper slot, every contribution re-derived from the cache), kept
// as the reference of the two-phase assembly above; tests only.
void fem_assemble_bsr_direct(const SceneBuffers& sb, const ElementCache& cache, real h2, struct BsrMatrix& A);
// Tets per chunk of the two-phase assembly's staging buffer (720 B per tet).
int fem_assembly_chunk_tets(int n_tets);
// The two-phase assembly with the two phases timed by device events (ms, summed over chunks); tests only.
void fem_assemble_bsr_timed(const SceneBuffers& sb, const ElementCache& cache, real h2, struct BsrMatrix& A,
                            float* ms_phase_a, float* ms_phase_b);
// The two-phase assembly with an explicit chunk length (tets per staging pass); tests only.
void fem_assemble_bsr_chunked(const SceneBuffers& sb, const ElementCache& cache, real h2, struct BsrMatrix& A,
                              int chunk_tets);

}  // namespace cs

#include "linsys/bsr.cuh"
