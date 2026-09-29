#pragma once
// The fused CG solve for small systems (implementation spec §12.3 item 14): one cooperative
// kernel runs the whole preconditioned CG loop, the operator apply (BSR, contact, friction,
// bending) and the reductions included, with grid-wide barriers between the phases of an
// iteration instead of kernel boundaries. One launch and one readback per solve. Used when the
// system is small enough for the grid to be co-resident (linear_solver.fused_max_rows).
#ifndef CS_PCG_FUSED_CUH
#define CS_PCG_FUSED_CUH

#include "core/typedef.cuh"
#include "contact/contact_spmv_view.cuh"
#include "fem/bending_spmv_view.cuh"

namespace cs {

struct FusedViews {
    int n_rows = 0;
    const int* row_ptr = nullptr;
    const int* col_idx = nullptr;
    const real* blocks = nullptr;
    bool has_contact = false;
    ContactSpmvView contact;
    bool has_bending = false;
    BendingSpmvView bending;
};

}  // namespace cs

#endif  // CS_PCG_FUSED_CUH
