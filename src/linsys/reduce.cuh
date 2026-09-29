#pragma once
// Deterministic device reductions (fixed-order two-level block sums, double accumulators)
// shared by the energy, PCG and statistics code (precision policy §11.1). Every function
// synchronizes and returns the value to the host.
#include "core/typedef.cuh"
#include "core/device_buffer.cuh"

namespace cs {

// scratch is resized as needed (kept between calls by the owner).
double reduce_sum(const real* a, int n, DeviceArray<double>& scratch);
// Same sum written to the device double at d_out, no host synchronisation (the line search
// collects its energy parts this way and reads them back once, IS §12.3 item 10). n <= 0
// writes 0 and never reads a.
void reduce_sum_to(const real* a, int n, DeviceArray<double>& scratch, double* d_out);
double reduce_dot(const real3* a, const real3* b, int n, DeviceArray<double>& scratch);
double reduce_sum_sq(const real3* a, int n, DeviceArray<double>& scratch);
double reduce_max(const real* a, int n, DeviceArray<double>& scratch);
double reduce_min(const real* a, int n, DeviceArray<double>& scratch);
// max_i max_c |a_i.c|
double reduce_max_abs(const real3* a, int n, DeviceArray<double>& scratch);
// max_i |a_i|_inf where a_i = x_i - y_i
double reduce_max_abs_diff(const real3* x, const real3* y, int n, DeviceArray<double>& scratch);

}  // namespace cs
