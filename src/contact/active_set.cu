// Device-side active-set union (implementation spec §6.10 step 7). See active_set.cuh.
#include "contact/active_set.cuh"
#include "core/cuda_check.h"

#ifdef CCCL_VERSION_GREATER_EQUAL_13_0
#include <cccl/thrust/copy.h>
#include <cccl/thrust/device_ptr.h>
#include <cccl/thrust/iterator/counting_iterator.h>
#include <cccl/thrust/set_operations.h>
#include <cccl/thrust/sort.h>
#include <cccl/thrust/transform.h>
#include <cccl/thrust/system/cuda/execution_policy.h>
#include <cccl/thrust/unique.h>
#else
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>
#endif

#include <algorithm>

namespace cs {

namespace {

constexpr int kBlock = 256;
inline unsigned grid_for(int n) { return unsigned((n + kBlock - 1) / kBlock); }

struct NonZero {
    __host__ __device__ bool operator()(int f) const { return f != 0; }
};

struct AddOffset {
    int offset;
    __host__ __device__ int operator()(int j) const { return offset + j; }
};

__global__ void k_gather_keys(int n, const int* __restrict__ sel, const unsigned long long* __restrict__ src,
                              unsigned long long* __restrict__ dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[sel[i]];
}

// Writes the united set from the tagged sources: v < n_old is an index into the old C, v >= n_old
// is n_old + an index into the hit arrays. Type counts by shared-memory partials and one integer
// atomic per block and type (order-independent).
__global__ void k_gather_union(int n, const int* __restrict__ v, int n_old,
                               const unsigned long long* __restrict__ ok, const unsigned char* __restrict__ ot,
                               const int4* __restrict__ oi, const real* __restrict__ ox, const real* __restrict__ ol,
                               const int* __restrict__ on, const unsigned long long* __restrict__ ck,
                               const unsigned char* __restrict__ ct, const int4* __restrict__ ci,
                               const real* __restrict__ cx, unsigned long long* __restrict__ nk,
                               unsigned char* __restrict__ nt, int4* __restrict__ ni, real* __restrict__ nx,
                               real* __restrict__ nl, int* __restrict__ nn, int* __restrict__ counts) {
    __shared__ int sc[4];
    if (threadIdx.x < 4) sc[threadIdx.x] = 0;
    __syncthreads();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int s = v[i];
        unsigned char ty;
        if (s < n_old) {
            nk[i] = ok[s]; ty = ot[s]; ni[i] = oi[s]; nx[i] = ox[s]; nl[i] = ol[s]; nn[i] = on[s];
        } else {
            s -= n_old;
            nk[i] = ck[s]; ty = ct[s]; ni[i] = ci[s]; nx[i] = cx[s]; nl[i] = real(0); nn[i] = 0;
        }
        nt[i] = ty;
        atomicAdd(&sc[ty & 3], 1);
    }
    __syncthreads();
    if (threadIdx.x < 4 && sc[threadIdx.x]) atomicAdd(&counts[threadIdx.x], sc[threadIdx.x]);
}

__global__ void k_gather_friction(int n, const int* __restrict__ sel, const int4* __restrict__ idx,
                                  const real4* __restrict__ weight, const real3* __restrict__ basis,
                                  const real* __restrict__ force, int4* __restrict__ fi, real4* __restrict__ fw,
                                  real3* __restrict__ fb, real* __restrict__ ff) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int s = sel[i];
    fi[i] = idx[s];
    fw[i] = weight[s];
    fb[2 * i + 0] = basis[2 * s + 0];
    fb[2 * i + 1] = basis[2 * s + 1];
    ff[i] = force[s];
}

template <typename T>
thrust::device_ptr<T> dp(T* p) { return thrust::device_pointer_cast(p); }

// Stable select of the indices i in [0, n) with flag[i] != 0 into sel (resized to n). Returns
// the count.
template <typename Policy>
int select_indices(const Policy& policy, int n, const int* flag, DeviceArray<int>& sel) {
    if (n <= 0) return 0;
    sel.loose_resize(size_t(n));
    auto end = thrust::copy_if(policy, thrust::make_counting_iterator(0), thrust::make_counting_iterator(n),
                               dp(flag), dp(sel.data()), NonZero{});
    return int(end - dp(sel.data()));
}

}  // namespace

// ---- DevicePool ---------------------------------------------------------------------------

char* DevicePool::allocate(std::ptrdiff_t n) {
    const std::size_t want = (std::size_t(n) + 255) & ~std::size_t(255);
    const std::size_t start = offset_;
    // offset_ counts the cycle's whole demand whether or not a request fits, so that the
    // next reset() sizes the pool to the sum of the requests, not to the largest one.
    offset_ += want;
    high_water_ = std::max(high_water_, offset_);
    if (offset_ <= pool_.capacity()) return pool_.data() + start;
    // Pool exhausted: a one-off block, freed at the next reset (by then the pool has grown).
    char* p = nullptr;
    CS_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(want, 1)));
    overflow_.push_back(p);
    return p;
}

void DevicePool::deallocate(char*, std::size_t) {}

void DevicePool::free_overflow() noexcept {
    for (char* p : overflow_) (void)cudaFree(p);
    overflow_.clear();
}

DevicePool::~DevicePool() { free_overflow(); }

void DevicePool::reset() {
    free_overflow();
    if (high_water_ > pool_.capacity()) pool_.reserve(high_water_ + high_water_ / 4);
    offset_ = 0;
}

// ---- ActiveSetUnion -----------------------------------------------------------------------

UnionResult ActiveSetUnion::unite(int n_pairs, DeviceArray<unsigned long long>& key, DeviceArray<unsigned char>& type,
                                  DeviceArray<int4>& idx, DeviceArray<real>& xi, DeviceArray<real>& lambda,
                                  DeviceArray<int>& ninact, const int* keep_flag, int hits,
                                  const unsigned long long* cand_key, const unsigned char* cand_type,
                                  const int4* cand_idx, const real* cand_xi, const int* cand_keep) {
    UnionResult r;
    pool_.reset();
    const auto policy = thrust::cuda::par(pool_);

    // 1. Survivors of C, in order (a stable select keeps them sorted).
    const int na = select_indices(policy, n_pairs, keep_flag, sel_a_);
    CS_SYNC_POINT("union select survivors");
    r.removed = n_pairs - na;
    if (na) {
        key_a_.loose_resize(size_t(na));
        k_gather_keys<<<grid_for(na), kBlock>>>(na, sel_a_.data(), key.data(), key_a_.data());
        CS_CUDA_KERNEL_CHECK();
    }

    // 2. Kept hits: select, sort by key (stable radix), drop duplicate keys.
    int nb = select_indices(policy, hits, cand_keep, sel_b_);
    if (nb) {
        key_b_.loose_resize(size_t(nb));
        k_gather_keys<<<grid_for(nb), kBlock>>>(nb, sel_b_.data(), cand_key, key_b_.data());
        CS_CUDA_KERNEL_CHECK();
        thrust::sort_by_key(policy, dp(key_b_.data()), dp(key_b_.data() + nb), dp(sel_b_.data()));
        CS_SYNC_POINT("union sort_by_key");
        auto e = thrust::unique_by_key(policy, dp(key_b_.data()), dp(key_b_.data() + nb), dp(sel_b_.data()));
        nb = int(e.first - dp(key_b_.data()));
        CS_SYNC_POINT("union unique_by_key");
        // tag the hit sources so one value array addresses both inputs
        thrust::transform(policy, dp(sel_b_.data()), dp(sel_b_.data() + nb), dp(sel_b_.data()), AddOffset{n_pairs});
    }
    r.kept = nb;

    // 3. Merge-path union; on a key present in both, thrust::set_union keeps the first range's
    //    element (C's record), which is the host merge's rule.
    const int cap = std::max(1, na + nb);
    key_out_.loose_resize(size_t(cap));
    val_out_.loose_resize(size_t(cap));
    int n_new = 0;
    if (na + nb) {
        auto e = thrust::set_union_by_key(policy, dp(key_a_.data()), dp(key_a_.data() + na), dp(key_b_.data()),
                                          dp(key_b_.data() + nb), dp(sel_a_.data()), dp(sel_b_.data()),
                                          dp(key_out_.data()), dp(val_out_.data()));
        n_new = int(e.first - dp(key_out_.data()));
        CS_SYNC_POINT("union set_union_by_key");
    }
    r.n_new = n_new;

    // 4. Gather the united records into the scratch set and swap it in.
    key2_.loose_resize(size_t(std::max(1, n_new)));
    type2_.loose_resize(size_t(std::max(1, n_new)));
    idx2_.loose_resize(size_t(std::max(1, n_new)));
    xi2_.loose_resize(size_t(std::max(1, n_new)));
    lambda2_.loose_resize(size_t(std::max(1, n_new)));
    ninact2_.loose_resize(size_t(std::max(1, n_new)));
    counts_.resize(4);
    counts_.zero();
    if (n_new) {
        k_gather_union<<<grid_for(n_new), kBlock>>>(
            n_new, val_out_.data(), n_pairs, key.data(), type.data(), idx.data(), xi.data(), lambda.data(),
            ninact.data(), cand_key, cand_type, cand_idx, cand_xi, key2_.data(), type2_.data(), idx2_.data(),
            xi2_.data(), lambda2_.data(), ninact2_.data(), counts_.data());
        CS_CUDA_KERNEL_CHECK();
        counts_.download(r.counts, 4);
    }
    key.swap(key2_); type.swap(type2_); idx.swap(idx2_);
    xi.swap(xi2_); lambda.swap(lambda2_); ninact.swap(ninact2_);
    key.resize(size_t(n_new)); type.resize(size_t(n_new)); idx.resize(size_t(n_new));
    xi.resize(size_t(n_new)); lambda.resize(size_t(n_new)); ninact.resize(size_t(n_new));
    return r;
}

// ---- friction compaction ------------------------------------------------------------------

int compact_friction(int n_pairs, const int* keep, const int4* idx, const real4* weight, const real3* basis,
                     const real* force, DeviceArray<int4>& fr_idx, DeviceArray<real4>& fr_weight,
                     DeviceArray<real3>& fr_basis, DeviceArray<real>& fr_force, DeviceArray<int>& sel,
                     DevicePool& pool) {
    pool.reset();
    const int n_fr = select_indices(thrust::cuda::par(pool), n_pairs, keep, sel);
    if (!n_fr) return 0;
    fr_idx.loose_resize(size_t(n_fr));
    fr_weight.loose_resize(size_t(n_fr));
    fr_basis.loose_resize(size_t(2) * n_fr);
    fr_force.loose_resize(size_t(n_fr));
    k_gather_friction<<<grid_for(n_fr), kBlock>>>(n_fr, sel.data(), idx, weight, basis, force, fr_idx.data(),
                                                   fr_weight.data(), fr_basis.data(), fr_force.data());
    CS_CUDA_KERNEL_CHECK();
    return n_fr;
}

}  // namespace cs
