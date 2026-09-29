#pragma once
// Device-side active-set union and friction-snapshot compaction (implementation spec §6.10
// step 7, §12.3 item 4). Replaces the host merge that M1 shipped as a stopgap: at 1.7 M pairs
// that merge moved 77 M pair records through the CPU per step (29% of the compressor's step).
//
// Determinism: every pass is a deterministic thrust primitive (stable select, radix sort,
// unique, merge-path set union) on unique 64-bit keys, plus one gather kernel whose only
// atomics are integer type counters. The result is bitwise identical to the host merge it
// replaces for the same inputs (test_active_set_union checks this against a host reference).
#ifndef CS_ACTIVE_SET_CUH
#define CS_ACTIVE_SET_CUH

#include "core/device_buffer.cuh"
#include "core/typedef.cuh"

#include <cstddef>
#include <vector>

namespace cs {

struct UnionResult {
    int n_new = 0;      // |C| after the union
    int removed = 0;    // entries of C dropped (keep_flag == 0)
    int kept = 0;       // distinct hits admitted (after cand_keep and duplicate removal)
    int counts[4] = {0, 0, 0, 0};   // |C| by pair type value 0..3 after the union
};

/// Temporary storage for thrust's algorithms: one device pool handed out bump-style, reset
/// once per union. Thrust allocates and frees temporaries inside every call; without a pool
/// that is four cudaMalloc/cudaFree pairs per outer iteration, which the small scenes feel.
class DevicePool {
public:
    using value_type = char;
    DevicePool() = default;
    DevicePool(const DevicePool&) = delete;
    DevicePool& operator=(const DevicePool&) = delete;
    ~DevicePool();
    char* allocate(std::ptrdiff_t n);
    void deallocate(char* p, std::size_t n);
    void reset();   // frees any overflow blocks, grows the pool to the last cycle's total demand
    std::size_t capacity() const { return pool_.capacity(); }

private:
    void free_overflow() noexcept;
    DeviceArray<char> pool_;
    std::size_t offset_ = 0;       // total demand so far in this cycle (bump pointer while it fits)
    std::size_t high_water_ = 0;   // largest cycle demand seen
    std::vector<char*> overflow_;  // one-off blocks handed out once the pool was exhausted
};

/// C (n_pairs entries, sorted by unique key; entry i survives when keep_flag[i] != 0) united
/// with the kept hits (cand_* entries j with cand_keep[j] != 0; any order, duplicates and
/// keys already in C allowed). On return the six arrays of C hold the union sorted by key:
/// surviving entries keep their λ and n, new entries get λ = 0 and n = 0, and a key present in
/// both takes C's record. The arrays are swapped with internal scratch, so their capacity
/// grows geometrically and no allocation happens in the steady state.
class ActiveSetUnion {
public:
    UnionResult unite(int n_pairs, DeviceArray<unsigned long long>& key, DeviceArray<unsigned char>& type,
                      DeviceArray<int4>& idx, DeviceArray<real>& xi, DeviceArray<real>& lambda,
                      DeviceArray<int>& ninact, const int* keep_flag, int hits,
                      const unsigned long long* cand_key, const unsigned char* cand_type, const int4* cand_idx,
                      const real* cand_xi, const int* cand_keep);

private:
    DevicePool pool_;
    DeviceArray<int> sel_a_, sel_b_, val_out_, counts_;
    DeviceArray<unsigned long long> key_a_, key_b_, key_out_;
    // the "other" set of pair arrays, swapped with the caller's after each union
    DeviceArray<unsigned long long> key2_;
    DeviceArray<unsigned char> type2_;
    DeviceArray<int4> idx2_;
    DeviceArray<real> xi2_, lambda2_;
    DeviceArray<int> ninact2_;
};

/// Friction snapshot compaction (MS §13): the pairs with keep[i] != 0, in order, copied into
/// fr_* (resized to the count). Returns the count. `sel` is caller-owned scratch.
int compact_friction(int n_pairs, const int* keep, const int4* idx, const real4* weight, const real3* basis,
                     const real* force, DeviceArray<int4>& fr_idx, DeviceArray<real4>& fr_weight,
                     DeviceArray<real3>& fr_basis, DeviceArray<real>& fr_force, DeviceArray<int>& sel,
                     DevicePool& pool);

}  // namespace cs

#endif  // CS_ACTIVE_SET_CUH
