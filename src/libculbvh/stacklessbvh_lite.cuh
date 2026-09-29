#pragma once
// birdpeople, 2025, 11/28, implementation of stacklessbvh_lite.

#include <memory>
#include <string>

#include "../core/typedef.h"
#include "../core/vector_type_t.h"
#include "bound.h"
#ifndef CCCL_VERSION_GREATER_EQUAL_13_0
#include "thrust/device_vector.h"
#include "thrust/device_ptr.h"
#else
#include <cccl/thrust/device_vector.h>
#include <cccl/thrust/device_ptr.h>
#endif
#include <vector>

namespace culbvh {
	/// <summary>
	/// Very simple high-performance GPU StacklessLBVH that takes in a list of bounding boxes and outputs overlapping pairs.
	/// Side note: null bounds (inf, -inf) as inputs are ignored automatically.
	/// This implementation is not highly optimized, but serves as a reference for implementing the stackless traversal and build. I have add enough comment for understanding the algorithm.
	/// This implementation can fully pass the simulation dataset test.
	/// </summary>
	template<typename T>
	struct alignas(64) LBVHNode {
		Bound<T> bounds[2];  // Bounds for left and right children
		uint32_t leftIdx;    // Index of left child
		uint32_t rightIdx;   // Index of right child
		uint32_t parentIdx;  // Index of parent node
		uint32_t fence;      // Used for range queries
	};

	// 32-byte aligned node for stackless traversal
	struct __align__(32) StacklessNode {
		Bound<float> bound;
		int data;   // -1 for internal, primitive index for leaf
		int escape; // Index of next node to visit on miss/finish
	};

	class LBVHStacklessLite {
	public:
		using aabb = Bound<float>;
		using vec_type = float3;

		LBVHStacklessLite();
		~LBVHStacklessLite();

		// root bounds of every node in this tree, one large bv (as of the last compute() /
		// refit_structure(); refit() does not update it).
		aabb bounds() const;

		bool is_valid() const { return numObjs > 0; }

		size_t size() const { return numObjs; }

		/// Number of leaf boxes the tree was built over (0 before compute()).
		size_t num_objects() const { return numObjs; }

		/// The caller's box array captured at compute(). The tree never copies it: the caller
		/// keeps that buffer alive for the lifetime of the tree and writes new boxes into it
		/// in place before calling refit() (bounds only) or refit_structure() (full rebuild).
		const aabb* leaf_aabbs() const;

		const thrust::device_vector<LBVHNode<float>>& internal_nodes() const;

		const thrust::device_ptr<aabb>& object_aabbs() const;

		/// <summary>
		/// Refits an existing aabb tree once all buffer has been allocated.
		/// Recompute the tree structure and refit the AABBs.
		/// </summary>
		void refit_structure();

		/// <summary>
		/// Refits an existing aabb tree once compute() has been called.
		/// Does not recompute the tree structure but only the AABBs: the leaf boxes are re-read
		/// from the buffer passed to compute() (write the new boxes there in place first), merged
		/// bottom-up and the stackless nodes are rewritten. No sort, no host synchronisation.
		/// Queries stay exact for arbitrary box changes; traversal efficiency degrades with the
		/// distance the boxes moved, so call refit_structure() once per step and refit() in between.
		/// </summary>
		void refit();

		/// <summary>
		/// Allocates memory and builds the LBVH from a list of AABBs.
		/// Can be called multiple times for memory reuse.
		/// </summary>
		/// <param name="devicePtr">The device pointer containing the AABBs</param>
		/// <param name="size">The number of AABBs</param>
		void compute(aabb* devicePtr, size_t size);

		/// <summary>
		/// Tests this BVH against another BVH. Outputs unique collision pairs.
		/// The calling BVH should be the smaller one for best performance. 
		/// </summary>
		/// <param name="d_res">Device pointer with pairs containing (in order) the calling BVH object ID and then the other BVH object ID.</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <param name="d_otherAABBs">The other BVH AABBs</param>
		/// <param name="otherSize">The number of other BVH AABBs</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t otherSize) const;

		/// <summary>
		/// Tests this BVH against itself. Outputs unique collision pairs: every unordered pair
		/// of overlapping leaves is written exactly once, as (x, y) with x < y.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs (x < y)</param>
		/// <param name="resSize">The number of entries allocated</param>
		/// <returns>The number of unique collision pairs written</returns>
		size_t query(int2* d_res, size_t resSize) const;

		/// Chunked queries (contact_solver implementation spec §12.3 item 15). The query objects
		/// are visited in their Morton order; a chunk is the range [q0, q1) of that order, so a
		/// caller can bound the result buffer per chunk instead of holding every candidate at
		/// once. The self query emits each unordered pair exactly once whichever chunk holds
		/// the larger-index object, so chunking never duplicates or drops a pair.
		size_t query_range(int2* d_res, size_t resSize, size_t q0, size_t q1) const;
		/// Sorts the other objects into Morton order once; query_other_range then runs the
		/// range [q0, q1) of that order against this tree.
		void prepare_other(aabb* d_otherAABBs, size_t otherSize) const;
		size_t query_other_range(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t q0, size_t q1) const;
		/// The same two queries without any host synchronisation: the pair count of the range
		/// is accumulated into *d_count (which the caller zeroes beforehand) and never read here.
		/// A count >= resSize means the buffer overflowed and the range must be redone smaller.
		void query_range_async(int2* d_res, size_t resSize, size_t q0, size_t q1, int* d_count) const;
		void query_other_range_async(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t q0, size_t q1, int* d_count) const;

		/// <summary>
		/// Tests this BVH query using a ground truth method(cpu brute force). Outputs unique collision pairs.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs, should be filled with the result of query()!!!</param>
		/// <param name="resSize">The number of collision pairs</param>
		/// <returns>True if the query is correct, false otherwise</returns>
		bool query_compare_ground_truth(int2* d_res, size_t resSize) const;

		/// <summary>
		/// Tests this BVH query against another BVH using a ground truth method(cpu brute force). Outputs unique collision pairs.
		/// </summary>
		/// <param name="d_res">Device pointer with unique object ID pairs, should be filled with the result of query()!!!</param>
		/// <param name="resSize">The number of collision pairs</param>
		/// <param name="d_otherAABBs">The other BVH AABBs</param>
		/// <param name="otherSize">The number of other BVH AABBs</param>
		/// <returns>True if the query is correct, false otherwise</returns>
		bool query_compare_ground_truth(int2* d_res, size_t resSize, aabb* d_otherAABBs, size_t otherSize) const;

		// Does a self check of the BVH structure for debugging purposes.
		void bvhSelfCheck() const;

	private:
		struct thrustImpl;
		std::unique_ptr<thrustImpl> impl;
		aabb rootBounds;
		size_t numObjs {0};
	};

	// Tests the LBVH with a simple test case of 100k objects.
	void testLBVHStacklessLite();
}
