# References

The paper is not part of the repository (arXiv's license does not cover redistribution); this file records
the references.

| Paper | Reference |
| --- | --- |
| `Zheng et al. - 2025 - Robust and Efficient Penetration-Free Elastodynamics without Barriers.pdf` | J. Zheng, Z. Luo, M. Li. *Robust and Efficient Penetration-Free Elastodynamics without Barriers.* arXiv:2512.12151, Dec 2025. The method reproduced here ("AL-IPC" / barrier-free contact). |

## Code references

| Reference | What it is | Used for |
| --- | --- | --- |
| https://github.com/spiriMirror/libuipc and the AL fork https://github.com/wiso-enoji/libuipc (branch `AL-release`) | Official AL-IPC integration contributed by Genesis AI | The [UIPC-AL] reference implementation the specs compare against; the animal-well assets |
| a private in-house PD/ADMM + MPM coupling solver (not published) | Infrastructure reference | CMake layout and presets, CUDA 13 CCCL handling, real types, the LBVH and ACCD sources it carried |
| https://github.com/ZiXuanVickyLu/culbvh | Stand-alone LBVH (Jerry Hsu's GPU LBVH) | `src/libculbvh` |
| https://github.com/ZiXuanVickyLu/cuGTE | Geometric Tools distance queries on CUDA | `src/ccd/libcuGTE` |

Paths of the form `../libuipc` or `../coupled_solver` in the specs refer to the author's local checkouts of these
at the time of writing.

## Papers cited by the specs (not stored locally)

* M. Li et al. *Incremental Potential Contact.* ACM TOG 39(4), 2020 — friction model, barrier baseline.
* M. Li, D. Kaufman, C. Jiang. *Codimensional Incremental Potential Contact.* ACM TOG 40(4), 2021 — additive CCD (ACCD).
* B. Smith, F. De Goes, T. Kim. *Stable Neo-Hookean Flesh Simulation.* ACM TOG 37(2), 2018 — energy and analytic eigen-system.
* B. Smith, F. De Goes, T. Kim. *Analytic Eigensystems for Isotropic Distortion Energies.* ACM TOG 38(1), 2019 — twist/flip/scaling modes.
* J. Nocedal, S. Wright. *Numerical Optimization*, 2nd ed., 2006 — augmented Lagrangian theory (Thm. 17.6).
* T. Karras. *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees.* HPG 2012 — LBVH.
* K. Huang et al. *GIPC* (TOG 43(2), 2024) and *StiffGIPC* (TOG 44(3), 2025) — GPU IPC baselines, symmetric BSR SpMV.
* D. Doyen, A. Ern, S. Piperno. *Time-integration schemes for the finite element dynamic Signorini problem.* SISC 33(1), 2011 — 1D analytic contact benchmarks.
