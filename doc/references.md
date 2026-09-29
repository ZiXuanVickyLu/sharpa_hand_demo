# References

The PDFs live in `knowledge/`, which is gitignored (PDFs do not delta-compress; committing them
would add ~13 MB per paper to every clone permanently). This file records what they are.

| File in `knowledge/` | Reference |
| --- | --- |
| `Zheng et al. - 2025 - Robust and Efficient Penetration-Free Elastodynamics without Barriers.pdf` | J. Zheng, Z. Luo, M. Li. *Robust and Efficient Penetration-Free Elastodynamics without Barriers.* arXiv:2512.12151, Dec 2025. The method reproduced here ("AL-IPC" / barrier-free contact). |

## Code references (local checkouts)

| Path | What it is | Used for |
| --- | --- | --- |
| `../libuipc` (remote `origin` = spiriMirror/libuipc, remote `al` = wiso-enoji/libuipc, branch `al/AL-release`) | Official AL-IPC integration contributed by Genesis AI (`src/backends/cuda/active_set_system`, `contact_system/al_*`, `engine/advance_al.cu`, `apps/AL_examples`) | Operational order of the outer loop, active-set merge/filter kernels, slack/λ update, friction snapshot; see math spec §16 for where we deviate |
| `../coupled_solver` | In-house PD/ADMM + MPM coupling solver | CMake layout and presets, CUDA 13 CCCL handling, `typedef.cuh` real types, `libculbvh` stackless LBVH, ACCD + GTE distance queries, buffer pool pattern, PCG structure, bgeo I/O, scene JSON conventions |
| `../culbvh` | Stand-alone LBVH repo (same code as `coupled_solver/src/libculbvh`) | Reference and benchmarks |
| `../cwheels` | CPU solver wheels (IPC/PD solvers, ipc-toolkit vendored under `ext/ipc-toolkit`) | CPU oracles for distance gradients and elastic eigen-systems in tests |

## Papers cited by the specs (not stored locally)

* M. Li et al. *Incremental Potential Contact.* ACM TOG 39(4), 2020 — friction model, barrier baseline.
* M. Li, D. Kaufman, C. Jiang. *Codimensional Incremental Potential Contact.* ACM TOG 40(4), 2021 — additive CCD (ACCD).
* B. Smith, F. De Goes, T. Kim. *Stable Neo-Hookean Flesh Simulation.* ACM TOG 37(2), 2018 — energy and analytic eigen-system.
* B. Smith, F. De Goes, T. Kim. *Analytic Eigensystems for Isotropic Distortion Energies.* ACM TOG 38(1), 2019 — twist/flip/scaling modes.
* J. Nocedal, S. Wright. *Numerical Optimization*, 2nd ed., 2006 — augmented Lagrangian theory (Thm. 17.6).
* T. Karras. *Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees.* HPG 2012 — LBVH.
* K. Huang et al. *GIPC* (TOG 43(2), 2024) and *StiffGIPC* (TOG 44(3), 2025) — GPU IPC baselines, symmetric BSR SpMV.
* D. Doyen, A. Ern, S. Piperno. *Time-integration schemes for the finite element dynamic Signorini problem.* SISC 33(1), 2011 — 1D analytic contact benchmarks.
