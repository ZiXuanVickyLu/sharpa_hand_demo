# Third-party components

This project's own code, configurations, documents, recordings and clips are under the MIT License (`LICENSE`). The
following components are included under their own terms:

| component | origin | license |
| --- | --- | --- |
| `ext/cnpy` | Carl Rogers, https://github.com/rogersce/cnpy | MIT (`ext/cnpy/LICENSE`) |
| `ext/warp_svd` | NVIDIA Warp `warp/native/svd.h` | Apache-2.0 with Eric Jang's MIT notice (`ext/warp_svd/LICENSE.md`) |
| `ext/HouGeoIO` | bgeo I/O of this project's authors; `extern/houjson` adapted from https://github.com/dkoerner/houio | project license; houjson: no license stated upstream |
| `src/libculbvh` | Jerry Hsu's GPU LBVH, via https://github.com/ZiXuanVickyLu/culbvh | MIT (`src/libculbvh/LICENSE.md`) |
| `src/ccd/libcuGTE` | David Eberly's Geometric Tools distance queries, via https://github.com/ZiXuanVickyLu/cuGTE | Boost-1.0 (`src/ccd/libcuGTE/LICENSE`) |
| `asset/rod/*.msh` | IPC, https://github.com/ipc-sim/IPC | MIT (`asset/rod/LICENSE`) |
| `asset/card_shuffle/card15x7.obj` | Codim-IPC, https://github.com/ipc-sim/Codim-IPC | Apache-2.0 (`asset/card_shuffle/LICENSE`) |
| `asset/animal_well/*` | libuipc, https://github.com/spiriMirror/libuipc (AL-release branch of https://github.com/wiso-enoji/libuipc) | Apache-2.0 (`asset/animal_well/LICENSE`) |
| `asset/trapped_balls/fluffy_ball.mesh`, `cylinder.obj` | the authors' supplementary repository of Zheng, Luo and Li 2025 | no license stated (`asset/trapped_balls/NOTICE`) |
| `asset/robot/yam_sharpa/meshes/yam` | I2RT YAM arm, https://github.com/i2rt-robotics/i2rt | MIT (`asset/robot/yam_sharpa/README.md`) |
| `asset/robot/yam_sharpa/meshes/{left,right}_hand` | Sharpa Wave hand, https://github.com/sharpa-robotics/sharpa-urdf-usd-xml | see that repository |

The paper the solver implements (Zheng, Luo, Li 2025, arXiv:2512.12151) is cited in `doc/references.md`; its PDF is not
part of the repository.
