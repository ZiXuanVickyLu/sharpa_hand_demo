#!/usr/bin/env bash
# Records where the animal-well benchmark assets came from (they are included here, see LICENSE):
# the libuipc AL-release tree. Re-fetching needs a checkout of https://github.com/wiso-enoji/libuipc
# (branch AL-release) at UIPC_DIR with the branch available as `al/AL-release`.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UIPC="${UIPC_DIR:-$HERE/../../../libuipc}"
REF="${UIPC_REF:-al/AL-release}"
git -C "$UIPC" show "$REF:assets/sim_data/tetmesh/animal_well.msh" > "$HERE/animal_well.msh"
git -C "$UIPC" show "$REF:assets/sim_data/trimesh/pool.obj" > "$HERE/pool.obj"
git -C "$UIPC" show "$REF:assets/sim_data/trimesh/pool1.obj" > "$HERE/pool1.obj"
ls -la "$HERE"
