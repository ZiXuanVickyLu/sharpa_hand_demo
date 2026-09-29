#!/usr/bin/env bash
# Fetches the animal-well benchmark assets from the libuipc AL-release tree (not committed
# here: the tet mesh is 71 MB). Requires the sibling checkout ../../../libuipc with the
# fork branch available as `al/AL-release` (see doc/references.md).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
UIPC="${UIPC_DIR:-$HERE/../../../libuipc}"
REF="${UIPC_REF:-al/AL-release}"
git -C "$UIPC" show "$REF:assets/sim_data/tetmesh/animal_well.msh" > "$HERE/animal_well.msh"
git -C "$UIPC" show "$REF:assets/sim_data/trimesh/pool.obj" > "$HERE/pool.obj"
git -C "$UIPC" show "$REF:assets/sim_data/trimesh/pool1.obj" > "$HERE/pool1.obj"
ls -la "$HERE"
