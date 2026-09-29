#!/usr/bin/env bash
# Fetches the squishy-ball asset of Zheng, Luo & Li 2025 (Figures 1 and 21) from the authors'
# supplementary repository. fluffy_ball.mesh is 33 MB and gitignored here.
#   github.com/wiso-enoji/Barrier-Free-Supplementary  ->  assets.zip  ->  assets/fluffy_ball.mesh
# The file is in MEDIT format in the authors' raw units (half-extent 17.95); the configs apply
# the paper's scale (0.03 for Fig. 1, 0.027 for Fig. 21) themselves. Apply it exactly once.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 https://github.com/wiso-enoji/Barrier-Free-Supplementary.git "$TMP/supp"
unzip -q -o "$TMP/supp/assets.zip" -d "$TMP/assets"
cp "$TMP/assets/assets/fluffy_ball.mesh" "$HERE/fluffy_ball.mesh" 2>/dev/null || cp "$TMP/assets/fluffy_ball.mesh" "$HERE/fluffy_ball.mesh"
cp "$TMP/assets/assets/cylinder.obj" "$HERE/cylinder.obj" 2>/dev/null || cp "$TMP/assets/cylinder.obj" "$HERE/cylinder.obj"
ls -la "$HERE"
