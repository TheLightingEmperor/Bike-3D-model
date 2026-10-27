#!/usr/bin/env bash
# Full pipeline: build -> preview renders -> bake + game exports.
# Usage: BLENDER=/path/to/blender scripts/make_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BLENDER="${BLENDER:-blender}"
BLEND=bike/wasteland_scrambler.blend

"$BLENDER" -b --factory-startup --python scripts/build_bike.py -- --out "$PWD/$BLEND"
"$BLENDER" -b "$BLEND" --python scripts/render_previews.py -- \
    --cams Cam_Hero --samples 64 --res 1400x1260 --outdir renders
"$BLENDER" -b "$BLEND" --python scripts/render_previews.py -- \
    --cams Cam_SideRight,Cam_RearThreeQuarter,Cam_FrontThreeQuarterLeft,Cam_EngineDetail \
    --samples 40 --res 1100x990 --outdir renders
"$BLENDER" -b "$BLEND" --python scripts/export_game_assets.py -- \
    --outdir exports --res "${TEX_RES:-2048}" --samples 16
