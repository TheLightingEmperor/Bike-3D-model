#!/usr/bin/env bash
# Full pipeline: build -> preview renders -> bake + game exports.
# Usage: BLENDER=/path/to/blender scripts/make_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BLENDER="${BLENDER:-blender}"
BLEND=bike/wasteland_scrambler.blend

python3 scripts/make_textures.py
"$BLENDER" -b --factory-startup --python scripts/build_bike.py -- --out "$PWD/$BLEND"
"$BLENDER" -b "$BLEND" --python scripts/check_mechanics.py -- --static --lines
"$BLENDER" -b "$BLEND" --python scripts/render_previews.py -- \
    --cams Cam_Hero --samples 64 --res 1400x1260 --outdir renders
"$BLENDER" -b "$BLEND" --python scripts/render_previews.py -- \
    --cams Cam_SideRight,Cam_RearThreeQuarter,Cam_FrontThreeQuarterLeft,Cam_EngineDetail,Cam_HeadStock,Cam_LeftDetail,Cam_RearDetail \
    --samples 40 --res 1100x990 --outdir renders
"$BLENDER" -b "$BLEND" --python scripts/render_previews.py -- \
    --cams Cam_Cockpit,Cam_Gauges --samples 40 --res 1100x990 --outdir renders --set speedo_kmh=62,tacho_rpm=3400,steer=-12
EXPORT_DIR="${EXPORT_DIR:-exports}"
"$BLENDER" -b "$BLEND" --python scripts/export_game_assets.py -- \
    --outdir "$EXPORT_DIR" --res "${TEX_RES:-2048}" --samples 16
if [ "$EXPORT_DIR" != "exports" ]; then mkdir -p exports && cp -r "$EXPORT_DIR"/. exports/; fi
