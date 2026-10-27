"""Render preview images from the studio cameras.

blender -b bike/wasteland_scrambler.blend --python scripts/render_previews.py -- \
        --cams Cam_Hero,Cam_SideRight --samples 128 --res 1400x1260 --outdir renders [--set speedo_kmh=62]
"""
import bpy
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


cams = arg("--cams", "Cam_Hero").split(",")
samples = int(arg("--samples", "96"))
w, h = (int(v) for v in arg("--res", "1400x1260").split("x"))
outdir = os.path.abspath(arg("--outdir", "renders"))
os.makedirs(outdir, exist_ok=True)
sc = bpy.context.scene
root = bpy.data.objects.get("Bike_Root")
for kv in filter(None, arg("--set", "").split(",")):          # e.g. --set speedo_kmh=62,tacho_rpm=3400
    k, v = kv.split("=")
    root[k] = float(v)
sc.cycles.samples = samples
sc.render.resolution_x, sc.render.resolution_y = w, h
try:
    sc.render.threads_mode = 'AUTO'
except Exception:
    pass
for c in cams:
    sc.camera = bpy.data.objects[c]
    sc.render.filepath = os.path.join(outdir, c.replace("Cam_", "").lower() + ".png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", sc.render.filepath)
