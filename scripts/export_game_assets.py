"""
Bake + export a game-ready version of the bike.

    blender -b bike/wasteland_scrambler.blend --python scripts/export_game_assets.py -- \
            --outdir exports --res 2048 --samples 24

What it does (the source .blend is never modified):
  1. Applies every modifier (viewport levels = game LOD), converts curves to meshes,
     removes boolean cutters / array helpers / the studio.
  2. Groups procedural materials into texture sets (Paint, Metal, Rubber), smart-UVs each set
     into one shared atlas with uniform texel density.
  3. Bakes BaseColor / Roughness / Metallic (emission trick) and a tangent-space Normal map
     (captures all procedural bump detail) with Cycles.
  4. Rebuilds glTF-friendly Principled materials from the baked maps. Uniform materials
     (glass, lenses, wires) stay as plain factors.
  5. Exports GLB (with embedded textures), FBX (embedded textures), USD + USDZ, and saves
     a "_game.blend" plus the PNG maps.

  --resume    reuse maps already in <outdir>/textures (the UV unwrap is deterministic), so an
              interrupted bake carries on where it stopped
  --uv-only   unwrap and log a hash of each set's UVs, then quit (determinism check)
  --uv-from   a previous "_game.blend": copy its UVs (per mesh, when the geometry matches) instead of
              unwrapping, so UVs stay stable between versions and --resume can reuse that set's maps
  --reexport  run on a "_game.blend": skip flatten/bake, just write the files again
  --formats   comma list of glb,fbx,usd,usdz,blend to write (default: all)
"""
import bpy
import bmesh
import os
import sys
import time

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default


OUT = os.path.abspath(arg("--outdir", "exports"))
RES = int(arg("--res", "2048"))
SAMPLES = int(arg("--samples", "24"))
NAME = arg("--name", "wasteland_scrambler")
SKIP_BAKE = "--skip-bake" in argv
RESUME = "--resume" in argv
UV_ONLY = "--uv-only" in argv
UV_FROM = arg("--uv-from", None)
REEXPORT = "--reexport" in argv
FORMATS = set(arg("--formats", "glb,fbx,usd,usdz,blend").split(","))
TEX = os.path.join(OUT, "textures")
os.makedirs(TEX, exist_ok=True)

SETS = {
    "Paint": ("M_Paint_",),
    "Metal": ("M_Steel", "M_Pipe", "M_Chrome", "M_Alu", "M_DarkMetal", "M_Hardware", "M_BraidedLine", "M_HeatWrap", "M_Ceramic"),
    "Rubber": ("M_Tire", "M_Rubber", "M_Hose", "M_Seat", "M_Loom"),
}

t0 = time.time()
sc = bpy.context.scene
bike = bpy.data.collections["Bike"]


def log(*a):
    print("[export %6.1fs]" % (time.time() - t0), *a, flush=True)


def viewport_setup():
    """Open on Material Preview framed on the bike: Solid shading hides the textures."""
    for scr in bpy.data.screens:
        for a in scr.areas:
            if a.type != 'VIEW_3D':
                continue
            sp = a.spaces.active
            sp.shading.type = 'MATERIAL'
            if sp.region_3d:
                sp.region_3d.view_location = (0.0, -0.05, 0.62)
                sp.region_3d.view_distance = 3.6


def finish():
    """Write GLB / FBX / USD / USDZ of the Bike collection and a self-contained _game.blend."""
    # point images at <outdir>/textures (where the maps were baked or copied) so relative paths resolve
    for img in bpy.data.images:
        p = os.path.join(TEX, os.path.basename(img.filepath_raw or ""))
        if img.source == 'FILE' and img.filepath_raw and os.path.isfile(p):
            img.filepath = p
    bpy.ops.object.select_all(action='DESELECT')
    for o in bike.all_objects:
        o.select_set(True)
    if "glb" in FORMATS:
        glb = os.path.join(OUT, NAME + ".glb")
        bpy.ops.export_scene.gltf(filepath=glb, export_format='GLB', use_selection=True, export_apply=True,
                                  export_yup=True, export_extras=True, export_cameras=False, export_lights=False,
                                  export_animations=False, export_image_format='AUTO')
        log("GLB", glb, os.path.getsize(glb) // 1024, "KB")
    if "fbx" in FORMATS:
        fbx = os.path.join(OUT, NAME + ".fbx")
        bpy.ops.export_scene.fbx(filepath=fbx, use_selection=True, object_types={'EMPTY', 'MESH'}, apply_unit_scale=True,
                                 apply_scale_options='FBX_SCALE_ALL', path_mode='COPY', embed_textures=True,
                                 add_leaf_bones=False, bake_anim=False, mesh_smooth_type='FACE', use_custom_props=True)
        log("FBX", fbx, os.path.getsize(fbx) // 1024, "KB")
    # no world: the exporter would turn it into a dome light with a generated colour texture
    try:
        if "usd" in FORMATS:
            usd = os.path.join(OUT, NAME + ".usdc")
            bpy.ops.wm.usd_export(filepath=usd, selected_objects_only=True, export_materials=True, export_textures=False,
                                  relative_paths=True, convert_world_material=False, export_lights=False)
            log("USD", usd, os.path.getsize(usd) // 1024, "KB")
        if "usdz" in FORMATS:
            usdz = os.path.join(OUT, NAME + ".usdz")   # self-contained, opens in macOS Quick Look / iOS AR
            bpy.ops.wm.usd_export(filepath=usdz, selected_objects_only=True, export_materials=True, export_textures=True,
                                  convert_world_material=False, export_lights=False)
            log("USDZ", usdz, os.path.getsize(usdz) // 1024, "KB")
    except Exception as e:
        log("USD export skipped:", e)
    if "blend" in FORMATS:
        # pack the maps so the file works on its own, with paths relative to it for unpacking
        for img in bpy.data.images:
            if img.source == 'FILE' and not img.packed_file:
                img.pack()
        viewport_setup()
        game = os.path.join(OUT, NAME + "_game.blend")
        if bpy.data.filepath != game:
            bpy.ops.wm.save_as_mainfile(filepath=game, compress=True, relative_remap=True)
        bpy.ops.file.make_paths_relative()
        bpy.ops.wm.save_mainfile(compress=True)
        log("BLEND", game, os.path.getsize(game) // 1024, "KB")


if REEXPORT:
    finish()
    log("DONE reexport")
    sys.exit(0)


# ------------------------------------------------------------------ 1. flatten the asset
for o in bpy.data.objects:
    o.hide_set(False)
    o.hide_viewport = False
# drivers / constraints evaluated once at rest pose; then strip drivers that would break
dg = bpy.context.evaluated_depsgraph_get()
helpers = [o for o in bike.all_objects if o.name.endswith("_ArrayOffset") or o.name in ("Chain_Path",)]
cutters = list(bpy.data.collections["Cutters"].all_objects)

converted = 0
mesh_cache = {}
for o in list(bike.all_objects):
    if o in cutters or o in helpers or o.type not in {'MESH', 'CURVE'}:
        continue
    if o.type == 'MESH' and not o.modifiers:
        continue      # untouched (keeps shared bolt meshes instanced)
    ev = o.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    me.name = o.name + "_mesh"
    if o.type == 'MESH':
        o.modifiers.clear()
        old = o.data
        o.data = me
        if old.users == 0:
            bpy.data.meshes.remove(old)
    else:
        new = bpy.data.objects.new(o.name + "_tmp", me)
        for c in o.users_collection:
            c.objects.link(new)
        new.parent = o.parent
        new.matrix_parent_inverse = o.matrix_parent_inverse.copy()
        new.matrix_basis = o.matrix_basis.copy()
        for ch in o.children:
            ch.parent = new
        nm = o.name
        bpy.data.objects.remove(o)
        new.name = nm
    converted += 1
# constraints on the spring/shock still work on meshes; strip the spring's (geometry is baked at rest)
for o in bike.all_objects:
    if o.animation_data:
        for d in list(o.animation_data.drivers):
            if 'modifiers' in d.data_path:
                o.animation_data.drivers.remove(d)
for o in {*helpers, *cutters}:
    bpy.data.objects.remove(o)
if "Studio" in bpy.data.collections:
    for o in list(bpy.data.collections["Studio"].all_objects):
        bpy.data.objects.remove(o)
log("converted", converted, "objects")

meshes = [o for o in bike.all_objects if o.type == 'MESH']
tris = 0
for o in meshes:
    o.data.calc_loop_triangles()
    tris += len(o.data.loop_triangles)
log("mesh objects:", len(meshes), "triangles:", tris)


# ------------------------------------------------------------------ 2. texture sets
def is_procedural(m):
    if not m or not m.use_nodes:
        return False
    b = next((n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    return b is not None and any(b.inputs[k].is_linked for k in ("Base Color", "Roughness", "Metallic", "Normal"))


def set_of(m):
    for s, prefixes in SETS.items():
        if m.name.startswith(prefixes):
            return s
    return None


set_objs = {s: [] for s in SETS}
for o in meshes:
    mats = [sl.material for sl in o.material_slots if sl.material]
    if not mats or not any(is_procedural(m) for m in mats):
        continue
    s = set_of(mats[0])
    if s:
        set_objs[s].append(o)
for s, obs in set_objs.items():
    log("set", s, len(obs), "objects")


def unwrap(objs):
    bpy.ops.object.mode_set(mode='OBJECT') if bpy.context.object and bpy.context.object.mode != 'OBJECT' else None
    bpy.ops.object.select_all(action='DESELECT')
    seen = set()
    act = None
    for o in objs:
        if o.data.name in seen:
            continue
        seen.add(o.data.name)
        o.select_set(True)
        act = o
    bpy.context.view_layer.objects.active = act
    for o in objs:
        if not o.data.uv_layers:
            o.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.0, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(udim_source='CLOSEST_UDIM', rotate=True, margin_method='FRACTION', margin=0.0025, shape_method='CONCAVE')
    bpy.ops.object.mode_set(mode='OBJECT')


def load_uv_source(path):
    """{mesh name: (vertex coords, loop uvs)} from a previous game export (linked, so names never clash)."""
    import numpy as np
    with bpy.data.libraries.load(path, link=True) as (src, dst):
        dst.meshes = list(src.meshes)
    out = {}
    for me in dst.meshes:
        if me is None or not me.uv_layers:
            continue
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        uv = np.empty(len(me.loops) * 2, dtype=np.float32)
        me.uv_layers.active.data.foreach_get("uv", uv)
        out[me.name] = (co, uv)
    for lib in list(bpy.data.libraries):
        bpy.data.libraries.remove(lib)
    return out


def copy_uvs(objs, src):
    """Copy the whole set's UVs from src, only if every mesh in it matches (same vertices, same loop count)."""
    import numpy as np
    pairs = []
    for me in {o.data for o in objs}:
        hit = src.get(me.name)
        if hit is None or len(hit[1]) != len(me.loops) * 2:
            return me.name
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        if len(co) != len(hit[0]) or not np.allclose(co, hit[0], atol=1e-5):
            return me.name
        pairs.append((me, hit[1]))
    for me, uv in pairs:
        if not me.uv_layers:
            me.uv_layers.new(name="UVMap")
        me.uv_layers.active.data.foreach_set("uv", uv)
    return None


def uv_hash(objs):
    import hashlib
    import numpy as np
    h = hashlib.sha1()
    for me in sorted({o.data for o in objs}, key=lambda d: d.name):
        uv = me.uv_layers.active.data
        a = np.empty(len(uv) * 2, dtype=np.float32)
        uv.foreach_get("uv", a)
        h.update(me.name.encode())
        h.update(np.round(a, 5).tobytes())
    return h.hexdigest()[:12]


# ------------------------------------------------------------------ 3. bake
sc.render.engine = 'CYCLES'
sc.cycles.device = 'CPU'
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = False
sc.render.bake.margin = 8
sc.render.bake.use_clear = True


def principled(m):
    return next(n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')


def output(m):
    return next(n for n in m.node_tree.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output)


def bake_pass(objs, mats, img, kind):
    """kind: 'color' | 'rough' | 'metal' | 'normal'"""
    tmp = []
    for m in mats:
        nt = m.node_tree
        b, out = principled(m), output(m)
        tex = nt.nodes.new('ShaderNodeTexImage')
        tex.image = img
        nt.nodes.active = tex
        tmp.append((m, tex, None, None))
        if kind == 'normal':
            continue
        sock = {'color': 'Base Color', 'rough': 'Roughness', 'metal': 'Metallic'}[kind]
        em = nt.nodes.new('ShaderNodeEmission')
        src = b.inputs[sock]
        if src.is_linked:
            nt.links.new(src.links[0].from_socket, em.inputs['Color'])
        else:
            v = src.default_value
            em.inputs['Color'].default_value = (v[0], v[1], v[2], 1) if kind == 'color' else (v, v, v, 1)
        orig = out.inputs['Surface'].links[0].from_socket
        nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
        tmp[-1] = (m, tex, em, orig)
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if kind == 'normal':
        sc.cycles.samples = max(4, SAMPLES // 3)
        bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', margin=8)
    else:
        sc.cycles.samples = SAMPLES
        bpy.ops.object.bake(type='EMIT', margin=8)
    for m, tex, em, orig in tmp:
        nt = m.node_tree
        if em:
            nt.links.new(orig, output(m).inputs['Surface'])
            nt.nodes.remove(em)
        nt.nodes.remove(tex)


game_mats = {}
uv_src = load_uv_source(UV_FROM) if UV_FROM else None
for s, objs in set_objs.items():
    if not objs:
        continue
    mismatch = copy_uvs(objs, uv_src) if uv_src is not None else "no source"
    if mismatch is None:
        log("copied UVs", s, "from", os.path.basename(UV_FROM), "uv", uv_hash(objs))
    else:
        if uv_src is not None:
            log("UVs for", s, "not copied (%s differs): unwrapping, its maps will be re-baked" % mismatch)
        unwrap(objs)
        log("unwrapped", s, "uv", uv_hash(objs))
    # maps can only be reused if the UVs are the ones they were baked on
    reuse = RESUME and (mismatch is None or uv_src is None)
    if UV_ONLY:
        continue
    mats = sorted({sl.material for o in objs for sl in o.material_slots if sl.material and is_procedural(sl.material)},
                  key=lambda m: m.name)
    imgs = {}
    for kind, cs in (("color", 'sRGB'), ("rough", 'Non-Color'), ("metal", 'Non-Color'), ("normal", 'Non-Color')):
        nm = "T_%s_%s" % (s, {"color": "BaseColor", "rough": "Roughness", "metal": "Metallic", "normal": "Normal"}[kind])
        path = os.path.join(TEX, nm + ".png")
        if reuse and os.path.exists(path):
            img = bpy.data.images.load(path)
            if tuple(img.size) == (RES, RES):
                img.name = nm
                img.colorspace_settings.name = cs
                imgs[kind] = img
                log("reused", nm)
                continue
            bpy.data.images.remove(img)
        img = bpy.data.images.new(nm, RES, RES, alpha=False, float_buffer=False)
        img.colorspace_settings.name = cs
        if kind == 'normal':
            img.generated_color = (0.5, 0.5, 1.0, 1.0)
        if not SKIP_BAKE:
            bake_pass(objs, mats, img, kind)
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
        imgs[kind] = img
        log("baked", nm)
    # ---------------------------------------------------- 4. game material
    gm = bpy.data.materials.new("MAT_%s" % s)
    gm.use_nodes = True
    nt = gm.node_tree
    b = principled(gm)
    def tex_node(img, loc):
        t = nt.nodes.new('ShaderNodeTexImage')
        t.image = img
        t.location = loc
        return t
    nt.links.new(tex_node(imgs["color"], (-600, 300)).outputs['Color'], b.inputs['Base Color'])
    nt.links.new(tex_node(imgs["rough"], (-600, 0)).outputs['Color'], b.inputs['Roughness'])
    nt.links.new(tex_node(imgs["metal"], (-600, -300)).outputs['Color'], b.inputs['Metallic'])
    nmap = nt.nodes.new('ShaderNodeNormalMap')
    nmap.location = (-300, -600)
    nt.links.new(tex_node(imgs["normal"], (-600, -600)).outputs['Color'], nmap.inputs['Color'])
    nt.links.new(nmap.outputs['Normal'], b.inputs['Normal'])
    game_mats[s] = gm
    for o in objs:
        for sl in o.material_slots:
            if sl.material and is_procedural(sl.material):
                sl.material = gm

if UV_ONLY:
    sys.exit(0)

# clean up the now unused procedural materials
for m in list(bpy.data.materials):
    if m.users == 0:
        bpy.data.materials.remove(m)

finish()
log("DONE tris=%d" % tris)
