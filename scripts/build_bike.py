"""
Wasteland Scrambler - procedural concept-bike builder for Blender 4.2+ (tested on 4.5 LTS).

Run headless:
    blender -b --factory-startup --python scripts/build_bike.py -- --out bike/wasteland_scrambler.blend

Or paste the whole file into Blender's Text Editor / a Blender-MCP `execute_blender_code`
call and run it: it wipes the current scene and rebuilds the bike from scratch.

Conventions
-----------
* Units: metres, real-world scale (wheelbase ~1.46 m, seat height ~0.9 m).
* Bike faces -Y (Blender "Front" view looks at its nose). +Z up. +X is the rider's LEFT.
  (glTF export maps this to the glTF standard: +Y up, asset facing +Z.)
* Everything hangs off the `Bike_Root` empty. Moving parts sit under pivot empties:
      Bike_Root
        FrontEnd_SteeringAxis (tilted to the rake angle)
          FrontEnd_Steer            <- rotate local Z to steer
            FrontEnd_ForkTravel     <- translate local Z to compress the fork
              FrontWheel_Pivot      <- rotate local X to spin
        RearEnd_SwingarmPivot       <- rotate local X to move the suspension
          RearWheel_Pivot           <- rotate local X to spin
  Custom properties on Bike_Root (steer, fork_travel, swingarm_angle, distance) drive them.
* Modifiers are left live (bevels, subdivision, arrays, mirrors, booleans, screws) so the
  asset stays editable. Viewport subdivision levels are the "game" LOD, render levels the
  "VFX/hero" LOD; exporters use the viewport levels.
"""

import bpy
import bmesh
import math
import os
import random
import sys
from math import pi, sin, cos, radians as rad
from mathutils import Vector, Matrix, Euler

random.seed(11)

# ---------------------------------------------------------------------------------------------
# Global layout parameters (edit these to re-proportion the bike, then re-run)
# ---------------------------------------------------------------------------------------------
P = dict(
    front_axle=Vector((0.0, -0.84, 0.37)),   # wheelbase 1.56 m: room for 29" tyres ahead of the engine
    rear_axle=Vector((0.0, 0.72, 0.375)),
    rake=25.0,                 # steering head angle from vertical (deg) -> trail ~117 mm
    fork_offset=0.05,          # steering axis behind fork tubes
    yokes=(0.60, 0.83),        # bottom / top yoke position along the fork tubes (m from the axle)
    steer_lock=35.0,           # steering stop angle (deg)
    fork_travel=0.12,          # front wheel travel along the fork (m)
    swing_bump=8.0,            # swingarm rotation at full bump / droop (deg)
    swing_droop=-6.0,
    front_tire=dict(R=0.37, half_w=0.108, R_rim=0.215, lug_h=0.024, lugs=21),
    rear_tire=dict(R=0.375, half_w=0.122, R_rim=0.215, lug_h=0.026, lugs=23),
    swing_pivot=Vector((0.0, 0.22, 0.43)),
    yellow=(0.80, 0.43, 0.006),
)
SLIDER_R, STAN_R, CHAIN_HALF = 0.0395, 0.0265, 0.0095


def tire_halfwidth(spec):
    """Half width of a tyre including the shoulder knobs (the real lateral envelope)."""
    return spec['half_w'] + 0.609 * spec['lug_h'] + 0.002


def layout():
    """Derive packaging dimensions from the tyre envelopes so nothing rubs (see check_mechanics.py)."""
    P['front_env'] = tire_halfwidth(P['front_tire'])
    P['rear_env'] = tire_halfwidth(P['rear_tire'])
    P['fork_half_span'] = round(P['front_env'] + 0.012 + SLIDER_R, 4)       # slider clears the knobs by 12 mm
    P['chain_x'] = round(P['rear_env'] + 0.012 + CHAIN_HALF, 4)             # chain clears the knobs by 12 mm
    P['arm_x'] = round(P['chain_x'] + CHAIN_HALF + 0.008 + 0.018, 4)        # swingarm arm centre line


layout()
TEXDIR = ""
try:
    TEXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bike", "textures")
except NameError:
    TEXDIR = os.environ.get("BIKE_TEXDIR", "")

# ---------------------------------------------------------------------------------------------
# Scene / collection helpers
# ---------------------------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.scale_length = 1.0
    sc.render.engine = 'CYCLES'


COLL = {}


def coll(name, parent=None):
    if name in COLL:
        return COLL[name]
    c = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(c)
    COLL[name] = c
    return c


PARENT_JOBS = []   # (child, parent) resolved at the end, keeping world transforms


def parent_later(child, parent):
    PARENT_JOBS.append((child, parent))
    return child


def resolve_parents():
    bpy.context.view_layer.update()
    # parents first so their matrix_world is final
    # clean local transforms: parent inverse = identity, basis = parent^-1 @ world
    worlds = {c.name: c.matrix_world.copy() for c, _ in PARENT_JOBS}
    for p in {p for _, p in PARENT_JOBS}:
        worlds.setdefault(p.name, p.matrix_world.copy())
    for child, parent in PARENT_JOBS:
        child.parent = parent
        child.matrix_parent_inverse = Matrix.Identity(4)
        child.matrix_basis = worlds[parent.name].inverted() @ worlds[child.name]
    bpy.context.view_layer.update()


def frame_from_z(z, xhint=None):
    """3x3 rotation whose local Z points along z."""
    z = Vector(z).normalized()
    xhint = Vector(xhint) if xhint is not None else Vector((1, 0, 0))
    if abs(z.dot(xhint.normalized())) > 0.95:
        xhint = Vector((0, 1, 0)) if abs(z.y) < 0.9 else Vector((0, 0, 1))
    y = z.cross(xhint).normalized()
    x = y.cross(z).normalized()
    return Matrix((x, y, z)).transposed()


def mat4(loc=(0, 0, 0), rot3=None):
    m = rot3.to_4x4() if rot3 is not None else Matrix.Identity(4)
    m.translation = Vector(loc)
    return m


def finish_mesh(me, smooth=True, angle=32):
    if smooth:
        me.shade_smooth()
        me.set_sharp_from_angle(angle=rad(angle))


def obj_from_bm(name, bm, collection, mat=None, matrix=None, smooth=True, angle=32, uv=True):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    if uv and not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    ob = bpy.data.objects.new(name, me)
    collection.objects.link(ob)
    if matrix is not None:
        ob.matrix_world = matrix
    if mat is not None:
        if isinstance(mat, (list, tuple)):
            for m in mat:
                me.materials.append(m)
        else:
            me.materials.append(mat)
    finish_mesh(me, smooth, angle)
    return ob


def linked_copy(src, name, matrix, collection):
    ob = bpy.data.objects.new(name, src.data)
    collection.objects.link(ob)
    ob.matrix_world = matrix
    for m in src.modifiers:   # copy modifier stack (data is shared)
        nm = ob.modifiers.new(m.name, m.type)
        for attr in dir(m):
            if attr.startswith('_') or attr in {'rna_type', 'name', 'type', 'is_override_data',
                                                 'persistent_uid', 'is_active', 'is_override_data_editable'}:
                continue
            try:
                setattr(nm, attr, getattr(m, attr))
            except Exception:
                pass
    return ob


# ---------------------------------------------------------------------------------------------
# Modifier helpers (non-destructive)
# ---------------------------------------------------------------------------------------------

def mod_bevel(ob, width=0.003, segments=2, angle=35, harden=True, name="Bevel"):
    m = ob.modifiers.new(name, 'BEVEL')
    m.width = width
    m.segments = segments
    m.limit_method = 'ANGLE'
    m.angle_limit = rad(angle)
    m.use_clamp_overlap = True
    m.harden_normals = harden
    return m


def mod_subsurf(ob, view=1, render=2, name="Subdivision"):
    m = ob.modifiers.new(name, 'SUBSURF')
    m.levels = view
    m.render_levels = render
    m.quality = 3
    return m


def mod_mirror(ob, axis=0, mirror_obj=None, merge=True, name="Mirror"):
    m = ob.modifiers.new(name, 'MIRROR')
    m.use_axis = [axis == 0, axis == 1, axis == 2]
    if mirror_obj:
        m.mirror_object = mirror_obj
    m.use_mirror_merge = merge
    m.merge_threshold = 0.0005
    return m


def mod_solidify(ob, thick=0.003, offset=-1, name="Solidify"):
    m = ob.modifiers.new(name, 'SOLIDIFY')
    m.thickness = thick
    m.offset = offset
    m.use_even_offset = True
    return m


def mod_weighted_normals(ob):
    m = ob.modifiers.new("WeightedNormal", 'WEIGHTED_NORMAL')
    m.keep_sharp = True
    return m


def radial_array(ob, count, axis=0, name="RadialArray", phase=0.0):
    """Array `ob` around its own local axis with an empty offset + a driver so that
    changing the modifier Count automatically keeps a full 360 deg ring."""
    e = bpy.data.objects.new(ob.name + "_ArrayOffset", None)
    e.empty_display_type = 'ARROWS'
    e.empty_display_size = 0.03
    for c in ob.users_collection:
        c.objects.link(e)
    e.parent = ob
    e.hide_viewport = False
    e.hide_render = True
    rot = [0.0, 0.0, 0.0]
    rot[axis] = 2 * pi / count
    e.rotation_euler = rot
    m = ob.modifiers.new(name, 'ARRAY')
    m.count = count
    m.use_relative_offset = False
    m.use_object_offset = True
    m.offset_object = e
    fc = e.driver_add('rotation_euler', axis)
    drv = fc.driver
    drv.type = 'SCRIPTED'
    var = drv.variables.new()
    var.name = 'n'
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = 'OBJECT'
    var.targets[0].id = ob
    var.targets[0].data_path = f'modifiers["{name}"].count'
    drv.expression = '2*pi/n'
    return m


def mod_boolean(ob, cutter, name="Boolean", op='DIFFERENCE'):
    m = ob.modifiers.new(name, 'BOOLEAN')
    m.operation = op
    m.object = cutter
    m.solver = 'EXACT'
    cutter.display_type = 'WIRE'
    cutter.hide_render = True
    cutter.visible_camera = False
    cutter.visible_shadow = False
    parent_later(cutter, ob)
    return m


# ---------------------------------------------------------------------------------------------
# Geometry primitives (all return bmesh or objects with clean quad topology + UVs)
# ---------------------------------------------------------------------------------------------

def bm_cyl(bm, r1, r2, depth, segs=24, matrix=None, caps=True, base_at_zero=True):
    M = Matrix.Translation((0, 0, depth / 2 if base_at_zero else 0))
    if matrix is not None:
        M = matrix @ M
    bmesh.ops.create_cone(bm, cap_ends=caps, cap_tris=False, segments=segs, radius1=r1,
                          radius2=r2, depth=depth, matrix=M, calc_uvs=True)


def bm_box(bm, size, matrix=None):
    M = Matrix.Diagonal((size[0], size[1], size[2], 1.0))
    if matrix is not None:
        M = matrix @ M
    bmesh.ops.create_cube(bm, size=1.0, matrix=M, calc_uvs=True)


def cyl_obj(name, p0, p1, r, collection, mat, segs=24, r2=None, bevel=0.0, caps=True,
            angle=40, xhint=None):
    """Cylinder from p0 to p1; object origin at p0, local +Z along the axis."""
    p0, p1 = Vector(p0), Vector(p1)
    L = (p1 - p0).length
    bm = bmesh.new()
    bm_cyl(bm, r, r if r2 is None else r2, L, segs, caps=caps)
    ob = obj_from_bm(name, bm, collection, mat, mat4(p0, frame_from_z(p1 - p0, xhint)), angle=angle)
    if bevel > 0:
        mod_bevel(ob, bevel, 2, 50)
    return ob


def box_obj(name, center, size, collection, mat, rot=(0, 0, 0), bevel=0.004, segs=2, angle=40):
    bm = bmesh.new()
    bm_box(bm, size)
    ob = obj_from_bm(name, bm, collection, mat,
                     mat4(center, Euler([rad(a) for a in rot]).to_matrix()), angle=angle)
    if bevel > 0:
        mod_bevel(ob, bevel, segs, 40)
    return ob


def revolve_bm(profile, segs=64, closed=False, axis='Z', phase=0.0, arc=2 * pi, full=True):
    """Revolve a 2D profile [(axial, radius), ...] around an axis. Builds UVs (u=angle, v=profile)."""
    bm = bmesh.new()
    uv_l = bm.loops.layers.uv.new("UVMap")
    n = len(profile)
    steps = segs if full else segs + 1
    rings = []
    for i in range(steps):
        t = phase + arc * i / segs
        ring = []
        for a, r in profile:
            if axis == 'X':
                co = (a, r * cos(t), r * sin(t))
            elif axis == 'Y':
                co = (r * cos(t), a, r * sin(t))
            else:
                co = (r * cos(t), r * sin(t), a)
            ring.append(bm.verts.new(co))
        rings.append(ring)
    # profile arc-length for v coords
    vlen = [0.0]
    for j in range(1, n):
        vlen.append(vlen[-1] + math.hypot(profile[j][0] - profile[j - 1][0], profile[j][1] - profile[j - 1][1]))
    tot = vlen[-1] if vlen[-1] > 0 else 1
    jmax = n if closed else n - 1
    nseg = segs if full else segs
    for i in range(nseg):
        r0, r1 = rings[i], rings[(i + 1) % steps]
        for j in range(jmax):
            j1 = (j + 1) % n
            vs = (r0[j], r1[j], r1[j1], r0[j1])
            if len(set(vs)) < 4:
                continue
            try:
                f = bm.faces.new(vs)
            except ValueError:
                continue
            u0, u1 = i / segs, (i + 1) / segs
            v0 = vlen[j] / tot
            v1 = (vlen[j1] / tot) if j1 > j else 1.0
            for loop, uv in zip(f.loops, ((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
                loop[uv_l].uv = uv
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def revolve_obj(name, profile, collection, mat, matrix, segs=64, closed=False, axis='X', angle=35,
                phase=0.0):
    bm = revolve_bm(profile, segs, closed, axis, phase)
    return obj_from_bm(name, bm, collection, mat, matrix, angle=angle)


def extrude_bm(bm, pts2d, thick, matrix=None, z0=None):
    """Extrude a planar polygon (local XY) along local Z (centred unless z0 given)."""
    z0 = -thick / 2 if z0 is None else z0
    M = matrix or Matrix.Identity(4)
    bot = [bm.verts.new(M @ Vector((x, y, z0))) for x, y in pts2d]
    top = [bm.verts.new(M @ Vector((x, y, z0 + thick))) for x, y in pts2d]
    n = len(pts2d)
    bm.faces.new(list(reversed(bot)))
    bm.faces.new(top)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((bot[i], bot[j], top[j], top[i]))
    return bm


def extrude_obj(name, pts2d, thick, collection, mat, matrix, bevel=0.002, angle=40, z0=None):
    bm = bmesh.new()
    extrude_bm(bm, pts2d, thick, z0=z0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = obj_from_bm(name, bm, collection, mat, matrix, angle=angle)
    if bevel > 0:
        mod_bevel(ob, bevel, 2, 40)
    return ob


def rrect(w, h, r, seg=4, cx=0.0, cy=0.0):
    pts = []
    r = min(r, w / 2 - 1e-4, h / 2 - 1e-4)
    for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        ccx, ccy = sx * (w / 2 - r), sy * (h / 2 - r)
        a0 = k * pi / 2
        for i in range(seg + 1):
            a = a0 + (pi / 2) * i / seg
            pts.append((cx + ccx + r * cos(a), cy + ccy + r * sin(a)))
    return pts


def circle_pts(r, n=24, cx=0.0, cy=0.0, a0=0.0):
    return [(cx + r * cos(a0 + 2 * pi * i / n), cy + r * sin(a0 + 2 * pi * i / n)) for i in range(n)]


def convex_hull(pts):
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def cage_obj(name, collection, mat, shape_fn, cuts=3, view=2, render=3, size=(1, 1, 1)):
    """Subdivision-surface body panel. A subdivided cube is mapped through shape_fn(u,v,w)->Vector
    with u,v,w in [-1,1]. The cage stays editable (tweak verts, the subsurf does the rest)."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0, calc_uvs=True)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    for v in bm.verts:
        u, vv, w = (max(-1, min(1, round(c, 5))) for c in v.co)
        v.co = shape_fn(u, vv, w)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if bm.calc_volume(signed=True) < 0:          # guarantee outward normals (game engines cull back faces)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    ob = obj_from_bm(name, bm, collection, mat, smooth=True, angle=180)
    mod_subsurf(ob, view, render)
    return ob


def lerp(a, b, t):
    return a + (b - a) * t


def interp(xs, ys, x):
    if x <= xs[0]:
        return ys[0]
    for i in range(len(xs) - 1):
        if x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return lerp(ys[i], ys[i + 1], t)
    return ys[-1]


# ---------------------------------------------------------------------------------------------
# Curves (pipes, hoses, frame tubes, cables) - stay editable as splines
# ---------------------------------------------------------------------------------------------

def curve_obj(name, pts, radius, collection, mat, mode='smooth', bend=0.04, res=10, bevel_res=4,
              caps=True, closed=False, radii=None, tilt=None):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = res
    cu.bevel_mode = 'ROUND'
    cu.bevel_depth = radius
    cu.bevel_resolution = bevel_res
    cu.use_fill_caps = caps and not closed
    cu.twist_mode = 'MINIMUM'
    pts = [Vector(p) for p in pts]
    sp = cu.splines.new('BEZIER')
    if mode == 'smooth' or len(pts) < 3:
        sp.bezier_points.add(len(pts) - 1)
        for bp, p in zip(sp.bezier_points, pts):
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
        sp.use_cyclic_u = closed
    else:
        # straight runs with circular-ish bends of radius `bend` at each interior vertex
        ctrl = []   # (co, hl, hr)
        k = 0.5523
        a0, a1 = pts[0], pts[1]
        d = (a1 - a0).normalized()
        ctrl.append((a0, a0 - d * 0.01, a0 + d * 0.01))
        for i in range(1, len(pts) - 1):
            A, Pc, B = pts[i - 1], pts[i], pts[i + 1]
            da, db = (A - Pc), (B - Pc)
            r = min(bend, da.length * 0.45, db.length * 0.45)
            p1 = Pc + da.normalized() * r
            p2 = Pc + db.normalized() * r
            ctrl.append((p1, p1 + (p1 - Pc) * k, p1 + (Pc - p1) * k))
            ctrl.append((p2, p2 + (Pc - p2) * k, p2 + (p2 - Pc) * k))
        b0, b1 = pts[-2], pts[-1]
        d = (b1 - b0).normalized()
        ctrl.append((b1, b1 - d * 0.01, b1 + d * 0.01))
        sp.bezier_points.add(len(ctrl) - 1)
        for bp, (co, hl, hr) in zip(sp.bezier_points, ctrl):
            bp.handle_left_type = bp.handle_right_type = 'FREE'
            bp.co, bp.handle_left, bp.handle_right = co, hl, hr
    if radii:
        for bp, rr in zip(sp.bezier_points, radii):
            bp.radius = rr
    if tilt:
        for bp, tt in zip(sp.bezier_points, tilt):
            bp.tilt = tt
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    if mat:
        cu.materials.append(mat)
    return ob


def poly_curve_obj(name, pts, collection, closed=True):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('POLY')
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (c[0], c[1], c[2], 1.0)
    sp.use_cyclic_u = closed
    ob = bpy.data.objects.new(name, cu)
    collection.objects.link(ob)
    return ob


# ---------------------------------------------------------------------------------------------
# Materials: procedural PBR, layered wear (paint -> primer -> bare metal -> rust -> dirt)
# ---------------------------------------------------------------------------------------------

def _sock(node, ident, out=False):
    for s in (node.outputs if out else node.inputs):
        if s.identifier == ident:
            return s
    raise KeyError(ident)


def _set(nt, sock, val):
    if isinstance(val, bpy.types.NodeSocket):
        nt.links.new(val, sock)
    else:
        if sock.type == 'RGBA' and len(val) == 3:
            val = (*val, 1.0)
        sock.default_value = val


def N(nt, t, loc, inputs=None, **props):
    n = nt.nodes.new(t)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    for k, v in (inputs or {}).items():
        _set(nt, n.inputs[k], v)
    return n


def mixc(nt, loc, fac, a, b, blend='MIX'):
    m = N(nt, 'ShaderNodeMix', loc, data_type='RGBA', blend_type=blend)
    m.clamp_factor = True
    _set(nt, _sock(m, 'Factor_Float'), fac)
    _set(nt, _sock(m, 'A_Color'), a)
    _set(nt, _sock(m, 'B_Color'), b)
    return _sock(m, 'Result_Color', True)


def mixf(nt, loc, fac, a, b):
    m = N(nt, 'ShaderNodeMix', loc, data_type='FLOAT')
    _set(nt, _sock(m, 'Factor_Float'), fac)
    _set(nt, _sock(m, 'A_Float'), a)
    _set(nt, _sock(m, 'B_Float'), b)
    return _sock(m, 'Result_Float', True)


def mth(nt, loc, op, a, b=0.0, clamp=False):
    m = N(nt, 'ShaderNodeMath', loc, operation=op, use_clamp=clamp)
    _set(nt, m.inputs[0], a)
    _set(nt, m.inputs[1], b)
    return m.outputs[0]


def ramp(nt, loc, fac, stops):
    r = N(nt, 'ShaderNodeValToRGB', loc)
    _set(nt, r.inputs['Fac'], fac)
    el = r.color_ramp.elements
    while len(el) > 2:
        el.remove(el[-1])
    for i, (pos, col) in enumerate(stops):
        e = el[i] if i < 2 else el.new(pos)
        e.position = pos
        e.color = col if len(col) == 4 else (*col, 1.0)
    return r.outputs['Color']


def noise(nt, loc, vec, scale, detail=8.0, rough=0.6, distort=0.0, lac=2.0):
    n = N(nt, 'ShaderNodeTexNoise', loc)
    _set(nt, n.inputs['Vector'], vec)
    n.inputs['Scale'].default_value = scale
    n.inputs['Detail'].default_value = detail
    n.inputs['Roughness'].default_value = rough
    n.inputs['Distortion'].default_value = distort
    n.inputs['Lacunarity'].default_value = lac
    return n


def new_mat(name, color_hint=(0.5, 0.5, 0.5)):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = (*color_hint, 1.0)   # viewport solid colour
    nt = m.node_tree
    nt.nodes.clear()
    out = N(nt, 'ShaderNodeOutputMaterial', (1400, 0))
    b = N(nt, 'ShaderNodeBsdfPrincipled', (1100, 0))
    nt.links.new(b.outputs['BSDF'], out.inputs['Surface'])
    return m, nt, b


def _dirt_mask(nt, co, amount, x=-200, y=-600, ao_dist=0.06, low_z=0.18, high_z=0.75):
    ao = N(nt, 'ShaderNodeAmbientOcclusion', (x - 600, y), samples=8,
           inputs={'Distance': ao_dist})
    ao.only_local = False
    inv = mth(nt, (x - 400, y), 'SUBTRACT', 1.0, ao.outputs['AO'])
    cav = ramp(nt, (x - 250, y), inv, [(0.08, (0, 0, 0)), (0.5, (1, 1, 1))])
    geo = N(nt, 'ShaderNodeNewGeometry', (x - 600, y - 250))
    sep = N(nt, 'ShaderNodeSeparateXYZ', (x - 450, y - 250))
    nt.links.new(geo.outputs['Position'], sep.inputs[0])
    mr = N(nt, 'ShaderNodeMapRange', (x - 300, y - 250), inputs={'From Min': low_z, 'From Max': high_z,
                                                                 'To Min': 1.0, 'To Max': 0.0})
    nt.links.new(sep.outputs['Z'], mr.inputs['Value'])
    nz = noise(nt, (x - 450, y - 450), co, 3.0, 6, 0.55)
    brk = ramp(nt, (x - 250, y - 450), nz.outputs['Fac'], [(0.4, (0, 0, 0)), (0.62, (1, 1, 1))])
    s = mth(nt, (x - 100, y - 100), 'MAXIMUM', cav, mth(nt, (x - 100, y - 300), 'MULTIPLY', mr.outputs['Result'], brk))
    return mth(nt, (x, y), 'MULTIPLY', s, amount, clamp=True)


def _streaks(nt, co, loc, scale, stretch=14.0):
    """Vertical drip/streak mask (object space, stretched noise)."""
    mp = N(nt, 'ShaderNodeMapping', loc)
    nt.links.new(co, mp.inputs['Vector'])
    mp.inputs['Scale'].default_value = (stretch * scale, stretch * scale, 0.9 * scale)
    nz = noise(nt, (loc[0] + 200, loc[1]), mp.outputs['Vector'], 1.0, 6, 0.55, 0.0)
    return ramp(nt, (loc[0] + 400, loc[1]), nz.outputs['Fac'], [(0.5, (0, 0, 0)), (0.72, (1, 1, 1))])


def _top_dust(nt, co, loc, amount):
    geo = N(nt, 'ShaderNodeNewGeometry', loc)
    sep = N(nt, 'ShaderNodeSeparateXYZ', (loc[0] + 150, loc[1]))
    nt.links.new(geo.outputs['Normal'], sep.inputs[0])
    up = ramp(nt, (loc[0] + 300, loc[1]), sep.outputs['Z'], [(0.45, (0, 0, 0)), (0.95, (1, 1, 1))])
    nz = noise(nt, (loc[0] + 150, loc[1] - 200), co, 9.0, 8, 0.6)
    brk = ramp(nt, (loc[0] + 300, loc[1] - 200), nz.outputs['Fac'], [(0.35, (0, 0, 0)), (0.7, (1, 1, 1))])
    return mth(nt, (loc[0] + 500, loc[1]), 'MULTIPLY', mth(nt, (loc[0] + 450, loc[1] - 100), 'MULTIPLY', up, brk), amount, clamp=True)


def mat_paint(name, rgb, rough=0.34, chip=0.5, rust=0.6, dirt=0.5, scale=1.0, coat=0.35,
              primer=(0.20, 0.19, 0.17), metal=(0.22, 0.21, 0.2), scratches=0.6, streak=0.5, dust=0.5, smudge=0.6,
              decal=None):
    """decal: dict(image=bpy Image (white = ink), center=(y, z), size=m, ink=rgb) projected onto both +-X sides
    in object space. It sits *under* the wear layers so chips, scratches and grime break it up like real paint."""
    m, nt, b = new_mat(name, rgb)
    tc = N(nt, 'ShaderNodeTexCoord', (-2000, 0))
    co = tc.outputs['Object']
    n1 = noise(nt, (-1700, 500), co, 13 * scale, 15, 0.7, 0.35)
    n2 = noise(nt, (-1700, 200), co, 1.4 * scale, 4, 0.55)
    fac = mixf(nt, (-1500, 400), 0.38, n1.outputs['Fac'], n2.outputs['Fac'])
    t = 0.615 - 0.085 * chip
    m_primer = ramp(nt, (-1300, 650), fac, [(t - 0.016, (0, 0, 0)), (t - 0.009, (1, 1, 1))])
    m_chip0 = ramp(nt, (-1300, 450), fac, [(t, (0, 0, 0)), (t + 0.005, (1, 1, 1))])
    # scratches: thin iso-lines of a distorted noise, masked to worn zones
    ns = noise(nt, (-1700, -100), co, 4.5 * scale, 3, 0.5, 3.0)
    lines = ramp(nt, (-1500, -100), ns.outputs['Fac'], [(0.493, (0, 0, 0)), (0.5, (1, 1, 1))])
    lines2 = ramp(nt, (-1500, -250), ns.outputs['Fac'], [(0.5, (1, 1, 1)), (0.507, (0, 0, 0))])
    ln = mth(nt, (-1350, -150), 'MULTIPLY', lines, lines2)
    zone = ramp(nt, (-1350, -300), n2.outputs['Fac'], [(0.45, (0, 0, 0)), (0.6, (1, 1, 1))])
    m_scr = mth(nt, (-1200, -200), 'MULTIPLY', mth(nt, (-1250, -150), 'MULTIPLY', ln, zone), scratches)
    m_chip = mth(nt, (-1100, 400), 'MAXIMUM', m_chip0, m_scr, clamp=True)
    m_primer = mth(nt, (-1100, 600), 'MAXIMUM', m_primer, m_chip, clamp=True)
    n3 = noise(nt, (-1700, -450), co, 16 * scale, 10, 0.7, 0.4)
    m_r = ramp(nt, (-1300, 250), n3.outputs['Fac'], [(0.52 - 0.2 * rust, (0, 0, 0)), (0.6 - 0.2 * rust, (1, 1, 1))])
    m_rust = mth(nt, (-1000, 250), 'MULTIPLY', m_chip0, m_r)
    stk = _streaks(nt, co, (-1700, -700), scale)
    stk_s = mth(nt, (-1000, -650), 'MULTIPLY', stk, streak)
    # paint colour: sun-fade variation and darker drip streaks
    pv = ramp(nt, (-1300, 50), n2.outputs['Fac'], [(0.25, tuple(c * 0.72 for c in rgb)), (0.75, tuple(min(1, c * 1.1 + 0.015) for c in rgb))])
    pv = mixc(nt, (-1000, 50), mth(nt, (-1100, -50), 'MULTIPLY', stk_s, 0.55), pv, tuple(c * 0.45 for c in rgb))
    if decal:
        sep = N(nt, 'ShaderNodeSeparateXYZ', (-1700, 900))
        nt.links.new(co, sep.inputs[0])
        cy, cz = decal['center']
        sz = decal['size']
        u = mth(nt, (-1550, 950), 'MULTIPLY_ADD', sep.outputs['Y'], -1.0 / sz)
        u.node.inputs[2].default_value = 0.5 + cy / sz
        v = mth(nt, (-1550, 850), 'MULTIPLY_ADD', sep.outputs['Z'], 1.0 / sz)
        v.node.inputs[2].default_value = 0.5 - cz / sz
        cmb = N(nt, 'ShaderNodeCombineXYZ', (-1400, 900))
        nt.links.new(u, cmb.inputs[0])
        nt.links.new(v, cmb.inputs[1])
        tex = N(nt, 'ShaderNodeTexImage', (-1250, 900), image=decal['image'], extension='CLIP', interpolation='Cubic')
        tex.image.colorspace_settings.name = 'Non-Color'
        nt.links.new(cmb.outputs[0], tex.inputs['Vector'])
        nsep = N(nt, 'ShaderNodeSeparateXYZ', (-1550, 700))
        nt.links.new(tc.outputs['Normal'], nsep.inputs[0])
        side = ramp(nt, (-1250, 700), mth(nt, (-1400, 700), 'ABSOLUTE', nsep.outputs['X']), [(0.45, (0, 0, 0)), (0.65, (1, 1, 1))])
        dm_ = mth(nt, (-1100, 800), 'MULTIPLY', tex.outputs['Color'], side)
        ink = mixc(nt, (-1100, 650), mth(nt, (-1200, 600), 'MULTIPLY', n2.outputs['Fac'], 0.25), decal.get('ink', (0.012, 0.011, 0.01)), pv)
        pv = mixc(nt, (-950, 700), dm_, pv, ink)
    rust_col = ramp(nt, (-1300, -800), n3.outputs['Color'],
                    [(0.35, (0.075, 0.028, 0.01)), (0.6, (0.26, 0.085, 0.025)), (0.8, (0.40, 0.17, 0.05))])
    c = mixc(nt, (-850, 500), m_primer, pv, primer)
    c = mixc(nt, (-700, 500), m_chip, c, metal)
    c = mixc(nt, (-550, 500), m_rust, c, rust_col)
    # rust bleeding down from chips along the streaks
    bleed = mth(nt, (-700, 300), 'MULTIPLY', stk_s, mth(nt, (-850, 300), 'MULTIPLY', zone, rust), clamp=True)
    c = mixc(nt, (-400, 500), mth(nt, (-550, 300), 'MULTIPLY', bleed, 0.6), c, (0.20, 0.07, 0.02))
    # large greasy smudges
    nsm = noise(nt, (-700, 800), co, 2.6 * scale, 8, 0.62, 0.6)
    m_sm = mth(nt, (-400, 800), 'MULTIPLY', ramp(nt, (-550, 800), nsm.outputs['Fac'], [(0.52, (0, 0, 0)), (0.75, (1, 1, 1))]), smudge)
    c = mixc(nt, (-330, 650), mth(nt, (-380, 700), 'MULTIPLY', m_sm, 0.55), c, (0.06, 0.045, 0.03))
    dmask = _dirt_mask(nt, co, dirt)
    grime_col = ramp(nt, (-400, -900), n2.outputs['Fac'], [(0.3, (0.035, 0.028, 0.02)), (0.7, (0.09, 0.07, 0.05))])
    c = mixc(nt, (-250, 500), dmask, c, grime_col)
    dmask_dust = _top_dust(nt, co, (-900, -1100), dust)
    c = mixc(nt, (-100, 500), dmask_dust, c, (0.30, 0.26, 0.20))
    nt.links.new(c, b.inputs['Base Color'])
    r = mixf(nt, (-850, 0), n2.outputs['Fac'], rough * 0.8, rough * 1.35)
    r = mixf(nt, (-700, 0), m_chip, r, 0.42)
    r = mixf(nt, (-550, 0), m_rust, r, 0.88)
    r = mixf(nt, (-400, 0), dmask, r, 0.9)
    r = mixf(nt, (-320, 0), m_sm, r, 0.75)
    r = mixf(nt, (-250, 0), dmask_dust, r, 0.95)
    nt.links.new(r, b.inputs['Roughness'])
    met = mth(nt, (-550, -150), 'MULTIPLY', m_chip, mth(nt, (-700, -150), 'SUBTRACT', 0.9, m_rust, clamp=True))
    met = mth(nt, (-400, -150), 'MULTIPLY', met, mth(nt, (-550, -250), 'SUBTRACT', 1.0, mth(nt, (-650, -250), 'MAXIMUM', dmask, dmask_dust), clamp=True))
    nt.links.new(met, b.inputs['Metallic'])
    cw = mth(nt, (-400, -300), 'MULTIPLY', mth(nt, (-550, -350), 'SUBTRACT', 1.0,
                                               mth(nt, (-700, -350), 'MAXIMUM', m_primer, mth(nt, (-800, -400), 'MAXIMUM', dmask, dmask_dust)), clamp=True), coat)
    nt.links.new(cw, b.inputs['Coat Weight'])
    b.inputs['Coat Roughness'].default_value = 0.2
    h = mth(nt, (-700, -500), 'ADD', mth(nt, (-850, -500), 'MULTIPLY', m_primer, -0.35),
            mth(nt, (-850, -600), 'MULTIPLY', m_chip, -0.6))
    h = mth(nt, (-550, -500), 'ADD', h, mth(nt, (-700, -650), 'MULTIPLY', m_rust, n3.outputs['Fac']))
    h = mth(nt, (-400, -500), 'ADD', h, mth(nt, (-550, -650), 'MULTIPLY', dmask_dust, mth(nt, (-700, -750), 'MULTIPLY', n1.outputs['Fac'], 0.3)))
    bump = N(nt, 'ShaderNodeBump', (800, -400), inputs={'Strength': 0.45, 'Distance': 0.0012})
    nt.links.new(h, bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def mat_metal(name, rgb, rough=0.35, rust=0.4, grime=0.4, scale=1.0, rust_rough=0.85, dust=0.35,
              rust_cols=((0.09, 0.03, 0.012), (0.30, 0.10, 0.03), (0.42, 0.19, 0.06)), aniso=0.0):
    m, nt, b = new_mat(name, rgb)
    tc = N(nt, 'ShaderNodeTexCoord', (-1800, 0))
    co = tc.outputs['Object']
    n1 = noise(nt, (-1500, 300), co, 5 * scale, 12, 0.65, 0.3)
    n2 = noise(nt, (-1500, 0), co, 22 * scale, 8, 0.6)
    lo = 0.62 - 0.22 * rust
    m_rust = ramp(nt, (-1100, 300), n1.outputs['Fac'], [(lo, (0, 0, 0)), (lo + 0.07, (1, 1, 1))])
    rc = ramp(nt, (-1100, 0), n2.outputs['Fac'], [(0.3, rust_cols[0]), (0.55, rust_cols[1]), (0.75, rust_cols[2])])
    mv = ramp(nt, (-1100, -200), n2.outputs['Fac'], [(0.2, tuple(c * 0.8 for c in rgb)), (0.8, tuple(min(1, c * 1.1) for c in rgb))])
    c = mixc(nt, (-700, 300), m_rust, mv, rc)
    stk = _streaks(nt, co, (-1700, -700), scale)
    c = mixc(nt, (-550, 300), mth(nt, (-650, 200), 'MULTIPLY', stk, 0.5 * rust), c, rust_cols[1])
    dm = _dirt_mask(nt, co, grime)
    c = mixc(nt, (-400, 300), dm, c, (0.045, 0.037, 0.03))
    ds = _top_dust(nt, co, (-900, -1100), dust * grime)
    c = mixc(nt, (-300, 300), ds, c, (0.28, 0.25, 0.2))
    nt.links.new(c, b.inputs['Base Color'])
    rr = mixf(nt, (-700, 0), n2.outputs['Fac'], rough * 0.75, rough * 1.3)
    rr = mixf(nt, (-550, 0), m_rust, rr, rust_rough)
    rr = mixf(nt, (-400, 0), dm, rr, 0.8)
    rr = mixf(nt, (-300, 0), ds, rr, 0.92)
    nt.links.new(rr, b.inputs['Roughness'])
    met = mth(nt, (-400, -200), 'MULTIPLY', mth(nt, (-550, -200), 'SUBTRACT', 1.0, m_rust, clamp=True),
              mth(nt, (-550, -300), 'SUBTRACT', 1.0, mth(nt, (-700, -300), 'MAXIMUM', dm, ds), clamp=True))
    nt.links.new(met, b.inputs['Metallic'])
    if aniso:
        b.inputs['Anisotropic'].default_value = aniso
    h = mth(nt, (-600, -500), 'MULTIPLY', m_rust, n2.outputs['Fac'])
    bump = N(nt, 'ShaderNodeBump', (800, -400), inputs={'Strength': 0.5, 'Distance': 0.001})
    nt.links.new(h, bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def mat_rubber(name, rgb=(0.022, 0.021, 0.02), rough=0.78, mud=0.5, scale=1.0, mud_col=(0.15, 0.115, 0.08)):
    m, nt, b = new_mat(name, rgb)
    tc = N(nt, 'ShaderNodeTexCoord', (-1800, 0))
    co = tc.outputs['Object']
    n1 = noise(nt, (-1500, 300), co, 4 * scale, 10, 0.62, 0.3)
    n2 = noise(nt, (-1500, 0), co, 90 * scale, 4, 0.5)
    ao = N(nt, 'ShaderNodeAmbientOcclusion', (-1500, -300), samples=8, inputs={'Distance': 0.03})
    cav = ramp(nt, (-1250, -300), mth(nt, (-1350, -300), 'SUBTRACT', 1.0, ao.outputs['AO']),
               [(0.1, (0, 0, 0)), (0.5, (1, 1, 1))])
    t = 0.58 - 0.2 * mud
    mm = ramp(nt, (-1100, 300), n1.outputs['Fac'], [(t, (0, 0, 0)), (t + 0.12, (1, 1, 1))])
    mm = mth(nt, (-900, 200), 'MULTIPLY', mth(nt, (-1000, 100), 'MAXIMUM', mm, cav), mud, clamp=True)
    dust = mth(nt, (-900, 0), 'MULTIPLY', n1.outputs['Fac'], 0.35 * mud)
    tot = mth(nt, (-750, 100), 'MAXIMUM', mm, dust, clamp=True)
    mc = ramp(nt, (-900, -150), n2.outputs['Fac'], [(0.3, tuple(c * 0.7 for c in mud_col)), (0.7, tuple(c * 1.25 for c in mud_col))])
    c = mixc(nt, (-500, 300), tot, rgb, mc)
    nt.links.new(c, b.inputs['Base Color'])
    nt.links.new(mixf(nt, (-500, 0), tot, rough, 0.96), b.inputs['Roughness'])
    b.inputs['Specular IOR Level'].default_value = 0.35
    h = mth(nt, (-600, -500), 'ADD', mth(nt, (-750, -500), 'MULTIPLY', n2.outputs['Fac'], 0.25), mth(nt, (-750, -600), 'MULTIPLY', mm, n1.outputs['Fac']))
    bump = N(nt, 'ShaderNodeBump', (800, -400), inputs={'Strength': 0.35, 'Distance': 0.0015})
    nt.links.new(h, bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def mat_leather(name, rgb=(0.028, 0.023, 0.02)):
    m, nt, b = new_mat(name, rgb)
    tc = N(nt, 'ShaderNodeTexCoord', (-1800, 0))
    co = tc.outputs['Object']
    vor = N(nt, 'ShaderNodeTexVoronoi', (-1500, 300), feature='DISTANCE_TO_EDGE')
    nt.links.new(co, vor.inputs['Vector'])
    vor.inputs['Scale'].default_value = 180
    n1 = noise(nt, (-1500, 0), co, 6, 10, 0.6, 0.2)
    wear = ramp(nt, (-1100, 0), n1.outputs['Fac'], [(0.5, (0, 0, 0)), (0.7, (1, 1, 1))])
    c = mixc(nt, (-700, 200), wear, rgb, (0.09, 0.075, 0.06))
    nt.links.new(c, b.inputs['Base Color'])
    nt.links.new(mixf(nt, (-700, 0), wear, 0.48, 0.75), b.inputs['Roughness'])
    b.inputs['Sheen Weight'].default_value = 0.1
    grain = ramp(nt, (-1100, 300), vor.outputs['Distance'], [(0.0, (0, 0, 0)), (0.08, (1, 1, 1))])
    bump = N(nt, 'ShaderNodeBump', (800, -400), inputs={'Strength': 0.25, 'Distance': 0.0006})
    nt.links.new(grain, bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'], b.inputs['Normal'])
    return m


def mat_simple(name, rgb, rough=0.5, metal=0.0, transmission=0.0, ior=1.45, emission=None,
               coat=0.0, alpha=1.0):
    m, nt, b = new_mat(name, rgb)
    b.inputs['Base Color'].default_value = (*rgb, 1.0)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    b.inputs['Transmission Weight'].default_value = transmission
    b.inputs['IOR'].default_value = ior
    b.inputs['Coat Weight'].default_value = coat
    if emission:
        b.inputs['Emission Color'].default_value = (*emission[0], 1.0)
        b.inputs['Emission Strength'].default_value = emission[1]
    return m


def build_materials():
    Y = P['yellow']
    M = {}
    M['paint_yellow'] = mat_paint("M_Paint_Yellow_Worn", Y, chip=0.62, rust=0.65, dirt=0.8, scale=1.2, scratches=0.8, streak=0.7, dust=0.45)
    M['paint_yellow_heavy'] = mat_paint("M_Paint_Yellow_Battered", Y, chip=0.9, rust=0.85, dirt=0.95, scale=1.5, scratches=1.0, streak=0.85)
    M['paint_red'] = mat_paint("M_Paint_Red_Worn", (0.35, 0.025, 0.015), chip=0.7, rust=0.7, dirt=0.4, scale=3)
    M['paint_black'] = mat_paint("M_Paint_Black_Worn", (0.025, 0.025, 0.025), chip=0.5, rust=0.6, dirt=0.4, scale=2, rough=0.45)
    M['emblem'] = mat_paint("M_Emblem_Ink", (0.014, 0.012, 0.01), rough=0.5, chip=0.75, rust=0.0, dirt=0.4, scale=4,
                            coat=0.1, primer=Y, metal=Y, scratches=1.0, streak=0.3, dust=0.4, smudge=0.3)
    M['steel_rusty'] = mat_metal("M_Steel_Rusty", (0.50, 0.48, 0.45), 0.4, rust=0.7, grime=0.55, scale=3, dust=0.15,
                                 rust_cols=((0.05, 0.022, 0.01), (0.17, 0.065, 0.025), (0.32, 0.14, 0.045)))
    M['steel_heavy_rust'] = mat_metal("M_Steel_HeavyRust", (0.42, 0.40, 0.38), 0.45, rust=1.05, grime=0.55, scale=2.2, dust=0.15,
                                      rust_cols=((0.04, 0.018, 0.01), (0.14, 0.055, 0.022), (0.30, 0.13, 0.045)))
    M['copper_rust'] = mat_metal("M_Pipe_CopperRust", (0.45, 0.22, 0.12), 0.4, rust=0.9, grime=0.45, scale=6, dust=0.1,
                                 rust_cols=((0.035, 0.016, 0.009), (0.11, 0.045, 0.02), (0.24, 0.10, 0.035)))
    M['chrome'] = mat_metal("M_Chrome_Pitted", (0.80, 0.80, 0.79), 0.09, rust=0.32, grime=0.45, scale=7, dust=0.15,
                            rust_cols=((0.05, 0.025, 0.012), (0.16, 0.06, 0.02), (0.28, 0.12, 0.04)))
    M['steel_bright'] = mat_metal("M_Steel_Machined", (0.70, 0.70, 0.69), 0.22, rust=0.3, grime=0.3, scale=3, aniso=0.4)
    M['alu_cast'] = mat_metal("M_Alu_Cast_Dirty", (0.55, 0.55, 0.54), 0.48, rust=0.25, grime=0.65, scale=2,
                              rust_cols=((0.18, 0.17, 0.15), (0.30, 0.28, 0.24), (0.35, 0.22, 0.12)))
    M['alu_fins'] = mat_metal("M_Alu_Fins", (0.62, 0.62, 0.61), 0.38, rust=0.45, grime=0.55, scale=2.5,
                              rust_cols=((0.12, 0.10, 0.08), (0.30, 0.13, 0.05), (0.40, 0.25, 0.12)))
    M['dark_metal'] = mat_metal("M_DarkMetal_Grimy", (0.10, 0.10, 0.10), 0.45, rust=0.2, grime=0.4, scale=4, dust=0.2)
    M['zinc'] = mat_metal("M_Hardware_Zinc", (0.6, 0.58, 0.52), 0.35, rust=0.5, grime=0.3, scale=20)
    M['rubber_tire'] = mat_rubber("M_Tire_Rubber_Muddy", mud=0.6)
    M['rubber'] = mat_rubber("M_Rubber_Black", mud=0.15, rough=0.7, scale=3)
    M['hose'] = mat_rubber("M_Hose_Rubber", rgb=(0.03, 0.028, 0.025), mud=0.25, rough=0.6, scale=6)
    M['leather'] = mat_leather("M_Seat_Leather")
    M['plastic_black'] = mat_simple("M_Plastic_Black", (0.02, 0.02, 0.02), rough=0.55)
    M['glass'] = mat_simple("M_Glass_Lens", (0.95, 0.95, 0.95), rough=0.03, transmission=1.0, ior=1.5)
    M['reflector'] = mat_simple("M_Reflector_Chrome", (0.9, 0.9, 0.9), rough=0.06, metal=1.0)
    M['lens_orange'] = mat_simple("M_Lens_Amber", (0.95, 0.35, 0.02), rough=0.15, transmission=0.6, ior=1.5,
                                  emission=((1.0, 0.35, 0.02), 0.0))
    M['lens_red'] = mat_simple("M_Lens_Red", (0.6, 0.02, 0.01), rough=0.12, transmission=0.6, ior=1.5,
                               emission=((1.0, 0.02, 0.01), 0.0))
    M['bulb'] = mat_simple("M_Bulb", (1, 0.95, 0.8), rough=0.1, emission=((1.0, 0.9, 0.7), 0.0))
    M['wire_red'] = mat_simple("M_Wire_Red", (0.4, 0.02, 0.015), rough=0.4)
    M['wire_black'] = mat_simple("M_Wire_Black", (0.02, 0.02, 0.02), rough=0.45)
    M['wire_yellow'] = mat_simple("M_Wire_Yellow", (0.6, 0.45, 0.02), rough=0.45)
    M['braided'] = mat_metal("M_BraidedLine", (0.5, 0.5, 0.5), 0.45, rust=0.3, grime=0.6, scale=40)
    for k, rgb in (('green', (0.03, 0.25, 0.04)), ('blue', (0.02, 0.06, 0.35)), ('brown', (0.12, 0.05, 0.02)),
                   ('white', (0.7, 0.7, 0.66)), ('orange', (0.7, 0.2, 0.01)), ('grey', (0.25, 0.25, 0.25))):
        M['wire_' + k] = mat_simple("M_Wire_" + k.capitalize(), rgb, rough=0.42)
    M['loom'] = mat_rubber("M_Loom_Sleeve", rgb=(0.018, 0.018, 0.018), mud=0.2, rough=0.62, scale=60)
    M['lens_green'] = mat_simple("M_Lens_Green", (0.05, 0.6, 0.1), rough=0.12, transmission=0.5, emission=((0.1, 1.0, 0.2), 0.0))
    M['lens_blue'] = mat_simple("M_Lens_Blue", (0.03, 0.12, 0.8), rough=0.12, transmission=0.5, emission=((0.1, 0.3, 1.0), 0.0))
    M['brass'] = mat_metal("M_Terminal_Brass", (0.75, 0.55, 0.25), 0.3, rust=0.35, grime=0.4, scale=40,
                           rust_cols=((0.12, 0.10, 0.04), (0.25, 0.3, 0.2), (0.3, 0.35, 0.25)))
    M['battery_case'] = mat_simple("M_Plastic_BatteryCase", (0.06, 0.06, 0.065), rough=0.5)
    M['ceramic'] = mat_simple("M_Ceramic", (0.85, 0.82, 0.75), rough=0.25)
    MAT_CACHE['ceramic'] = M['ceramic']
    return M


# ---------------------------------------------------------------------------------------------
# Hardware: shared-mesh bolts/nuts (edit one, all update; instance friendly for games)
# ---------------------------------------------------------------------------------------------

HW = {}


def hardware_meshes(M):
    # hex bolt head with washer, local +Z out of the surface
    bm = bmesh.new()
    bm_cyl(bm, 0.0085, 0.0085, 0.0015, 24)                                   # washer
    bm_cyl(bm, 0.0065, 0.0065, 0.0055, 6, Matrix.Translation((0, 0, 0.0015)))  # hex
    bm_cyl(bm, 0.0035, 0.0035, 0.0025, 12, Matrix.Translation((0, 0, 0.007)))  # stub
    me = bpy.data.meshes.new("HW_Bolt_M8")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M['zinc'])
    finish_mesh(me, True, 40)
    HW['bolt'] = me
    # socket cap screw
    bm = bmesh.new()
    bm_cyl(bm, 0.0055, 0.0055, 0.005, 20)
    bm_cyl(bm, 0.0026, 0.0026, 0.0035, 6, Matrix.Translation((0, 0, 0.0025)))
    me = bpy.data.meshes.new("HW_CapScrew_M6")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M['steel_rusty'])
    finish_mesh(me, True, 40)
    HW['cap'] = me
    # nut
    bm = bmesh.new()
    bm_cyl(bm, 0.011, 0.011, 0.009, 6)
    bm_cyl(bm, 0.006, 0.006, 0.004, 16, Matrix.Translation((0, 0, 0.009)))
    me = bpy.data.meshes.new("HW_AxleNut_M16")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(M['zinc'])
    finish_mesh(me, True, 40)
    HW['nut'] = me


def bolt(loc, normal, collection, kind='bolt', scale=1.0, name="HW_Bolt", spin=None):
    ob = bpy.data.objects.new(name, HW[kind])
    collection.objects.link(ob)
    R = frame_from_z(Vector(normal))
    R = R @ Matrix.Rotation(spin if spin is not None else random.uniform(0, pi), 3, 'Z')
    ob.matrix_world = mat4(loc, R) @ Matrix.Scale(scale, 4)
    return ob


def bolt_ring(center, normal, radius, count, collection, kind='bolt', scale=1.0, phase=0.0, name="HW_Bolt"):
    n = Vector(normal).normalized()
    R = frame_from_z(n)
    x, y = R.col[0], R.col[1]
    obs = []
    for i in range(count):
        a = phase + 2 * pi * i / count
        p = Vector(center) + (x * cos(a) + y * sin(a)) * radius
        obs.append(bolt(p, n, collection, kind, scale, name))
    return obs


# ---------------------------------------------------------------------------------------------
# WHEELS
# ---------------------------------------------------------------------------------------------

def tire_profile_fn(R, half_w, R_rim, lug_h, n=3.2, phimax=2.3):
    Rc = R - lug_h
    p = 2.0 / n

    def S(x):
        return math.copysign(abs(x) ** p, x)
    Rb = R_rim + 0.006
    k = -S(cos(phimax))
    h = (Rc - Rb) / (1 + k)
    r0 = Rc - h

    def at(phi):
        return half_w * S(sin(phi)), r0 + h * S(cos(phi))

    def normal(phi, e=1e-3):
        a1, r1 = at(phi - e)
        a2, r2 = at(phi + e)
        ta, tr = a2 - a1, r2 - r1
        L = math.hypot(ta, tr) or 1.0
        return -tr / L, ta / L, L / (2 * e)      # (na, nr, ds/dphi)
    return at, normal, phimax


def build_tire(prefix, axle, spec, col, M, lug_rot_phase=0.0):
    R, hw, Rr, lh, nl = spec['R'], spec['half_w'], spec['R_rim'], spec['lug_h'], spec['lugs']
    at, nrm, phimax = tire_profile_fn(R, hw, Rr, lh)
    # --- carcass: superellipse section + sidewall rib, revolved ---
    prof = []
    steps = 56
    for i in range(steps + 1):
        phi = -phimax + 2 * phimax * i / steps
        a, r = at(phi)
        na, nr, _ = nrm(phi)
        bump = 0.0032 * math.exp(-((abs(phi) - 1.92) / 0.06) ** 2) + 0.0018 * math.exp(-((abs(phi) - 1.72) / 0.03) ** 2)
        prof.append((a + na * bump, r + nr * bump))
    carcass = revolve_obj(prefix + "_Tire_Carcass", prof, col, M['rubber_tire'], mat4(axle), segs=128,
                          closed=False, axis='X', angle=60)

    # --- one tread pitch (two staggered chevron knobs); radial array repeats it ---
    dth = 2 * pi / nl
    bm = bmesh.new()

    def lug(phi0, phi1, th0, width, skew, sections=10, sink=0.004, inset=0.0045):
        secs = []
        for i in range(sections + 1):
            t = i / sections
            phi = phi0 + (phi1 - phi0) * t
            ap = abs(phi)
            a, r = at(phi)
            na, nr, dsd = nrm(phi)
            hgt = lh * (1.0 if ap < 1.05 else max(0.6, 1.0 - (ap - 1.05) * 0.75))
            # flare the knob slightly wider toward the shoulder
            wm = width * (1.0 + 0.18 * min(1, ap / 1.4))
            thc = th0 + skew * ap
            dphi_top = 0.0
            if i == 0:
                dphi_top = math.copysign(inset / dsd, phi1 - phi0)
            elif i == sections:
                dphi_top = -math.copysign(inset / dsd, phi1 - phi0)
            sec = []
            for (side, up) in ((-1, 0), (1, 0), (1, 1), (-1, 1)):
                ph = phi + (dphi_top if up else 0.0)
                aa, rr = at(ph)
                na2, nr2, _ = nrm(ph)
                hh = hgt if up else -sink
                rad_ = rr + nr2 * hh
                half = (wm / 2 - (inset if up else 0.0)) / max(rad_, 1e-3)
                th = thc + side * half
                sec.append(bm.verts.new((aa + na2 * hh, rad_ * cos(th), rad_ * sin(th))))
            secs.append(sec)
        for i in range(sections):
            s0, s1 = secs[i], secs[i + 1]
            for k in range(4):
                k1 = (k + 1) % 4
                bm.faces.new((s0[k], s0[k1], s1[k1], s1[k]))
        bm.faces.new(list(reversed(secs[0])))
        bm.faces.new(secs[-1])

    w_knob = 0.046 if hw > 0.11 else 0.042
    lug(0.06, 1.62, lug_rot_phase, w_knob, 0.22 * dth)
    lug(-0.06, -1.62, lug_rot_phase + dth / 2, w_knob, 0.22 * dth)
    # small centre tie-knob between the main knobs
    lug(-0.05, 0.05, lug_rot_phase + dth * 0.25, 0.022, 0.0, sections=2)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    lugs = obj_from_bm(prefix + "_Tire_Knobs", bm, col, M['rubber_tire'], mat4(axle), angle=40)
    mod_bevel(lugs, 0.0018, 2, 30, name="KnobEdgeBevel")
    radial_array(lugs, nl, axis=0, name="TreadArray")
    return carcass, lugs


def build_rim(prefix, axle, spec, col, M):
    Rr, hw = spec['R_rim'], spec['half_w']
    wr = 0.832 * hw + 0.004
    outer = [(-wr, Rr + 0.012), (-wr + 0.004, Rr + 0.0135), (-wr + 0.0065, Rr + 0.004), (-wr + 0.017, Rr + 0.002),
             (-0.35 * wr, Rr - 0.012), (0.35 * wr, Rr - 0.012), (wr - 0.017, Rr + 0.002), (wr - 0.0065, Rr + 0.004),
             (wr - 0.004, Rr + 0.0135), (wr, Rr + 0.012)]
    inner = [(wr + 0.002, Rr + 0.004), (wr, Rr - 0.012), (wr - 0.01, Rr - 0.02), (0.3 * wr, Rr - 0.026),
             (-0.3 * wr, Rr - 0.026), (-wr + 0.01, Rr - 0.02), (-wr, Rr - 0.012), (-wr - 0.002, Rr + 0.004)]
    rim = revolve_obj(prefix + "_Rim", outer + inner, col, M['paint_yellow_heavy'], mat4(axle), segs=96,
                      closed=True, axis='X', angle=35)
    mod_weighted_normals(rim)
    # valve stem
    vs = cyl_obj(prefix + "_ValveStem", axle + Vector((0.012, 0, -(Rr - 0.02))), axle + Vector((0.012, 0, -(Rr - 0.055))),
                 0.0035, col, M['steel_rusty'], segs=12)
    # wheel centre: cast dish with windows (boolean against a radial-arrayed cutter)
    t = 0.009
    pts = [(0.058, 0.0), (0.075, -0.010), (0.14, -0.014), (Rr - 0.05, -0.006), (Rr - 0.03, 0.0)]
    prof = [(a - t / 2, r) for r, a in pts] + [(0.012, Rr - 0.027), (0.012, Rr - 0.024), (-0.012, Rr - 0.024)]
    prof = [(a - t / 2, r) for r, a in pts]
    prof += [(-0.012, Rr - 0.026), (0.012, Rr - 0.026)]
    prof += [(a + t / 2, r) for r, a in reversed(pts)]
    centre = revolve_obj(prefix + "_WheelCentre", prof, col, M['dark_metal'], mat4(axle), segs=96, closed=True,
                         axis='X', angle=35)
    # window cutter (one window, arrayed)
    r_in, r_out = 0.085, Rr - 0.055
    wpts = []
    for i in range(7):
        a = -0.24 + 0.48 * i / 6
        wpts.append((r_out * cos(a), r_out * sin(a)))
    for i in range(7):
        a = 0.20 - 0.40 * i / 6
        wpts.append((r_in * cos(a), r_in * sin(a)))
    bm = bmesh.new()
    # extrude along X: polygon lies in local YZ -> build in XY then rotate
    extrude_bm(bm, wpts, 0.12)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    R = Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))  # local X->world Y, Y->Z, Z->X
    cutter = obj_from_bm(prefix + "_WheelCentre_WindowCutter", bm, COLL['Cutters'], None,
                         mat4(axle, R), smooth=False)
    radial_array(cutter, 6, axis=2, name="WindowArray")
    mod_boolean(centre, cutter, "Windows")
    mod_bevel(centre, 0.0015, 2, 30)
    # hub
    hub_prof = [(-0.075, 0.012), (-0.075, 0.03), (-0.06, 0.032), (-0.058, 0.07), (-0.05, 0.072), (-0.045, 0.045),
                (0.045, 0.045), (0.05, 0.072), (0.058, 0.07), (0.06, 0.032), (0.075, 0.03), (0.075, 0.012)]
    hub = revolve_obj(prefix + "_Hub", hub_prof, col, M['alu_cast'], mat4(axle), segs=48, closed=True, axis='X')
    mod_bevel(hub, 0.0012, 2, 40)
    bolt_ring(axle + Vector((-0.06, 0, 0)), (-1, 0, 0), 0.062, 6, col, 'cap', 1.0, name=prefix + "_HubBolt")
    bolt_ring(axle + Vector((0.06, 0, 0)), (1, 0, 0), 0.062, 6, col, 'cap', 1.0, name=prefix + "_HubBolt")
    return [rim, vs, centre, hub]


def build_rotor(prefix, axle, a, r_out, r_in, col, M, holes=18, windows=8, side=-1):
    th = 0.005
    prof = [(a - th / 2, r_in), (a - th / 2, r_out - 0.001), (a - th / 2 + 0.001, r_out), (a + th / 2 - 0.001, r_out),
            (a + th / 2, r_out - 0.001), (a + th / 2, r_in)]
    rotor = revolve_obj(prefix + "_BrakeRotor", prof, col, M['steel_bright'], mat4(axle), segs=128, closed=True, axis='X')
    band_in = r_out - (r_out - r_in) * 0.42
    # drilled holes (two staggered rows) - cutter is one pair of holes, arrayed
    bm = bmesh.new()
    for rr, ph in ((band_in + (r_out - band_in) * 0.35, 0.0), (band_in + (r_out - band_in) * 0.7, pi / holes)):
        M3 = Matrix.Rotation(pi / 2, 4, 'Y')
        bm_cyl(bm, 0.0038, 0.0038, 0.04, 12, Matrix.Translation((0, rr * cos(ph), rr * sin(ph))) @ M3 @ Matrix.Translation((0, 0, -0.02)))
    cut = obj_from_bm(prefix + "_Rotor_HoleCutter", bm, COLL['Cutters'], None, mat4(axle + Vector((a, 0, 0))), smooth=False)
    radial_array(cut, holes, axis=0, name="HoleArray")
    mod_boolean(rotor, cut, "DrilledHoles")
    # carrier windows (the big rectangular openings seen on the concept)
    w_in, w_out = r_in + 0.012, band_in - 0.008
    wp = []
    for i in range(5):
        aa = -0.17 + 0.34 * i / 4
        wp.append((w_out * cos(aa), w_out * sin(aa)))
    for i in range(5):
        aa = 0.14 - 0.28 * i / 4
        wp.append((w_in * cos(aa), w_in * sin(aa)))
    bm = bmesh.new()
    extrude_bm(bm, wp, 0.05)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    R = Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    wcut = obj_from_bm(prefix + "_Rotor_WindowCutter", bm, COLL['Cutters'], None, mat4(axle + Vector((a, 0, 0)), R), smooth=False)
    radial_array(wcut, windows, axis=2, name="WindowArray")
    mod_boolean(rotor, wcut, "CarrierWindows")
    mod_bevel(rotor, 0.0006, 1, 30)
    # floating-disc bobbins at the band / carrier boundary
    bm = bmesh.new()
    rb = band_in - 0.002
    bm_cyl(bm, 0.0055, 0.0055, th + 0.004, 16, Matrix.Translation((-(th + 0.004) / 2, rb, 0)) @ Matrix.Rotation(pi / 2, 4, 'Y'))
    bob = obj_from_bm(prefix + "_Rotor_Bobbins", bm, col, M['alu_cast'], mat4(axle + Vector((a, 0, 0))))
    radial_array(bob, windows, axis=0, name="BobbinArray")
    bolt_ring(axle + Vector((a + side * th / 2, 0, 0)), (side, 0, 0), r_in + 0.006, 6, col, 'cap', 0.9,
              phase=0.3, name=prefix + "_RotorBolt")
    return rotor


def sprocket_pts(N, r_p, depth=0.0065):
    pts = []
    d = 2 * pi / N
    rr, rt = r_p - depth, r_p + depth * 0.95
    for k in range(N):
        c = k * d
        for f, r in ((-0.5, rr - 0.0008), (-0.36, rr), (-0.2, (rr + rt) / 2 - 0.001), (-0.1, rt), (0.1, rt),
                     (0.2, (rr + rt) / 2 - 0.001), (0.36, rr)):
            pts.append((r * cos(c + f * d), r * sin(c + f * d)))
    return pts


def build_sprocket(name, center, N, r_p, thick, col, M, hub_r=0.035):
    R = Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    ob = extrude_obj(name, sprocket_pts(N, r_p), thick, col, M['steel_rusty'], mat4(center, R), bevel=0.0006)
    if r_p > 0.06:
        # lightening holes
        bm = bmesh.new()
        extrude_bm(bm, circle_pts(0.018, 20, 0.062, 0.0), 0.05)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        cut = obj_from_bm(name + "_HoleCutter", bm, COLL['Cutters'], None, mat4(center, R), smooth=False)
        radial_array(cut, 6, axis=2, name="HoleArray")
        mod_boolean(ob, cut, "LighteningHoles")
        bolt_ring(center + Vector((thick / 2, 0, 0)), (1, 0, 0), 0.042, 6, col, 'nut', 0.6, name=name + "_Nut")
    return ob


def build_wheel(prefix, axle, spec, col, M, rotor=None, sprocket=None, axle_half=0.16, tone_ring=False):
    pivot = bpy.data.objects.new(prefix + "_Pivot", None)
    pivot.empty_display_type = 'CIRCLE'
    pivot.empty_display_size = spec['R'] + 0.04
    pivot.rotation_mode = 'XYZ'
    col.objects.link(pivot)
    pivot.matrix_world = mat4(axle)   # local X == axle, so rotating local X spins the wheel
    before = set(col.objects)
    build_tire(prefix, axle, spec, col, M)
    build_rim(prefix, axle, spec, col, M)
    if rotor:
        build_rotor(prefix, axle, rotor['a'], rotor['r_out'], rotor['r_in'], col, M, rotor.get('holes', 18),
                    rotor.get('windows', 8), side=-1)
        if abs(rotor['a']) > 0.08:       # rotor sits outboard of the hub flange: spacer carrier
            revolve_obj(prefix + "_RotorSpacer", [(-0.075, 0.03), (-0.075, 0.058), (rotor['a'] + 0.003, 0.058), (rotor['a'] + 0.003, 0.03)],
                        col, M['alu_cast'], mat4(axle), segs=40, axis='X', closed=True)
    if sprocket:
        build_sprocket(prefix + "_Sprocket", axle + Vector((sprocket['a'], 0, 0)), sprocket['N'], sprocket['r_p'],
                       0.006, col, M)
        # cush-drive sprocket carrier: brings the sprocket out to the chain line, clear of the fat tyre
        a1 = sprocket['a'] - 0.003
        cprof = [(0.06, 0.03), (0.06, 0.072), (0.07, 0.074), (a1 - 0.012, 0.074), (a1 - 0.008, 0.058), (a1, 0.056), (a1, 0.03)]
        revolve_obj(prefix + "_SprocketCarrier", cprof, col, M['alu_cast'], mat4(axle), segs=48, axis='X', closed=True)
        bolt_ring(axle + Vector((0.07, 0, 0)), (1, 0, 0), 0.066, 6, col, 'cap', 0.8, phase=0.2, name=prefix + "_CarrierBolt")
    if tone_ring:
        R = Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
        extrude_obj(prefix + "_ToneRing", sprocket_pts(40, 0.064, 0.0028), 0.003, col, M['steel_rusty'],
                    mat4(axle + Vector((0.079, 0, 0)), R), bevel=0.0004)
    # axle + nuts
    cyl_obj(prefix + "_Axle", axle + Vector((-axle_half, 0, 0)), axle + Vector((axle_half, 0, 0)), 0.0095, col,
            M['steel_bright'], segs=20)
    bolt(axle + Vector((-axle_half, 0, 0)), (-1, 0, 0), col, 'nut', 1.0, prefix + "_AxleNut")
    bolt(axle + Vector((axle_half, 0, 0)), (1, 0, 0), col, 'nut', 0.9, prefix + "_AxleCap")
    for ob in set(col.objects) - before:
        if ob.parent is None and ob is not pivot:
            parent_later(ob, pivot)
    return pivot


# ---------------------------------------------------------------------------------------------
# FRONT END (steering, forks, front wheel, headlight)
# ---------------------------------------------------------------------------------------------

def make_empty(name, matrix, col, kind='PLAIN_AXES', size=0.1):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = kind
    e.empty_display_size = size
    col.objects.link(e)
    e.matrix_world = matrix
    return e


class ForkFrame:
    """Local frame of the fork: X lateral, Y backwards (perpendicular to the tubes), Z up the tubes."""

    def __init__(self):
        self.rake = rad(P['rake'])
        self.A = P['front_axle'].copy()
        self.d = Vector((0, sin(self.rake), cos(self.rake)))
        self.nb = Vector((0, cos(self.rake), -sin(self.rake)))
        self.R = Matrix.Rotation(-self.rake, 3, 'X')

    def p(self, t, y=0.0, x=0.0):
        return self.A + self.d * t + self.nb * y + Vector((x, 0, 0))

    def M(self, t, y=0.0, x=0.0, extra=None):
        R = self.R if extra is None else self.R @ extra
        return mat4(self.p(t, y, x), R)


A = {}             # named anchor points shared between sub-assemblies: name -> (Vector, group)


def anchor(name, p, group='static'):
    A[name] = (Vector(p), group)
    return Vector(p)


def build_front(M, root):
    col = coll('FrontEnd', COLL['Bike'])
    wcol = coll('Wheels', COLL['Bike'])
    F = ForkFrame()
    off, fx = P['fork_offset'], P['fork_half_span']
    yb, yt = P['yokes']
    Rf = F.R
    ax = make_empty("FrontEnd_SteeringAxis", F.M(0.70, off), col, 'SINGLE_ARROW', 0.35)
    steer = make_empty("FrontEnd_Steer", F.M(0.70, off), col, 'CIRCLE', 0.12)
    travel = make_empty("FrontEnd_ForkTravel", F.M(0.0), col, 'SINGLE_ARROW', 0.15)
    parent_later(ax, root)
    parent_later(steer, ax)
    parent_later(travel, steer)
    S, T = [], []

    # --- stanchions (fixed in the yokes), caps ---
    bm = bmesh.new()
    bm_cyl(bm, STAN_R, STAN_R, 0.60, 40, Matrix.Translation((-fx, 0, 0.0)))
    stan = obj_from_bm("Fork_Stanchions", bm, col, M['chrome'], F.M(0.30), angle=50)
    mod_mirror(stan)
    S.append(stan)
    prof = [(0.0, 0.0), (0.0, 0.0285), (0.004, 0.0295), (0.022, 0.0295), (0.026, 0.024), (0.03, 0.012), (0.042, 0.012),
            (0.043, 0.006), (0.043, 0.0)]
    cap = revolve_obj("Fork_TopCaps", prof, col, M['steel_rusty'], F.M(0.895), segs=6, closed=False, axis='Z', angle=20)
    for v in cap.data.vertices:
        v.co.x -= fx
    mod_mirror(cap)
    S.append(cap)

    # --- yokes ---
    def yoke(name, t, thick, extra_pts=()):
        pts = []
        for cx in (-fx, fx):
            pts += circle_pts(0.046, 24, cx, 0.0)
        pts += circle_pts(0.04, 24, 0.0, off)
        pts += list(extra_pts)
        hull = convex_hull([(round(x, 6), round(y, 6)) for x, y in pts])
        ob = extrude_obj(name, hull, thick, col, M['alu_cast'], F.M(t), bevel=0.003)
        bm = bmesh.new()
        for cx in (-fx, fx):
            bm_cyl(bm, STAN_R + 0.0004, STAN_R + 0.0004, thick * 3, 32, Matrix.Translation((cx, 0, -thick * 1.5)))
        bm_cyl(bm, 0.018, 0.018, thick * 3, 24, Matrix.Translation((0, off, -thick * 1.5)))
        cut = obj_from_bm(name + "_BoreCutter", bm, COLL['Cutters'], None, F.M(t), smooth=False)
        mod_boolean(ob, cut, "Bores")
        ob.modifiers.move(len(ob.modifiers) - 1, 0)
        for cx in (-fx, fx):
            for dz in (-thick * 0.22, thick * 0.22):
                S.append(bolt(F.p(t + dz, 0.047, cx * 0.86), F.nb, col, 'cap', 1.0, name + "_PinchBolt"))
        return ob
    S.append(yoke("Fork_BottomYoke", yb, 0.034))
    S.append(yoke("Fork_TopYoke", yt, 0.026, circle_pts(0.03, 12, 0.045, off + 0.035) + circle_pts(0.03, 12, -0.045, off + 0.035)))
    S.append(cyl_obj("Fork_SteeringStemNut", F.p(yt + 0.013, off), F.p(yt + 0.03, off), 0.02, col, M['steel_rusty'], segs=6, bevel=0.001))
    S.append(cyl_obj("Fork_SteeringStem", F.p(yb - 0.02, off), F.p(yt + 0.012, off), 0.017, col, M['steel_rusty'], segs=24))
    for sx in (-1, 1):      # steering stop bolts (hit the frame lugs at full lock)
        S.append(bolt(F.p(yb, off + 0.038, sx * 0.05), F.nb, col, 'bolt', 1.0, "Fork_SteeringStopBolt"))

    # --- sliders, seals, axle lugs: ride on the fork travel ---
    prof = [(-0.045, 0.0), (-0.045, 0.026), (-0.03, 0.034), (0.25, 0.036), (0.36, 0.038), (0.372, SLIDER_R), (0.385, SLIDER_R),
            (0.385, 0.0285), (0.36, 0.0285)]
    sl = revolve_obj("Fork_Sliders", prof, col, M['paint_yellow_heavy'], F.M(0.0), segs=40, closed=False, axis='Z', angle=40)
    for v in sl.data.vertices:
        v.co.x -= fx
    mod_mirror(sl)
    T.append(sl)
    seal = revolve_obj("Fork_DustSeals", [(0.383, 0.0285), (0.384, 0.0365), (0.392, 0.037), (0.402, 0.031), (0.404, 0.027)],
                       col, M['rubber'], F.M(0.0), segs=40, closed=False, axis='Z')
    for v in seal.data.vertices:
        v.co.x -= fx
    mod_mirror(seal)
    T.append(seal)
    bm = bmesh.new()
    bm_box(bm, (0.05, 0.07, 0.07), Matrix.Translation((-fx, 0.004, -0.012)))
    bm_cyl(bm, 0.03, 0.03, 0.06, 32, Matrix.Translation((-fx - 0.03, 0, 0)) @ Matrix.Rotation(pi / 2, 4, 'Y'))
    lug = obj_from_bm("Fork_AxleLugs", bm, col, M['paint_yellow_heavy'], F.M(0.0), angle=40)
    mod_bevel(lug, 0.006, 3, 40)
    mod_mirror(lug)
    T.append(lug)
    for sx in (-1, 1):
        for dz in (-0.022, -0.042):
            T.append(bolt(F.p(dz, 0.035, sx * (fx + 0.012)), F.nb, col, 'cap', 1.0, "Fork_AxlePinchBolt"))
    # fork brace: yellow arch plate bolted to the front of the sliders, the fender hangs from it
    W = fx + 0.03
    arch = [(-W, 0.26), (-W, 0.36), (-W + 0.06, 0.448), (W - 0.06, 0.448), (W, 0.36), (W, 0.26),
            (W - 0.05, 0.26), (W - 0.05, 0.35), (W - 0.095, 0.424), (-(W - 0.095), 0.424), (-(W - 0.05), 0.35), (-(W - 0.05), 0.26)]
    br = extrude_obj("Fork_Brace", arch, 0.012, col, M['paint_yellow'], F.M(0.0, 0, 0, Matrix.Rotation(pi / 2, 3, 'X')) @
                     Matrix.Translation((0, 0, 0.044)), bevel=0.003)
    T.append(br)
    for sx in (-1, 1):
        for z in (0.28, 0.335):
            T.append(bolt(F.p(z, -0.05, sx * fx), -F.nb, col, 'bolt', 1.0, "Fork_BraceBolt"))
        T.append(bolt(F.p(0.436, -0.05, sx * 0.05), -F.nb, col, 'bolt', 0.9, "Fender_MountBolt"))
    # front fender: U-section swept around the wheel, narrower than the gap between the sliders
    A_ = F.A
    Rfd = P['front_tire']['R'] + 0.026
    hwf = fx - SLIDER_R - 0.0085
    sec = [(-hwf, -0.03), (-hwf + 0.0015, -0.011), (-hwf + 0.01, 0.002), (-0.07, 0.010), (0.0, 0.013), (0.07, 0.010),
           (hwf - 0.01, 0.002), (hwf - 0.0015, -0.011), (hwf, -0.03)]
    bm = bmesh.new()
    nb_ = 30
    rows = []
    for i in range(nb_ + 1):
        beta = rad(38 + (124 - 38) * i / nb_)
        taper = 1.0 - 0.18 * (1 - i / nb_) ** 2
        rows.append([bm.verts.new(Vector((x * taper, -cos(beta) * (Rfd + ro), sin(beta) * (Rfd + ro)))) for x, ro in sec])
    for i in range(nb_):
        for j in range(len(sec) - 1):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    fender = obj_from_bm("Fender_Front", bm, col, M['paint_yellow'], mat4(A_), angle=60)
    mod_solidify(fender, 0.0035, 1)
    mod_subsurf(fender, 1, 2)
    mod_weighted_normals(fender)
    T.append(fender)

    # --- front brake caliper (right), banjo, hose guide ---
    rot_a = -0.078
    g = rad(32)
    cc = A_ + Vector((rot_a, 0.128 * cos(g), 0.128 * sin(g)))
    tg = Vector((0, -sin(g), cos(g)))
    rd = Vector((0, cos(g), sin(g)))
    T += caliper("Brake_Front", A_, rot_a, 0.152, g, col, M)
    banjo = A_ + Vector((rot_a - 0.027, 0, 0)) + rd * 0.128 + tg * 0.03
    T += banjo_fitting("Brake_Front_CaliperBanjo", banjo, (-1, 0, 0), col, M)
    anchor('caliper_banjo_F', banjo + Vector((-0.012, 0, 0.004)), 'travel')
    brx = -(fx - SLIDER_R) + 0.016
    T.append(box_obj("Brake_Front_CaliperBracket", F.p(0.10, 0.05, brx), (abs(brx) - abs(rot_a) + 0.0, 0.06, 0.075), col, M['alu_cast'],
                     rot=(-P['rake'], 0, 0), bevel=0.004))
    for z in (0.085, 0.125):
        T.append(bolt(F.p(z, 0.07, -(fx - SLIDER_R) + 0.004), (-1, 0, 0), col, 'bolt', 1.0, "Brake_Front_MountBolt"))
    gp = F.p(0.30, 0.0, -(fx + SLIDER_R + 0.006))
    T.append(cyl_obj("Brake_Front_HoseGuide", gp - F.d * 0.008, gp + F.d * 0.008, 0.0075, col, M['rubber'], segs=16))
    anchor('hose_guide_F', gp, 'travel')
    # --- wheel speed sensor reading the tone ring (left) ---
    sp_ = A_ + Vector((0.09, 0.064 * cos(rad(-35)), 0.064 * sin(rad(-35))))
    T.append(cyl_obj("Elec_SpeedSensor", sp_ + Vector((-0.006, 0, 0)), sp_ + Vector((0.016, 0, 0)), 0.0065, col, M['plastic_black'], segs=16, bevel=0.001))
    T.append(box_obj("Elec_SpeedSensorBracket", sp_ + Vector((0.03, 0.005, 0.004)), (fx - SLIDER_R - 0.09 - 0.01, 0.022, 0.006), col,
                     M['steel_rusty'], bevel=0.0015))
    anchor('speed_sensor', sp_ + Vector((0.018, 0, 0)), 'travel')

    # --- front wheel ---
    fw = build_wheel("FrontWheel", A_, P['front_tire'], wcol, M,
                     rotor=dict(a=rot_a, r_out=0.152, r_in=0.062, holes=20, windows=8), axle_half=fx + 0.03, tone_ring=True)
    parent_later(fw, travel)

    # --- headlight: bucket, chrome rim, fresnel lens, reflector, H4 bulb, grommets ---
    hc = F.p(0.67, -0.135)
    hc = Vector((0, hc.y, hc.z))
    SC = 1.14
    bucket = [(0.085, 0.014), (0.083, 0.03), (0.075, 0.05), (0.058, 0.068), (0.035, 0.08), (0.005, 0.0865),
              (-0.012, 0.0875), (-0.012, 0.0835), (0.004, 0.0825)]
    hl = []
    hl.append(revolve_obj("Headlight_Bucket", bucket, col, M['chrome'], mat4(hc), segs=64, axis='Y', angle=40))
    rim = [(-0.008, 0.0835), (-0.010, 0.091), (-0.02, 0.094), (-0.03, 0.093), (-0.036, 0.088), (-0.036, 0.081), (-0.03, 0.08), (-0.025, 0.082)]
    hl.append(revolve_obj("Headlight_Rim", rim, col, M['chrome'], mat4(hc), segs=64, axis='Y', closed=True, angle=40))
    lens = [(-0.045, 0.0), (-0.0445, 0.02)]
    for i in range(1, 9):
        r = 0.02 + 0.0075 * i
        lens += [(-0.0445 + 0.0015 * i, r - 0.0045), (-0.0430 + 0.0015 * i, r)]
    lens += [(-0.031, 0.083)]
    hl.append(revolve_obj("Headlight_Lens", lens, col, M['glass'], mat4(hc), segs=64, axis='Y', angle=25))
    refl = [(0.05, 0.012)] + [(0.05 - 0.072 * (k / 10) ** 2, 0.012 + 0.068 * (k / 10)) for k in range(1, 11)]
    hl.append(revolve_obj("Headlight_Reflector", refl, col, M['reflector'], mat4(hc), segs=64, axis='Y', angle=60))
    bulb = bmesh.new()
    bmesh.ops.create_uvsphere(bulb, u_segments=16, v_segments=10, radius=0.011, matrix=Matrix.Translation((0, 0.035, 0)))
    bm_cyl(bulb, 0.009, 0.009, 0.02, 16, Matrix.Translation((0, 0.04, 0)) @ Matrix.Rotation(-pi / 2, 4, 'X'))
    hl.append(obj_from_bm("Headlight_Bulb", bulb, col, M['bulb'], mat4(hc)))
    for o in hl:
        o.data.transform(Matrix.Scale(SC, 4))
    S += hl
    S.append(bolt(hc + Vector((0, -0.032 * SC, -0.093 * SC)), (0, -0.3, -1), col, 'cap', 0.7, "Headlight_RimScrew"))
    # rubber grommets in the back of the bucket: loom (bottom) and handlebar wiring (top)
    for nm, gp_, ad in (("Low", Vector((0, 0.074, -0.05)), Vector((0, 0.55, -0.83))), ("Top", Vector((0, 0.074, 0.05)), Vector((0, 0.55, 0.83))),
                        ("Bottom", Vector((0, 0.03, -0.083)), Vector((0, 0.25, -0.97)))):
        gpos = hc + gp_ * SC
        S.append(revolve_obj("Headlight_Grommet" + nm, [(-0.002, 0.0055), (0.004, 0.0085), (0.008, 0.0085), (0.01, 0.006), (0.01, 0.004), (-0.002, 0.004)],
                             col, M['rubber'], mat4(gpos, frame_from_z(ad)), segs=20, axis='Z', closed=True))
        anchor('headlight_grommet_' + nm.lower(), gpos + ad.normalized() * 0.008, 'steer')
    # ears: clamps on the stanchions + bent straps to the bucket, amber markers
    for sx in (-1, 1):
        clamp_c = F.p(0.665, 0, sx * fx)
        bm = bmesh.new()
        bm_cyl(bm, 0.036, 0.036, 0.034, 32, Matrix.Translation((0, 0, -0.017)))
        cl = obj_from_bm("Headlight_ForkClamp" + ("_L" if sx > 0 else "_R"), bm, col, M['paint_yellow'], mat4(clamp_c, Rf))
        mod_bevel(cl, 0.003, 2, 40)
        S.append(cl)
        S.append(bolt(clamp_c + F.nb * 0.037 + Vector((sx * 0.0, 0, 0)), F.nb, col, 'cap', 0.9, "Headlight_ClampBolt"))
        ear_a = clamp_c + Vector((-sx * 0.012, -0.03, 0.0))
        ear_b = hc + Vector((sx * 0.1, 0.014, 0.0))
        anchor('ear_a_' + ('L' if sx > 0 else 'R'), ear_a, 'steer')
        anchor('ear_b_' + ('L' if sx > 0 else 'R'), ear_b, 'steer')
        S.append(curve_obj("Headlight_Ear" + ("_L" if sx > 0 else "_R"),
                           [ear_a, (ear_a + ear_b) / 2 + Vector((sx * 0.012, 0, 0.012)), ear_b], 0.009, col, M['paint_yellow'], mode='smooth'))
        S.append(bolt(ear_b + Vector((sx * 0.004, 0, 0)), (sx, 0, 0), col, 'bolt', 1.3, "Headlight_EarBolt"))
        mk = clamp_c + Vector((sx * 0.055, -0.02, 0.012))
        S.append(cyl_obj("Indicator_Front_Stalk", clamp_c + Vector((sx * 0.034, -0.01, 0.01)), mk, 0.006, col, M['rubber'], segs=12))
        S.append(box_obj("Indicator_Front_Housing", mk + Vector((sx * 0.012, -0.012, 0)), (0.03, 0.032, 0.026), col, M['plastic_black'], bevel=0.004))
        S.append(box_obj("Indicator_Front_Lens", mk + Vector((sx * 0.012, -0.029, 0)), (0.026, 0.006, 0.022), col, M['lens_orange'], bevel=0.0025))
        anchor('ind_front_' + ('L' if sx > 0 else 'R'), clamp_c + Vector((sx * 0.034, -0.008, 0.004)), 'steer')
    anchor('headlight_c', hc, 'steer')
    for ob in S:
        parent_later(ob, steer)
    for ob in T:
        parent_later(ob, travel)
    return F, steer, travel


# ---------------------------------------------------------------------------------------------
# COCKPIT (bars, switchgear, levers, master cylinder, mirrors, instrument pod, ignition)
# ---------------------------------------------------------------------------------------------

def load_image(name):
    path = os.path.join(TEXDIR, name) if TEXDIR else ""
    if path and os.path.exists(path):
        img = bpy.data.images.load(path, check_existing=True)
        try:
            img.pack()
        except RuntimeError:
            pass
        return img
    print("WARNING: texture not found", path, "- run scripts/make_textures.py")
    return None


def mat_gauge(name, img, emit=None, emit_strength=1.2):
    m, nt, b = new_mat(name, (0.05, 0.05, 0.05))
    b.inputs['Roughness'].default_value = 0.42
    if img:
        t = N(nt, 'ShaderNodeTexImage', (-500, 200), image=img)
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
    if emit:
        e = N(nt, 'ShaderNodeTexImage', (-500, -200), image=emit)
        nt.links.new(e.outputs['Color'], b.inputs['Emission Color'])
        b.inputs['Emission Strength'].default_value = emit_strength
    return m


def disc_uv_bm(r, segs=64, z=0.0):
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    c = bm.verts.new((0, 0, z))
    ring = [bm.verts.new((r * cos(2 * pi * i / segs), r * sin(2 * pi * i / segs), z)) for i in range(segs)]
    for i in range(segs):
        f = bm.faces.new((c, ring[i], ring[(i + 1) % segs]))
        for lp in f.loops:
            lp[uv].uv = (0.5 + lp.vert.co.x / (2 * r), 0.5 + lp.vert.co.y / (2 * r))
    return bm


def build_cockpit(M, F, steer):
    col = coll('Cockpit', COLL['Bike'])
    S = []
    off = P['fork_offset']
    yt = P['yokes'][1]
    # --- risers + split clamps on the top yoke ---
    rp = F.p(yt + 0.013, off + 0.03)
    for sx in (-1, 1):
        base = rp + Vector((sx * 0.036, 0, 0))
        S.append(cyl_obj("Bar_RiserDamper", base, base + Vector((0, 0, 0.007)), 0.018, col, M['rubber'], segs=24, bevel=0.001))
        S.append(cyl_obj("Bar_Riser", base + Vector((0, 0, 0.007)), base + Vector((0, 0, 0.055)), 0.014, col, M['alu_cast'], segs=24, bevel=0.001))
        S.append(box_obj("Bar_ClampLower", base + Vector((0, 0, 0.061)), (0.036, 0.04, 0.013), col, M['alu_cast'], bevel=0.004, segs=2))
        S.append(box_obj("Bar_ClampCap", base + Vector((0, 0, 0.0755)), (0.036, 0.04, 0.0135), col, M['alu_cast'], bevel=0.0045, segs=3))
        for dy in (-0.013, 0.013):
            S.append(bolt(base + Vector((0, dy, 0.0825)), (0, 0, 1), col, 'cap', 0.9, "Bar_ClampBolt"))
    T0 = rp + Vector((0, 0, 0.068))
    anchor('bar_center', T0, 'steer')
    ty, tz = T0.y, T0.z
    bar_pts = [(-0.39, ty + 0.115, tz + 0.055), (-0.25, ty + 0.05, tz + 0.035), (-0.15, ty, tz),
               (0.15, ty, tz), (0.25, ty + 0.05, tz + 0.035), (0.39, ty + 0.115, tz + 0.055)]
    S.append(curve_obj("Bar_Handlebar", bar_pts, 0.0125, col, M['steel_rusty'], mode='bend', bend=0.07, res=16))
    S.append(curve_obj("Bar_CrossBrace", [(-0.14, ty + 0.012, tz + 0.055), (0.14, ty + 0.012, tz + 0.055)], 0.009, col, M['steel_rusty']))
    for sx in (-1, 1):
        a_ = Vector((sx * 0.135, ty + 0.004, tz + 0.006))
        S.append(curve_obj("Bar_CrossBraceLeg", [a_, a_ + Vector((0, 0.006, 0.049))], 0.009, col, M['steel_rusty']))
    S.append(cyl_obj("Bar_CrossBracePad", Vector((-0.09, ty + 0.012, tz + 0.055)), Vector((0.09, ty + 0.012, tz + 0.055)), 0.017, col,
                     M['rubber'], segs=24, bevel=0.004))

    def bar_frame(sx, s):
        p_end = Vector(bar_pts[-1] if sx > 0 else bar_pts[0])
        p_in = Vector(bar_pts[-2] if sx > 0 else bar_pts[1])
        dv = (p_end - p_in).normalized()
        p = p_end - dv * s
        z = (Vector((0, 0, 1)) - dv * dv.z).normalized()
        y = (z.cross(dv) * sx).normalized()
        x = y.cross(z)
        return p, dv, mat4(p, Matrix((x, y, z)).transposed())

    def L(Mx, v):
        return Mx @ Vector(v)
    # --- grips & bar-end weights ---
    grip_prof = [(0.0, 0.0125), (0.0, 0.0215), (0.004, 0.0222), (0.008, 0.0185)]
    for i in range(16):
        z = 0.012 + i * 0.0068
        grip_prof += [(z, 0.0172), (z + 0.0022, 0.0181), (z + 0.0046, 0.0181), (z + 0.0068, 0.0172)]
    grip_prof += [(0.124, 0.0165), (0.126, 0.0125)]
    for sx, nm in ((-1, "R"), (1, "L")):
        g0, dv, _ = bar_frame(sx, 0.128)
        S.append(revolve_obj("Bar_Grip_" + nm, grip_prof, col, M['rubber'], mat4(g0, frame_from_z(dv)), segs=32, axis='Z', angle=50))
        endp, _, _ = bar_frame(sx, 0.0)
        S.append(revolve_obj("Bar_EndWeight_" + nm, [(0.0, 0.0), (0.0, 0.0175), (0.006, 0.019), (0.02, 0.0185), (0.024, 0.014), (0.025, 0.0)],
                             col, M['chrome'], mat4(endp, frame_from_z(dv)), segs=32, axis='Z'))
        S.append(bolt(endp + dv * 0.025, dv, col, 'cap', 0.8, "Bar_EndScrew"))

    # --- switch housings (clamshell halves, buttons, wire boots); right one is also the throttle housing ---
    for sx, nm in ((1, "L"), (-1, "R")):
        p, dv, Mh = bar_frame(sx, 0.153)
        for half, zc in (("Upper", 0.0128), ("Lower", -0.0128)):
            bm = bmesh.new()
            bm_box(bm, (0.044, 0.054, 0.0245), Matrix.Translation((0, 0.002, zc)))
            ob = obj_from_bm("Bar_SwitchHousing_%s_%s" % (nm, half), bm, col, M['plastic_black'], Mh)
            mod_bevel(ob, 0.0075, 3, 30)
            S.append(ob)
        for dx in (-0.012, 0.012):
            S.append(bolt(L(Mh, (dx, -0.0255, -0.013)), (Mh.to_3x3() @ Vector((0, -1, 0))), col, 'cap', 0.55, "Bar_SwitchScrew"))
        R3 = Mh.to_3x3()
        if sx > 0:
            S.append(cyl_obj("Bar_Switch_Horn", L(Mh, (0.0, 0.028, -0.009)), L(Mh, (0.0, 0.0335, -0.009)), 0.0075, col, M['rubber'], segs=20, bevel=0.0015))
            S.append(box_obj("Bar_Switch_IndicatorSlot", L(Mh, (0.0, 0.0288, 0.0065)), (0.028, 0.002, 0.009), col, M['dark_metal'], bevel=0.0008))
            S.append(box_obj("Bar_Switch_IndicatorKnob", L(Mh, (0.004, 0.0315, 0.0065)), (0.009, 0.006, 0.007), col, M['plastic_black'], bevel=0.0015))
            S.append(box_obj("Bar_Switch_DipToggle", L(Mh, (0.008, 0.012, 0.0265)), (0.006, 0.014, 0.007), col, M['plastic_black'], bevel=0.0015))
            S.append(box_obj("Bar_Switch_ChokeLever", L(Mh, (-0.01, -0.014, 0.0262)), (0.005, 0.022, 0.005), col, M['alu_cast'], bevel=0.0012))
            S.append(cyl_obj("Bar_Switch_Pass", L(Mh, (0.0, -0.025, 0.006)), L(Mh, (0.0, -0.0295, 0.006)), 0.0055, col, M['plastic_black'], segs=16, bevel=0.001))
        else:
            S.append(box_obj("Bar_Switch_Kill", L(Mh, (0.0, 0.004, 0.0262)), (0.017, 0.013, 0.006), col, M['lens_red'], bevel=0.0018))
            S.append(box_obj("Bar_Switch_Lights", L(Mh, (0.0, 0.0298, 0.0035)), (0.015, 0.004, 0.011), col, M['plastic_black'], bevel=0.0012))
            # throttle cable outlets with adjusters (pull + return)
            for k, dx in enumerate((-0.007, 0.007)):
                a0 = L(Mh, (dx, -0.018, -0.02))
                a1 = L(Mh, (dx, -0.026, -0.034))
                S.append(cyl_obj("Bar_ThrottleAdjuster", a0, a1, 0.0042, col, M['steel_bright'], segs=12))
                S.append(cyl_obj("Bar_ThrottleAdjusterNut", a0.lerp(a1, 0.3), a0.lerp(a1, 0.5), 0.0058, col, M['zinc'], segs=6))
                anchor('throttle_out_%d' % k, a1 + (a1 - a0).normalized() * 0.002, 'steer')
            S.append(cyl_obj("Bar_ThrottleTube", p + dv * 0.022, p + dv * 0.024, 0.0215, col, M['plastic_black'], segs=32))
        bt0 = L(Mh, (-sx * 0.014, 0.004, -0.024))
        bt1 = L(Mh, (-sx * 0.016, 0.006, -0.036))
        S.append(cyl_obj("Bar_SwitchBoot_" + nm, bt0, bt1, 0.0062, col, M['rubber'], r2=0.0045, segs=16))
        anchor('switch_exit_' + nm, bt1 + (bt1 - bt0).normalized() * 0.002, 'steer')

    # --- lever perches, forged lever blades, brake master cylinder, clutch adjuster, mirrors ---
    def blade_outline(pts, widths, end_r):
        out_l, out_r = [], []
        for i, p in enumerate(pts):
            a = pts[max(0, i - 1)]
            b = pts[min(len(pts) - 1, i + 1)]
            t = Vector((b[0] - a[0], b[1] - a[1])).normalized()
            n = Vector((-t.y, t.x))
            w = widths[i] / 2
            out_l.append((p[0] + n.x * w, p[1] + n.y * w))
            out_r.append((p[0] - n.x * w, p[1] - n.y * w))
        # round caps
        c0, c1 = Vector(pts[0]), Vector(pts[-1])
        t1 = (c1 - Vector(pts[-2])).normalized()
        cap1 = []
        for k in range(1, 8):
            ang = math.atan2(t1.y, t1.x) + pi / 2 - pi * k / 8
            cap1.append((c1.x + cos(ang) * widths[-1] / 2, c1.y + sin(ang) * widths[-1] / 2))
        t0 = (Vector(pts[1]) - c0).normalized()
        cap0 = []
        for k in range(1, 10):
            ang = math.atan2(t0.y, t0.x) - pi / 2 - pi * k / 10
            cap0.append((c0.x + cos(ang) * end_r, c0.y + sin(ang) * end_r))
        return out_l + cap1 + list(reversed(out_r)) + cap0

    for sx, nm in ((-1, "R"), (1, "L")):
        p, dv, Mp = bar_frame(sx, 0.194)
        R3 = Mp.to_3x3()
        X_out = R3 @ Vector((sx, 0, 0))
        Zl = R3 @ Vector((0, 0, 1))
        Yl = R3 @ Vector((0, 1, 0))
        S.append(cyl_obj("Bar_LeverPerch_" + nm, p - dv * 0.012, p + dv * 0.012, 0.0195, col, M['alu_cast'], segs=24, bevel=0.002))
        for zz in (0.0085, -0.0085):
            S.append(box_obj("Bar_LeverPerchEar_" + nm, L(Mp, (0, -0.024, zz)), (0.022, 0.03, 0.005), col, M['alu_cast'], bevel=0.0015))
        S.append(bolt(L(Mp, (0, 0.019, 0.0)), Yl, col, 'cap', 0.7, "Bar_PerchClampBolt"))
        piv = (0.0, -0.035)
        cl = [piv, (sx * 0.018, -0.047), (sx * 0.06, -0.057), (sx * 0.12, -0.059), (sx * 0.166, -0.053)]
        outline = blade_outline(cl, [0.019, 0.017, 0.015, 0.0135, 0.012], 0.0098)
        if sx < 0:
            outline = list(reversed(outline))
        bm = bmesh.new()
        extrude_bm(bm, outline, 0.0072)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        lev = obj_from_bm("Bar_Lever_" + nm, bm, col, M['alu_cast'], Mp @ Matrix.Rotation(rad(-6), 4, 'X'), angle=40)
        mod_bevel(lev, 0.0018, 2, 30)
        S.append(lev)
        bmb = bmesh.new()
        bmesh.ops.create_uvsphere(bmb, u_segments=16, v_segments=10, radius=0.0068)
        S.append(obj_from_bm("Bar_LeverBall_" + nm, bmb, col, M['alu_cast'], Mp @ Matrix.Rotation(rad(-6), 4, 'X') @ Matrix.Translation((cl[-1][0], cl[-1][1], 0))))
        pv = L(Mp, (0, -0.035, 0))
        S.append(cyl_obj("Bar_LeverPivotBolt_" + nm, pv - Zl * 0.012, pv + Zl * 0.013, 0.0042, col, M['zinc'], segs=6))
        S.append(bolt(pv - Zl * 0.012, -Zl, col, 'nut', 0.45, "Bar_LeverPivotNut"))
        # mirror: threaded boss on the perch, stalk with a bend, ball joint, round head facing the rider
        mb = L(Mp, (0, -0.006, 0.019))
        S.append(cyl_obj("Bar_MirrorBoss_" + nm, mb, mb + Zl * 0.012, 0.0065, col, M['alu_cast'], segs=16))
        S.append(bolt(mb + Zl * 0.012, Zl, col, 'nut', 0.55, "Bar_MirrorLockNut"))
        m0 = mb + Zl * 0.016
        m1 = m0 + Zl * 0.11 + X_out * 0.03
        m2 = m1 + X_out * 0.08 + Zl * 0.03 + Yl * 0.01
        S.append(curve_obj("Bar_MirrorStalk_" + nm, [m0, m1, m2], 0.0048, col, M['dark_metal'], mode='bend', bend=0.03))
        bmb = bmesh.new()
        bmesh.ops.create_uvsphere(bmb, u_segments=12, v_segments=8, radius=0.0075)
        S.append(obj_from_bm("Bar_MirrorBall_" + nm, bmb, col, M['dark_metal'], mat4(m2)))
        mh = m2 + Yl * 0.016 + Zl * 0.012
        Rm = frame_from_z(Vector((0, 1, 0.12)))
        S.append(revolve_obj("Bar_MirrorHousing_" + nm, [(-0.02, 0.0), (-0.02, 0.036), (-0.014, 0.046), (-0.004, 0.0495), (0.004, 0.0495),
                                                         (0.0045, 0.046), (0.0, 0.0455), (0.0, 0.0)], col, M['plastic_black'],
                             mat4(mh, Rm), segs=40, axis='Z'))
        S.append(cyl_obj("Bar_MirrorGlass_" + nm, mh + Rm @ Vector((0, 0, 0.0005)), mh + Rm @ Vector((0, 0, 0.0025)), 0.0455, col, M['reflector'], segs=40))
        if sx < 0:
            # brake master cylinder body + reservoir with lid, screws, sight glass, banjo and brake-light switch
            mc0 = L(Mp, (0.008, -0.014, 0.0))
            mc1 = L(Mp, (0.072, -0.014, 0.0))
            S.append(cyl_obj("Brake_Front_MasterCyl", mc0, mc1, 0.0125, col, M['alu_cast'], segs=24, bevel=0.0015))
            S.append(box_obj("Brake_Front_MasterClamp", L(Mp, (0.04, 0.016, 0.0)), (0.024, 0.012, 0.03), col, M['alu_cast'], bevel=0.002))
            for dz in (-0.009, 0.009):
                S.append(bolt(L(Mp, (0.04, 0.022, dz)), Yl, col, 'cap', 0.6, "Brake_Front_MasterClampBolt"))
            rc = L(Mp, (0.04, -0.012, 0.031))
            S.append(box_obj("Brake_Front_Reservoir", rc, (0.052, 0.036, 0.026), col, M['alu_cast'], bevel=0.004, segs=2))
            S.append(box_obj("Brake_Front_ReservoirLid", rc + Zl * 0.0145, (0.054, 0.038, 0.004), col, M['dark_metal'], bevel=0.0012))
            for dx in (-0.02, 0.02):
                S.append(bolt(rc + Zl * 0.0165 + (R3 @ Vector((dx, 0, 0))), Zl, col, 'cap', 0.45, "Brake_Front_LidScrew"))
            sg = rc + Yl * 0.0182
            S.append(cyl_obj("Brake_Front_SightGlass", sg, sg + Yl * 0.0015, 0.0065, col, M['glass'], segs=16))
            S.append(cyl_obj("Brake_Front_Fluid", sg - Yl * 0.0004, sg, 0.006, col, mat_cache('fluid', lambda: mat_simple("M_BrakeFluid", (0.35, 0.22, 0.03), rough=0.1)), segs=16))
            bj = L(Mp, (0.068, -0.014, -0.013))
            S.append(cyl_obj("Brake_Front_MasterBanjo", bj, bj - Zl * 0.012, 0.0085, col, M['steel_bright'], segs=20))
            S.append(bolt(bj - Zl * 0.012, -Zl, col, 'bolt', 0.75, "Brake_Front_MasterBanjoBolt"))
            anchor('brake_banjo_F', bj - Zl * 0.006 - Yl * 0.012, 'steer')
            S.append(box_obj("Elec_FrontBrakeSwitch", L(Mp, (0.012, -0.024, -0.016)), (0.012, 0.01, 0.008), col, M['plastic_black'], bevel=0.0015))
            anchor('brake_switch_F', L(Mp, (0.018, -0.022, -0.022)), 'steer')
            S.append(cyl_obj("Bar_LeverAdjuster", L(Mp, (0.014, -0.041, -0.004)), L(Mp, (0.014, -0.041, 0.004)), 0.0082, col, M['alu_cast'], segs=16, bevel=0.0008))
        else:
            # clutch cable adjuster barrel + lock ring
            c0 = L(Mp, (-0.004, -0.03, -0.002))
            cdir = (R3 @ Vector((-0.3, -0.92, -0.25))).normalized()
            S.append(cyl_obj("Bar_ClutchAdjuster", c0, c0 + cdir * 0.022, 0.0052, col, M['alu_cast'], segs=16, bevel=0.0006))
            S.append(cyl_obj("Bar_ClutchAdjusterRing", c0 + cdir * 0.004, c0 + cdir * 0.009, 0.0072, col, M['alu_cast'], segs=12, bevel=0.0008))
            anchor('clutch_out', c0 + cdir * 0.024, 'steer')

    # --- instrument pod: steel shell, twin gauges with printed dials, needles, warning lamps ---
    Pc = T0 + Vector((0, -0.052, 0.08))
    tilt = rad(40)
    n = Vector((0, cos(tilt), sin(tilt)))
    upf = Vector((0, -sin(tilt), cos(tilt)))
    xf = upf.cross(n)
    MP = mat4(Pc, Matrix((xf, upf, n)).transposed())
    R3P = MP.to_3x3()
    shell = extrude_obj("Gauge_PodShell", rrect(0.18, 0.09, 0.02, 5), 0.062, col, M['alu_cast'], MP, bevel=0.004, z0=-0.062)
    S.append(shell)
    S.append(extrude_obj("Gauge_PodFace", rrect(0.172, 0.082, 0.017, 5), 0.002, col, M['dark_metal'], MP, bevel=0.0005, z0=0.0))
    S.append(box_obj("Gauge_PodVisor", MP @ Vector((0, 0.041, 0.012)), (0.18, 0.006, 0.026), col, M['alu_cast'], bevel=0.002))
    S[-1].matrix_world = MP @ Matrix.Translation((0, 0.042, 0.012))
    for x_, y_ in ((-0.08, -0.035), (0.08, -0.035), (-0.08, 0.034), (0.08, 0.034)):
        S.append(bolt(MP @ Vector((x_, y_, 0.002)), R3P @ Vector((0, 0, 1)), col, 'cap', 0.55, "Gauge_PodScrew"))
    gauges = {}
    for gname, cx, cy, rf, img_n, emit_n in (("Speedo", -0.042, 0.0, 0.034, "dial_speedo.png", "dial_speedo_emit.png"),
                                             ("Tacho", 0.046, 0.003, 0.027, "dial_tacho.png", None)):
        G = MP @ Matrix.Translation((cx, cy, 0.0))
        rb = rf + 0.0055
        S.append(revolve_obj("Gauge_%s_Cup" % gname, [(0.0, rb - 0.001), (-0.03, rb - 0.003), (-0.045, rb * 0.6), (-0.046, 0.0)],
                             col, M['dark_metal'], G, segs=48, axis='Z'))
        S.append(revolve_obj("Gauge_%s_Bezel" % gname, [(0.0, rf), (0.0, rb), (0.004, rb), (0.0078, rf + 0.0016), (0.0078, rf + 0.0006), (0.003, rf)],
                             col, M['chrome'], G, segs=64, axis='Z', closed=True))
        gm = mat_gauge("M_Gauge_" + gname, load_image(img_n), load_image(emit_n) if emit_n else None)
        S.append(obj_from_bm("Gauge_%s_Dial" % gname, disc_uv_bm(rf, 64, 0.0016), col, gm, G, smooth=False))
        S.append(revolve_obj("Gauge_%s_Glass" % gname, [(0.0071, 0.0), (0.0074, rf * 0.5), (0.0071, rf + 0.0012)], col, M['glass'], G, segs=48, axis='Z'))
        pivot = make_empty("Gauge_%s_NeedlePivot" % gname, G @ Matrix.Translation((0, 0, 0.0035)), col, 'ARROWS', 0.01)
        parent_later(pivot, steer)
        needle = [(-0.18 * rf, 0.0018), (0.0, 0.0019), (0.88 * rf, 0.0005), (0.88 * rf, -0.0005), (0.0, -0.0019), (-0.18 * rf, -0.0018)]
        bm = bmesh.new()
        extrude_bm(bm, needle, 0.0009)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        nd = obj_from_bm("Gauge_%s_Needle" % gname, bm, col, mat_cache('needle', lambda: mat_simple("M_Gauge_Needle", (0.9, 0.18, 0.03), rough=0.35,
                                                                                                    emission=((1.0, 0.25, 0.05), 0.4))),
                         G @ Matrix.Translation((0, 0, 0.0035)) @ Matrix.Rotation(rad(225), 4, 'Z'), smooth=False)
        parent_later(nd, pivot)
        S.append(cyl_obj("Gauge_%s_NeedleCap" % gname, G @ Vector((0, 0, 0.003)), G @ Vector((0, 0, 0.0062)), 0.0045, col, M['plastic_black'], segs=20, bevel=0.0008))
        gauges[gname] = nd
    for k, (lm, yy) in enumerate((('lens_green', 0.027), ('lens_blue', 0.009), ('lens_orange', -0.009), ('lens_red', -0.027))):
        G = MP @ Matrix.Translation((0.0035, yy, 0.0))
        S.append(revolve_obj("Gauge_LampBezel", [(0.0, 0.0042), (0.0, 0.0058), (0.0022, 0.0058), (0.0028, 0.0045)], col, M['chrome'], G, segs=24, axis='Z', closed=True))
        S.append(revolve_obj("Gauge_Lamp", [(0.0, 0.0), (0.0, 0.0043), (0.0026, 0.004), (0.0032, 0.0)], col, M[lm], G, segs=24, axis='Z'))
    # pod bracket to the bar clamps, rubber-mounted
    S.append(beam_obj("Gauge_PodBracket", MP @ Vector((0, -0.038, -0.03)), T0 + Vector((0, -0.016, 0.016)), 0.07, 0.005, 0.07, 0.005,
                      col, M['steel_rusty'], up=(0, 1, 0), bevel=0.0015, segs=1))
    for sx in (-1, 1):
        gp = MP @ Vector((sx * 0.025, -0.046, -0.03))
        S.append(cyl_obj("Gauge_PodRubberMount", gp, gp - (R3P @ Vector((0, 1, 0))) * 0.01, 0.0065, col, M['rubber'], segs=16, bevel=0.001))
    anchor('gauge_conn', MP @ Vector((0.0, -0.025, -0.064)), 'steer')
    # --- ignition switch on the top yoke (left) with key, ring and tag ---
    ig = F.p(yt + 0.013, off - 0.035, 0.085)
    d = F.d
    S.append(revolve_obj("Ignition_Bezel", [(0.0, 0.0), (0.0, 0.019), (0.004, 0.019), (0.008, 0.016), (0.009, 0.011), (0.009, 0.0)], col, M['chrome'],
                         mat4(ig, Rf_ := F.R), segs=32, axis='Z'))
    S.append(cyl_obj("Ignition_Barrel", ig + d * 0.009, ig + d * 0.0105, 0.010, col, M['dark_metal'], segs=24))
    S.append(box_obj("Ignition_KeyBlade", ig + d * 0.016, (0.0022, 0.007, 0.014), col, M['steel_bright'], rot=(-P['rake'], 0, 0), bevel=0.0004))
    kh = ig + d * 0.03
    S.append(box_obj("Ignition_KeyHead", kh, (0.006, 0.026, 0.02), col, M['plastic_black'], rot=(-P['rake'], 0, 0), bevel=0.0026, segs=3))
    ring_c = kh + d * 0.014 + F.nb * 0.012
    S.append(revolve_obj("Ignition_KeyRing", [(-0.0007, 0.0085), (0.0007, 0.0085), (0.0007, 0.0099), (-0.0007, 0.0099)], col, M['steel_bright'],
                         mat4(ring_c, Matrix.Rotation(pi / 2, 3, 'Y')), segs=24, axis='Z', closed=True))
    S.append(box_obj("Ignition_KeyTag", ring_c + Vector((0, 0.014, -0.012)), (0.003, 0.016, 0.026), col, M['paint_red'], rot=(-20, 0, 0), bevel=0.0012))
    anchor('ignition_back', ig - d * 0.03, 'steer')
    for ob in S:
        if ob.parent is None:
            parent_later(ob, steer)
    return gauges


# ---------------------------------------------------------------------------------------------
# Utility: ray-cast onto an evaluated object (used to seat details on subdivided bodywork)
# ---------------------------------------------------------------------------------------------

def surface_hit(ob, origin, direction):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    inv = ev.matrix_world.inverted()
    o = inv @ Vector(origin)
    d = (inv.to_3x3() @ Vector(direction)).normalized()
    ok, loc, nrm, _ = ev.ray_cast(o, d)
    if not ok:
        return None, None
    n = (ev.matrix_world.to_3x3().inverted().transposed() @ nrm).normalized()
    return ev.matrix_world @ loc, n


def beam_bm(bm, p0, p1, w0, h0, w1, h1, up=(0, 0, 1)):
    """Tapered box from p0 to p1 (w lateral, h along `up`)."""
    p0, p1 = Vector(p0), Vector(p1)
    ax = (p1 - p0).normalized()
    upv = Vector(up)
    side = ax.cross(upv).normalized()
    upv = side.cross(ax).normalized()
    vs = []
    for p, w, h in ((p0, w0, h0), (p1, w1, h1)):
        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vs.append(bm.verts.new(p + side * sx * w / 2 + upv * sz * h / 2))
    f = [(0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    for q in f:
        bm.faces.new([vs[i] for i in q])


def beam_obj(name, p0, p1, w0, h0, w1, h1, col, mat, up=(0, 0, 1), bevel=0.006, segs=3):
    bm = bmesh.new()
    beam_bm(bm, p0, p1, w0, h0, w1, h1, up)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = obj_from_bm(name, bm, col, mat, angle=40)
    if bevel:
        mod_bevel(ob, bevel, segs, 40)
    return ob


# ---------------------------------------------------------------------------------------------
# FRAME  (double cradle, every tube placed outside the front tyre's swept envelope)
# ---------------------------------------------------------------------------------------------

FRAME = {}


def build_frame(M, F, root):
    col = coll('Frame', COLL['Bike'])
    Yp = M['paint_yellow']
    off = P['fork_offset']
    yb, yt = P['yokes']
    obs = []
    obs.append(cyl_obj("Frame_HeadTube", F.p(yb + 0.025, off), F.p(yt - 0.025, off), 0.032, col, Yp, segs=40, bevel=0.002))
    for t in (yb + 0.025, yt - 0.025):
        obs.append(cyl_obj("Frame_HeadBearingCup", F.p(t - 0.006, off), F.p(t + 0.006, off), 0.036, col, M['steel_rusty'], segs=40))
    ht_top = F.p(yt - 0.09, off + 0.04)
    ht_low = F.p(yb + 0.07, off + 0.055)
    backbone = [ht_top, Vector((0, -0.33, 0.965)), Vector((0, -0.08, 0.895)), Vector((0, 0.13, 0.872))]
    FRAME['backbone'] = backbone
    obs.append(curve_obj("Frame_Backbone", backbone, 0.024, col, Yp, mode='bend', bend=0.08))
    rail = [Vector((0.075, 0.10, 0.858)), Vector((0.075, 0.74, 0.874)), Vector((0.0, 0.81, 0.874))]
    FRAME['rail_L'] = rail
    rails = curve_obj("Frame_SeatRails", rail, 0.0175, col, Yp, mode='bend', bend=0.06)
    mod_mirror(rails)
    obs.append(rails)
    obs.append(curve_obj("Frame_RailCross_Front", [Vector((-0.075, 0.12, 0.862)), Vector((0.075, 0.12, 0.862))], 0.016, col, Yp))
    obs.append(curve_obj("Frame_RailCross_Shock", [Vector((-0.075, 0.345, 0.866)), Vector((0.075, 0.345, 0.866))], 0.016, col, Yp))
    dt_low = F.p(0.50, off + 0.20)
    obs.append(curve_obj("Frame_DownTube", [ht_low, dt_low], 0.027, col, Yp))
    FRAME['downtube'] = [ht_low, dt_low]
    cradle = [Vector((0.0, dt_low.y, dt_low.z + 0.012)), Vector((0.07, -0.39, 0.64)), Vector((0.07, -0.37, 0.46)), Vector((0.07, -0.36, 0.32)),
              Vector((0.07, -0.28, 0.20)), Vector((0.07, 0.12, 0.20)), Vector((0.085, 0.215, 0.29)), Vector((0.09, 0.245, 0.50)),
              Vector((0.088, 0.318, 0.70)), Vector((0.075, 0.342, 0.862))]
    FRAME['cradle_L'] = cradle
    cr = curve_obj("Frame_Cradle", cradle, 0.0195, col, Yp, mode='bend', bend=0.06)
    mod_mirror(cr)
    obs.append(cr)
    sub = curve_obj("Frame_SubframeStrut", [Vector((0.09, 0.27, 0.58)), Vector((0.08, 0.58, 0.868))], 0.0145, col, Yp)
    mod_mirror(sub)
    obs.append(sub)
    obs.append(curve_obj("Frame_CradleCross", [Vector((-0.07, -0.10, 0.215)), Vector((0.07, -0.10, 0.215))], 0.015, col, Yp))
    obs.append(extrude_obj("Frame_HeadGusset", [(0.0, 0.0), (0.085, -0.03), (0.035, -0.075), (-0.004, -0.08)], 0.006, col, Yp,
                           mat4(F.p(yt - 0.075, off + 0.06), Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))), bevel=0.0015))
    # swingarm pivot plates - rear edge stops short of the swingarm cross brace
    plate = [(0.13, 0.33), (0.272, 0.33), (0.276, 0.52), (0.25, 0.60), (0.20, 0.62), (0.13, 0.55)]
    pp = extrude_obj("Frame_PivotPlates", plate, 0.018, col, M['paint_yellow_heavy'],
                     mat4(Vector((0.126, 0, 0)), Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))), bevel=0.004)
    mod_mirror(pp, mirror_obj=root)
    obs.append(pp)
    for (y, z) in ((0.15, 0.36), (0.25, 0.36), (0.255, 0.50), (0.17, 0.58), (0.22, 0.585)):
        for sx in (-1, 1):
            obs.append(bolt(Vector((sx * 0.135, y, z)), (sx, 0, 0), col, 'bolt', 1.0, "Frame_PlateBolt"))
    sp = P['swing_pivot']
    ph = P['arm_x'] + 0.032
    obs.append(cyl_obj("Frame_SwingarmPivotBolt", sp + Vector((-ph, 0, 0)), sp + Vector((ph, 0, 0)), 0.009, col, M['steel_bright'], segs=16))
    for sx in (-1, 1):
        obs.append(bolt(sp + Vector((sx * ph, 0, 0)), (sx, 0, 0), col, 'nut', 1.1, "Frame_PivotNut"))
    # engine mounts: front lower plates (cradle -> crankcase), head steady (down tube -> front head), rear via pivot plates
    for sx in (-1, 1):
        obs.append(box_obj("Frame_EngineMount_Front", Vector((sx * 0.082, -0.315, 0.335)), (0.008, 0.12, 0.05), col, M['steel_rusty'], bevel=0.002))
        for y in (-0.36, -0.27):
            obs.append(bolt(Vector((sx * 0.087, y, 0.335)), (sx, 0, 0), col, 'bolt', 1.0, "Frame_EngineMountBolt"))
    obs.append(beam_obj("Frame_HeadSteady", Vector((0.035, dt_low.y + 0.02, dt_low.z - 0.01)), Vector((0.035, -0.27, 0.665)), 0.006, 0.03, 0.006, 0.026,
                        col, M['steel_rusty'], up=(0, 0, 1), bevel=0.0015, segs=1))
    obs.append(bolt(Vector((0.04, -0.275, 0.665)), (1, 0, 0), col, 'bolt', 0.9, "Frame_HeadSteadyBolt"))
    # shock top mount tabs, steering stops, tank mounts
    for sx in (-1, 1):
        obs.append(box_obj("Frame_ShockTab", Vector((sx * 0.022, 0.345, 0.842)), (0.006, 0.04, 0.05), col, Yp, bevel=0.002))
        obs.append(box_obj("Frame_SteeringStopLug", F.p(yb + 0.03, off + 0.055, sx * 0.05), (0.012, 0.014, 0.012), col, Yp, bevel=0.002))
        obs.append(box_obj("Frame_TankMountTab", Vector((sx * 0.03, -0.36, 0.985)), (0.006, 0.03, 0.022), col, Yp, bevel=0.0015))
        obs.append(cyl_obj("Frame_TankMountRubber", Vector((sx * 0.033, -0.36, 0.985)), Vector((sx * 0.046, -0.36, 0.985)), 0.009, col, M['rubber'], segs=16, bevel=0.001))
    obs.append(box_obj("Frame_TankMountRear", Vector((0, 0.10, 0.898)), (0.03, 0.03, 0.006), col, Yp, bevel=0.0015))
    # horn on a bracket off the down tube (left, below the tank nose)
    hc = Vector((0.07, -0.47, 0.80))
    obs.append(revolve_obj("Elec_Horn", [(0.0, 0.0), (0.0, 0.045), (0.008, 0.048), (0.022, 0.046), (0.03, 0.02), (0.032, 0.0)],
                           col, M['steel_rusty'], mat4(hc, frame_from_z(Vector((0.5, -1.0, 0.1)))), segs=40, axis='Z'))
    obs.append(cyl_obj("Elec_HornGrille", hc + Vector((0.0005, -0.001, 0.0)), hc + Vector((0.0025, -0.005, 0.0004)), 0.034, col, M['dark_metal'], segs=40))
    obs.append(beam_obj("Elec_HornBracket", hc + Vector((-0.015, 0.03, 0.0)), Vector((0.022, -0.448, 0.80)), 0.005, 0.02, 0.005, 0.02, col,
                        M['steel_rusty'], bevel=0.001, segs=1))
    anchor('horn', hc + Vector((-0.018, 0.028, -0.01)))
    # electrics tray under the seat (ahead of the rear tyre's bump envelope)
    obs.append(box_obj("Elec_Tray", Vector((0, 0.505, 0.797)), (0.15, 0.11, 0.003), col, M['paint_black'], bevel=0.001))
    for sx in (-1, 1):
        obs.append(box_obj("Elec_TrayTab", Vector((sx * 0.072, 0.552, 0.83)), (0.004, 0.02, 0.065), col, M['paint_black'], bevel=0.001))
    return obs


# ---------------------------------------------------------------------------------------------
# BODYWORK (subdivision cages)
# ---------------------------------------------------------------------------------------------

def build_body(M, root):
    col = coll('Bodywork', COLL['Bike'])
    obs = []
    # ---- fuel tank: deep tunnel clears the backbone, nose clears the yokes at full lock ----
    U = [-1, -0.5, 0, 0.5, 1]
    YS = [-0.37, -0.25, -0.125, -0.005, 0.092]
    ZT = [1.08, 1.11, 1.10, 1.06, 0.99]
    ZB = [0.88, 0.83, 0.81, 0.815, 0.865]
    TD = [0.15, 0.17, 0.175, 0.14, 0.11]          # tunnel depth (frame backbone runs inside)
    HW = [0.115, 0.198, 0.205, 0.182, 0.125]
    WF = [0.62, 0.97, 1.0, 0.9, 0.5]

    def tank(u, v, w):
        y = interp(U, YS, u)
        zt, zb = interp(U, ZT, u), interp(U, ZB, u)
        z = lerp(zb, zt, (w + 1) / 2)
        roof = zb + interp(U, TD, u) * (1 - (abs(v) / 0.75) ** 4) if abs(v) < 0.75 else zb
        if w <= -0.99:
            z = max(z, roof)
        elif abs(u) >= 0.99 and abs(v) < 0.75 and z < roof:
            z = roof + 0.012 * (w + 1)            # open tunnel mouth at the nose and tail
        if u <= -0.99:
            y -= 0.02 * (w + 1) / 2
        hw = interp(U, HW, u) * interp(U, WF, w)
        if 0.2 < u < 0.8 and -0.8 < w < 0.2:
            hw *= 0.9
        return Vector((v * hw, y, z))
    tk = cage_obj("Body_FuelTank", col, M["paint_yellow"], tank, cuts=3, view=3, render=3)
    mod_weighted_normals(tk)
    obs.append(tk)
    bpy.context.view_layer.update()
    # painted emblem: projected decal in the tank paint (no separate geometry)
    hit, nn = surface_hit(tk, (0.5, -0.155, 0.965), (-1, 0, 0))
    mask = load_image("emblem_crown_mask.png")
    if hit is not None and mask is not None:
        tank_mat = mat_paint("M_Paint_Yellow_Tank", P['yellow'], chip=0.62, rust=0.65, dirt=0.8, scale=1.2, scratches=0.8, streak=0.7, dust=0.45,
                             decal=dict(image=mask, center=(hit.y, hit.z), size=0.124))
        tk.data.materials[0] = tank_mat
    top, n = surface_hit(tk, (0, -0.2, 1.3), (0, 0, -1))
    if top is not None:
        cap = revolve_obj("Body_FuelCap", [(0, 0.0), (0, 0.034), (0.004, 0.036), (0.012, 0.035), (0.016, 0.03), (0.018, 0.012),
                                           (0.024, 0.01), (0.026, 0.0)], col, M['chrome'], mat4(top - n * 0.002, frame_from_z(n)), segs=40, axis='Z')
        obs.append(cap)
        obs.append(box_obj("Body_FuelCap_Hinge", top + n * 0.015 + Vector((0, 0.04, 0)), (0.016, 0.018, 0.01), col, M['chrome'], bevel=0.002))
        bolt_ring(top + n * 0.001, n, 0.044, 6, col, 'cap', 0.6, name="Body_FuelCapScrew")
    # fuel petcock under the left rear of the tank
    pk, pn = surface_hit(tk, (0.085, -0.03, 0.5), (0, 0, 1))
    if pk is not None:
        obs.append(cyl_obj("Fuel_PetcockNut", pk + Vector((0, 0, 0.002)), pk - Vector((0, 0, 0.01)), 0.012, col, M['steel_rusty'], segs=6))
        obs.append(box_obj("Fuel_PetcockBody", pk - Vector((0, 0, 0.022)), (0.024, 0.03, 0.022), col, M['alu_cast'], bevel=0.004))
        obs.append(box_obj("Fuel_PetcockLever", pk - Vector((-0.016, 0.0, 0.022)), (0.008, 0.026, 0.008), col, M['plastic_black'], rot=(0, 0, 20), bevel=0.0015))
        obs.append(cyl_obj("Fuel_PetcockSpigot", pk - Vector((0, -0.008, 0.034)), pk - Vector((0, -0.008, 0.05)), 0.0045, col, M['brass'], segs=12))
        anchor('petcock_out', pk - Vector((0, -0.008, 0.05)))
    # ---- seat ----
    SU = [-1, -0.5, 0, 0.5, 1]
    SY = [0.055, 0.15, 0.28, 0.40, 0.52]
    STOP = [0.945, 0.935, 0.932, 0.94, 0.955]
    SBOT = [0.885, 0.872, 0.868, 0.872, 0.88]
    SHW = [0.085, 0.13, 0.142, 0.14, 0.125]

    def seat(u, v, w):
        y = interp(SU, SY, u)
        z = lerp(interp(SU, SBOT, u), interp(SU, STOP, u), (w + 1) / 2)
        hw = interp(SU, SHW, u) * interp([-1, -0.5, 0, 0.5, 1], [0.92, 1.0, 1.0, 0.96, 0.8], w)
        return Vector((v * hw, y, z))
    st = cage_obj("Body_Seat", col, M["leather"], seat, cuts=3, view=3, render=3)
    obs.append(st)

    def pan(u, v, w):
        p = seat(u, v, w)
        p.z = lerp(0.858, 0.874, (w + 1) / 2)
        p.x *= 1.02
        return p
    obs.append(cage_obj("Body_SeatPan", col, M['plastic_black'], pan, cuts=2, view=2, render=2))
    loop = []
    for i in range(48):
        a = 2 * pi * i / 48
        u = math.copysign(abs(cos(a)) ** 0.35, cos(a))
        v = math.copysign(abs(sin(a)) ** 0.35, sin(a))
        loop.append(seat(u * 0.995, v, 0.45))
    pip = curve_obj("Body_SeatPiping", loop, 0.0028, col, M['leather'], closed=True, res=4)
    sw = pip.modifiers.new("Shrinkwrap", 'SHRINKWRAP')
    sw.target = st
    sw.wrap_method = 'NEAREST_SURFACEPOINT'
    sw.offset = 0.0012
    sw.wrap_mode = 'OUTSIDE_SURFACE'
    sw.use_apply_on_spline = True
    obs.append(pip)
    # ---- tail cowl: underside clears the rear tyre at full bump ----
    TU = [-1, -0.5, 0, 0.5, 1]
    TY = [0.40, 0.53, 0.66, 0.79, 0.90]
    TT = [0.912, 0.936, 0.952, 0.96, 0.964]
    TB = [0.852, 0.855, 0.862, 0.868, 0.888]
    THW = [0.128, 0.13, 0.118, 0.098, 0.07]

    def tail(u, v, w):
        y = interp(TU, TY, u)
        z = lerp(interp(TU, TB, u), interp(TU, TT, u), (w + 1) / 2)
        hw = interp(TU, THW, u) * interp([-1, -0.5, 0, 0.5, 1], [0.9, 1.0, 1.0, 0.92, 0.7], w)
        if u >= 0.99:
            y += 0.02 * (w + 1) / 2
        return Vector((v * hw, y, z))
    tl = cage_obj("Body_TailCowl", col, M['paint_yellow'], tail, cuts=3, view=2, render=3)
    mod_weighted_normals(tl)
    obs.append(tl)
    bpy.context.view_layer.update()
    hit, nn = surface_hit(tl, (0, 1.2, 0.91), (0, -1, 0))
    if hit is not None:
        obs.append(box_obj("Light_Tail_Housing", hit + nn * 0.004, (0.085, 0.012, 0.038), col, M['plastic_black'], bevel=0.005, segs=3))
        obs.append(box_obj("Light_Tail_Lens", hit + nn * 0.011, (0.078, 0.006, 0.031), col, M['lens_red'], bevel=0.004, segs=2))
        anchor('tail_light', hit - nn * 0.03 + Vector((0, 0, -0.01)))
    for sx in (-1, 1):
        hit, nn = surface_hit(tl, (sx * 0.4, 0.66, 0.905), (-sx, 0, 0))
        if hit is not None:
            R = frame_from_z(nn)
            obs.append(revolve_obj("Body_TailReflector_Bezel" + ("_L" if sx > 0 else "_R"),
                                   [(0, 0.0), (0, 0.021), (0.003, 0.022), (0.006, 0.019), (0.006, 0.0)], col, M['chrome'], mat4(hit, R), segs=32, axis='Z'))
            obs.append(revolve_obj("Body_TailReflector" + ("_L" if sx > 0 else "_R"),
                                   [(0.0, 0.0), (0.0068, 0.0), (0.0066, 0.016), (0.004, 0.0185)], col, M['lens_red'], mat4(hit, R), segs=32, axis='Z'))
        ip = Vector((sx * 0.07, 0.86, 0.86))
        obs.append(cyl_obj("Indicator_Rear_Stalk", ip, ip + Vector((sx * 0.05, 0.0, 0.0)), 0.006, col, M['rubber'], segs=12))
        obs.append(box_obj("Indicator_Rear_Housing", ip + Vector((sx * 0.06, 0.006, 0)), (0.026, 0.03, 0.022), col, M['plastic_black'], bevel=0.004))
        obs.append(box_obj("Indicator_Rear_Lens", ip + Vector((sx * 0.06, 0.022, 0)), (0.022, 0.005, 0.018), col, M['lens_orange'], bevel=0.002))
        anchor('ind_rear_' + ('L' if sx > 0 else 'R'), ip + Vector((-sx * 0.004, 0, 0)))
    # ---- left side cover over the electrics (right side is left open for the rear exhaust) ----
    def side_panel(u, v, w):
        y = lerp(0.235, 0.37, (u + 1) / 2) + (w + 1) / 2 * 0.03 * ((u + 1) / 2)
        z = lerp(0.665, 0.845, (w + 1) / 2)
        x = 0.123 + (v + 1) / 2 * 0.018 - 0.01 * (w + 1) / 2
        return Vector((x, y, z))
    sp = cage_obj("Body_SideCover_L", col, M['paint_yellow'], side_panel, cuts=2, view=2, render=3)
    obs.append(sp)
    obs.append(revolve_obj("Body_SideBadge_L", [(0, 0.0), (0, 0.032), (0.004, 0.033), (0.007, 0.028), (0.008, 0.012), (0.009, 0.0)],
                           col, M['chrome'], mat4(Vector((0.142, 0.31, 0.76)), frame_from_z(Vector((1, 0, 0)))), segs=40, axis='Z'))
    for (y, z) in ((0.25, 0.69), (0.355, 0.69), (0.25, 0.825), (0.37, 0.825)):
        obs.append(bolt(Vector((0.141, y, z)), (1, 0, 0), col, 'cap', 0.8, "Body_SideCoverScrew"))
    # ---- front shrouds: flank the front cylinder head, behind the tyre envelope even at full lock ----
    def shroud(u, v, w):
        y = lerp(-0.355, -0.22, (u + 1) / 2) - 0.03 * (w + 1) / 2 * (1 - (u + 1) / 2)
        z = lerp(0.50, 0.70, (w + 1) / 2)
        x = 0.126 + (v + 1) / 2 * 0.04
        return Vector((x, y, z))
    sh = cage_obj("Body_FrontShrouds", col, M['paint_yellow_heavy'], shroud, cuts=1, view=2, render=3)
    for m_ in sh.modifiers:
        m_.levels, m_.render_levels = 1, 2
    mod_bevel(sh, 0.004, 2, 30, name="EdgeBevel")
    sh.modifiers.move(1, 0)
    mod_mirror(sh, mirror_obj=root)
    sh.modifiers.move(2, 0)
    obs.append(sh)
    for sx in (-1, 1):
        for (y, z) in ((-0.34, 0.53), (-0.245, 0.53), (-0.37, 0.67), (-0.255, 0.67)):
            obs.append(bolt(Vector((sx * 0.168, y, z)), (sx, 0, 0), col, 'bolt', 1.0, "Body_ShroudBolt"))
        for k in range(4):
            obs.append(box_obj("Body_ShroudLouvre", Vector((sx * 0.167, -0.30, 0.565 + k * 0.025)), (0.004, 0.07, 0.008), col, M['dark_metal'], bevel=0.0015))
        obs.append(box_obj("Body_ShroudBracket", Vector((sx * 0.1, -0.40, 0.62)), (0.05, 0.012, 0.03), col, M['steel_rusty'], bevel=0.0015))
    # ---- skid plate on spacers below the cradle ----
    path = [(-0.43, 0.30), (-0.41, 0.22), (-0.36, 0.152), (-0.2, 0.14), (0.0, 0.14), (0.13, 0.145)]
    secx = [(-0.115, 0.035), (-0.105, 0.0), (0.105, 0.0), (0.115, 0.035)]
    bm = bmesh.new()
    rows = []
    for (y, z) in path:
        rows.append([bm.verts.new((x, y, z + dz)) for x, dz in secx])
    for i in range(len(rows) - 1):
        for j in range(len(secx) - 1):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    skid = obj_from_bm("Body_SkidPlate", bm, col, M['paint_yellow_heavy'], angle=40)
    mod_solidify(skid, 0.005, -1)
    mod_bevel(skid, 0.0015, 2, 30)
    obs.append(skid)
    for y in (-0.25, 0.0, 0.10):
        for sx in (-1, 1):
            obs.append(cyl_obj("Body_SkidSpacer", Vector((sx * 0.07, y, 0.142)), Vector((sx * 0.07, y, 0.19)), 0.007, col, M['steel_rusty'], segs=12))
            obs.append(bolt(Vector((sx * 0.07, y, 0.141)), (0, 0, -1), col, 'bolt', 1.0, "Body_SkidBolt"))
    return obs, tk, st, tl


# ---------------------------------------------------------------------------------------------
# ENGINE  (air-cooled 50 deg V-twin, kick start, wet sump)
# ---------------------------------------------------------------------------------------------

MAT_CACHE = {}


def mat_cache(key, fn):
    if key not in MAT_CACHE:
        MAT_CACHE[key] = fn()
    return MAT_CACHE[key]


def finned_block(name, base_M, core_r, z0, z1, fin_w, fin_d, fin_t, pitch, col, M, corner=0.03, core_mat=None, fin_mat=None):
    obs = []
    bm = bmesh.new()
    bm_cyl(bm, core_r, core_r * 0.98, z1 - z0, 40, Matrix.Translation((0, 0, z0)))
    obs.append(obj_from_bm(name + "_Core", bm, col, core_mat or M['alu_cast'], base_M))
    n = int((z1 - z0 - fin_t) / pitch) + 1
    bm = bmesh.new()
    extrude_bm(bm, rrect(fin_w, fin_d, corner, 4), fin_t, z0=z0 + 0.002)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    fins = obj_from_bm(name + "_Fins", bm, col, fin_mat or M['alu_fins'], base_M, angle=40)
    a = fins.modifiers.new("FinArray", 'ARRAY')
    a.count = n
    a.use_relative_offset = False
    a.use_constant_offset = True
    a.constant_offset_displace = (0, 0, pitch)
    mod_bevel(fins, 0.0012, 1, 40)
    obs.append(fins)
    return obs


def build_cylinder(prefix, base, direction, M, col, exhaust):
    """exhaust: 'front' (port on the front face) or 'right' (port on the right flank). Plugs sit on the left."""
    obs = []
    dvec = Vector(direction).normalized()
    R = frame_from_z(dvec, Vector((1, 0, 0)))
    BM = mat4(base, R)
    obs += finned_block(prefix + "_Barrel", BM, 0.06, 0.0, 0.165, 0.205, 0.19, 0.0042, 0.0125, col, M)
    hb = bmesh.new()
    bm_box(hb, (0.165, 0.165, 0.085), Matrix.Translation((0, 0, 0.21)))
    head = obj_from_bm(prefix + "_Head", hb, col, M['alu_cast'], BM)
    mod_bevel(head, 0.012, 3, 40)
    obs.append(head)
    obs += finned_block(prefix + "_HeadFin", BM, 0.07, 0.17, 0.25, 0.235, 0.215, 0.004, 0.0135, col, M, corner=0.04)
    rb = bmesh.new()
    bm_box(rb, (0.15, 0.12, 0.04), Matrix.Translation((0, 0, 0.275)))
    rock = obj_from_bm(prefix + "_RockerCover", rb, col, M['alu_cast'], BM)
    mod_bevel(rock, 0.015, 4, 40)
    obs.append(rock)
    for k in range(3):
        rib = box_obj(prefix + "_RockerRib", Vector(), (0.12, 0.006, 0.006), col, M['alu_cast'], bevel=0.002)
        rib.matrix_world = BM @ Matrix.Translation((0, -0.04 + 0.04 * k, 0.296))
        obs.append(rib)
    for sx in (-1, 1):
        for sy in (-1, 1):
            obs.append(bolt(BM @ Vector((sx * 0.06, sy * 0.045, 0.2955)), dvec, col, 'cap', 0.9, prefix + "_RockerBolt"))
            obs.append(bolt(BM @ Vector((sx * 0.085, sy * 0.075, 0.0)), dvec, col, 'nut', 0.55, prefix + "_BaseNut"))
    # spark plug + cap on the LEFT flank, angled up
    sp0 = BM @ Vector((0.08, 0.0, 0.235))
    sp_dir = (R @ Vector((1, 0, 0.9))).normalized()
    obs.append(cyl_obj(prefix + "_SparkPlug_Hex", sp0, sp0 + sp_dir * 0.014, 0.0105, col, M['steel_bright'], segs=6))
    obs.append(cyl_obj(prefix + "_SparkPlug_Ceramic", sp0 + sp_dir * 0.014, sp0 + sp_dir * 0.04, 0.0055, col, M['ceramic'], segs=16))
    obs.append(cyl_obj(prefix + "_PlugCap", sp0 + sp_dir * 0.03, sp0 + sp_dir * 0.072, 0.011, col, M['rubber'], r2=0.008, segs=20, bevel=0.002))
    plug_tip = sp0 + sp_dir * 0.072
    # pushrod tubes (right flank)
    for k in (-1, 1):
        obs.append(cyl_obj(prefix + "_PushrodTube", BM @ Vector((-0.05, k * 0.03, -0.09)), BM @ Vector((-0.07, k * 0.03, 0.27)), 0.0085, col, M['chrome'], segs=16))
    if exhaust == 'front':
        ep = BM @ Vector((-0.035, -0.0825, 0.21))
        ep_dir = (R @ Vector((0, -1, -0.25))).normalized()
    else:
        ep = BM @ Vector((-0.0825, 0.01, 0.21))
        ep_dir = (R @ Vector((-1, 0.25, -0.2))).normalized()
    obs.append(cyl_obj(prefix + "_ExhaustFlange", ep, ep + ep_dir * 0.038, 0.03, col, M['steel_heavy_rust'], r2=0.026, segs=24, bevel=0.002))
    side = (R @ Vector((0, 0, 1))).cross(ep_dir).normalized()
    for k in (-1, 1):
        obs.append(bolt(ep + ep_dir * 0.038 + side * 0.022 * k, ep_dir, col, 'nut', 0.55, prefix + "_ExhaustStudNut"))
    inner = 1 if exhaust == 'front' else -1          # intake faces into the V
    ip = BM @ Vector((-0.02, inner * 0.0825, 0.205))
    ip_dir = (R @ Vector((0, inner, 0.0))).normalized()
    obs.append(cyl_obj(prefix + "_IntakeStub", ip, ip + ip_dir * 0.012, 0.021, col, M['alu_cast'], segs=24))
    rocker_banjo = BM @ Vector((-0.077, -0.035 if exhaust == 'right' else 0.02, 0.275))
    return obs, ep + ep_dir * 0.036, ep_dir, ip + ip_dir * 0.012, BM @ Vector((0, 0, 0.21)), plug_tip, rocker_banjo, BM


def caliper(name, axle, a, r_rotor, g, col, M, r_c=None, length=0.088):
    """Two-piece caliper straddling the rotor: inner + outer halves with a slot, bridge outside the disc edge."""
    obs = []
    r_c = r_c if r_c is not None else r_rotor - 0.024
    rd = Vector((0, cos(g), sin(g)))
    tg = Vector((0, -sin(g), cos(g)))
    Rm = Matrix((Vector((1, 0, 0)), tg, rd)).transposed()          # local x axial, y tangential, z radial
    base = axle + Vector((a, 0, 0)) + rd * r_c
    span = r_rotor + 0.012 - (r_c - 0.022)
    zc = (r_rotor + 0.012 + r_c - 0.022) / 2 - r_c
    for nm, sx in (("Outer", -1), ("Inner", 1)):
        c = base + Vector((sx * 0.0135, 0, 0)) + rd * zc
        ob = box_obj(name + "_Caliper" + nm, c, (0.017, length, span), col, M['alu_cast'], bevel=0.006, segs=3)
        ob.matrix_world = mat4(c, Rm)
        obs.append(ob)
        pad = box_obj(name + "_Pad" + nm, Vector(), (0.0035, length * 0.62, 0.032), col, M['dark_metal'], bevel=0.001)
        pad.matrix_world = mat4(base + Vector((sx * 0.0045, 0, 0)) + rd * 0.004, Rm)
        obs.append(pad)
    br = box_obj(name + "_CaliperBridge", Vector(), (0.044, length * 0.8, 0.008), col, M['alu_cast'], bevel=0.003)
    br.matrix_world = mat4(base + rd * (r_rotor - r_c + 0.008), Rm)
    obs.append(br)
    for k in (-1, 1):
        pc = base + Vector((-0.022, 0, 0)) + tg * 0.022 * k * (length / 0.088)
        obs.append(cyl_obj(name + "_CaliperPistonCap", pc, pc + Vector((-0.006, 0, 0)), 0.0145, col, M['alu_cast'], segs=24, bevel=0.002))
        obs.append(bolt(base + Vector((-0.022, 0, 0)) + tg * 0.036 * k * (length / 0.088) + rd * 0.012, (-1, 0, 0), col, 'cap', 1.0, name + "_BridgeBolt"))
    bl = base + Vector((-0.016, 0, 0)) + rd * (r_rotor - r_c + 0.004) - tg * 0.02
    obs.append(cyl_obj(name + "_BleedNipple", bl, bl + rd * 0.012, 0.003, col, M['steel_bright'], segs=6))
    return obs


def banjo_fitting(name, p, normal, col, M):
    n = Vector(normal).normalized()
    obs = [cyl_obj(name + "_Eye", p - n * 0.005, p + n * 0.005, 0.0085, col, M['steel_bright'], segs=20, bevel=0.0008)]
    obs.append(bolt(p + n * 0.005, n, col, 'bolt', 0.75, name + "_Bolt"))
    return obs


def banjo(name, p, normal, col, M, r=0.0075):
    n = Vector(normal).normalized()
    obs = [cyl_obj(name + "_Eye", p - n * 0.004, p + n * 0.004, r, col, M['steel_bright'], segs=20, bevel=0.0008)]
    obs.append(bolt(p + n * 0.004, n, col, 'bolt', 0.7, name + "_Bolt"))
    return obs


def build_engine(M, root):
    col = coll('Engine', COLL['Bike'])
    obs = []
    obs.append(box_obj("Engine_Crankcase", Vector((0, -0.02, 0.40)), (0.196, 0.42, 0.24), col, M['alu_cast'], bevel=0.045, segs=5))
    obs.append(cyl_obj("Engine_Crankcase_Nose", Vector((-0.096, -0.155, 0.39)), Vector((0.096, -0.155, 0.39)), 0.10, col, M['alu_cast'], segs=48, bevel=0.012))
    obs.append(box_obj("Engine_Gearbox", Vector((0, 0.12, 0.465)), (0.18, 0.15, 0.11), col, M['alu_cast'], bevel=0.035, segs=4))
    obs.append(box_obj("Engine_CaseSeam", Vector((0, -0.02, 0.40)), (0.006, 0.43, 0.245), col, M['dark_metal'], bevel=0.003))
    bm = bmesh.new()
    bm_box(bm, (0.17, 0.005, 0.026))
    sf = obj_from_bm("Engine_SumpFins", bm, col, M['alu_fins'], mat4(Vector((0, -0.20, 0.29))))
    a = sf.modifiers.new("FinArray", 'ARRAY')
    a.count = 14
    a.use_relative_offset = False
    a.use_constant_offset = True
    a.constant_offset_displace = (0, 0.024, 0)
    mod_bevel(sf, 0.001, 1)
    obs.append(sf)
    obs.append(cyl_obj("Engine_DrainPlug", Vector((0, 0.02, 0.282)), Vector((0, 0.02, 0.268)), 0.01, col, M['steel_bright'], segs=6))
    # right: chrome clutch/primary cover, points cover, kick-start shaft
    cc = Vector((-0.098, 0.04, 0.405))
    obs.append(revolve_obj("Engine_ClutchCover", [(0.0, 0.0), (0.0, 0.096), (-0.006, 0.099), (-0.02, 0.095), (-0.036, 0.078), (-0.046, 0.045),
                                                  (-0.048, 0.0)], col, M['chrome'], mat4(cc), segs=64, axis='X', angle=50))
    obs.append(revolve_obj("Engine_ClutchCover_Cap", [(-0.046, 0.0), (-0.046, 0.04), (-0.052, 0.04), (-0.056, 0.033), (-0.056, 0.0)], col,
                           M['steel_bright'], mat4(cc), segs=48, axis='X'))
    bolt_ring(cc + Vector((-0.011, 0, 0)), (-1, 0, 0), 0.087, 10, col, 'cap', 0.9, name="Engine_ClutchCoverScrew")
    bolt_ring(cc + Vector((-0.056, 0, 0)), (-1, 0, 0), 0.026, 3, col, 'cap', 0.7, name="Engine_ClutchCapScrew")
    pc = Vector((-0.098, -0.165, 0.37))
    obs.append(revolve_obj("Engine_PointsCover", [(0.0, 0.0), (0.0, 0.06), (-0.006, 0.062), (-0.018, 0.055), (-0.024, 0.03), (-0.025, 0.0)],
                           col, M['chrome'], mat4(pc), segs=48, axis='X'))
    bolt_ring(pc + Vector((-0.008, 0, 0)), (-1, 0, 0), 0.05, 5, col, 'cap', 0.7, name="Engine_PointsCoverScrew")
    # oil filter canister under the front, between the cradle tubes (left of the header)
    obs.append(cyl_obj("Engine_OilFilter", Vector((0.032, -0.222, 0.29)), Vector((0.032, -0.242, 0.247)), 0.03, col, M['paint_black'], segs=32, bevel=0.004))
    obs.append(cyl_obj("Engine_OilFilterNut", Vector((0.032, -0.242, 0.247)), Vector((0.032, -0.246, 0.239)), 0.009, col, M['steel_bright'], segs=6))
    # left: alternator cover, front sprocket on the chain line, sprocket cover (front half only)
    ac = Vector((0.098, -0.13, 0.385))
    obs.append(revolve_obj("Engine_AlternatorCover", [(0.0, 0.0), (0.0, 0.1), (0.006, 0.103), (0.02, 0.098), (0.035, 0.07), (0.042, 0.03),
                                                       (0.043, 0.0)], col, M['alu_cast'], mat4(ac), segs=64, axis='X'))
    bolt_ring(ac + Vector((0.01, 0, 0)), (1, 0, 0), 0.09, 9, col, 'cap', 0.9, name="Engine_AltCoverScrew")
    obs.append(cyl_obj("Engine_AltGrommet", ac + Vector((0.012, 0.05, 0.08)), ac + Vector((0.02, 0.05, 0.08)), 0.007, col, M['rubber'], segs=16))
    anchor('alt_out', ac + Vector((0.022, 0.05, 0.082)))
    cx = P['chain_x']
    obs.append(build_sprocket("Engine_FrontSprocket", Vector((cx, 0.11, 0.42)), 15, 0.0383, 0.007, col, M))
    obs.append(cyl_obj("Engine_Countershaft", Vector((0.095, 0.11, 0.42)), Vector((cx + 0.008, 0.11, 0.42)), 0.019, col, M['steel_rusty'], segs=24))
    obs.append(bolt(Vector((cx + 0.008, 0.11, 0.42)), (1, 0, 0), col, 'nut', 1.2, "Engine_SprocketNut"))
    obs.append(beam_obj("Engine_SprocketCover", Vector((0.135, 0.035, 0.455)), Vector((0.135, 0.105, 0.455)), cx + 0.03 - 0.1, 0.12, cx + 0.03 - 0.1, 0.1,
                        col, M['alu_cast'], up=(0, 0, 1), bevel=0.015, segs=3))
    # gear indicator / neutral switch, oil pressure switch, clutch actuator arm
    obs.append(cyl_obj("Elec_NeutralSwitch", Vector((0.09, 0.16, 0.425)), Vector((0.105, 0.16, 0.425)), 0.009, col, M['steel_bright'], segs=6))
    anchor('neutral_sw', Vector((0.108, 0.16, 0.425)))
    obs.append(cyl_obj("Elec_OilPressureSwitch", Vector((-0.09, -0.21, 0.47)), Vector((-0.108, -0.21, 0.478)), 0.0095, col, M['steel_bright'], segs=6))
    anchor('oilp_sw', Vector((-0.11, -0.21, 0.479)))
    obs.append(cyl_obj("Engine_ClutchActuatorBoss", Vector((0.098, -0.02, 0.50)), Vector((0.118, -0.02, 0.50)), 0.017, col, M['alu_cast'], segs=24, bevel=0.002))
    arm = [Vector((0.12, -0.02, 0.50)), Vector((0.125, -0.06, 0.53)), Vector((0.127, -0.075, 0.545))]
    obs.append(curve_obj("Engine_ClutchArm", arm, 0.005, col, M['steel_rusty'], mode='bend', bend=0.01))
    obs.append(cyl_obj("Engine_ClutchCableBracket", Vector((0.125, 0.02, 0.555)), Vector((0.125, 0.02, 0.575)), 0.006, col, M['steel_rusty'], segs=12))
    anchor('clutch_arm', Vector((0.127, 0.025, 0.582)))
    anchor('clutch_arm_end', Vector((0.127, -0.075, 0.548)))
    # cylinders (V-twin): front port forward, rear port to the right side
    a1, a2 = rad(28), rad(22)
    fobs, f_port, f_dir, f_int, f_head, f_plug, f_banjo, fBM = build_cylinder("Engine_FrontCyl", Vector((0, -0.075, 0.52)), (0, -sin(a1), cos(a1)), M, col, 'front')
    robs, r_port, r_dir, r_int, r_head, r_plug, r_banjo, rBM = build_cylinder("Engine_RearCyl", Vector((0, 0.065, 0.52)), (0, sin(a2), cos(a2)), M, col, 'right')
    obs += fobs + robs
    anchor('plug_F', f_plug)
    anchor('plug_R', r_plug)
    # carburettor in the V + pod filter on the right
    cb = Vector((-0.04, -0.0125, 0.725))
    obs.append(curve_obj("Engine_IntakeManifold_F", [f_int, cb + Vector((0, -0.03, 0.0))], 0.02, col, M['alu_cast']))
    obs.append(curve_obj("Engine_IntakeManifold_R", [r_int, cb + Vector((0, 0.03, 0.0))], 0.02, col, M['alu_cast']))
    obs.append(box_obj("Engine_Carb_Body", cb, (0.06, 0.06, 0.062), col, M['alu_cast'], bevel=0.012, segs=3))
    obs.append(cyl_obj("Engine_Carb_FloatBowl", cb + Vector((0, 0, -0.03)), cb + Vector((0, 0, -0.062)), 0.025, col, M['alu_cast'], segs=32, bevel=0.003))
    obs.append(cyl_obj("Engine_Carb_Top", cb + Vector((0, 0, 0.03)), cb + Vector((0, 0, 0.052)), 0.021, col, M['chrome'], segs=32, bevel=0.003))
    for k, dx in enumerate((-0.008, 0.008)):
        obs.append(cyl_obj("Engine_Carb_CableAdjuster", cb + Vector((dx, 0, 0.052)), cb + Vector((dx, 0, 0.066)), 0.0042, col, M['steel_bright'], segs=12))
        anchor('carb_throttle_%d' % k, cb + Vector((dx, 0, 0.068)))
    obs.append(cyl_obj("Engine_Carb_FuelInlet", cb + Vector((0.03, 0.0, -0.012)), cb + Vector((0.045, 0.0, -0.012)), 0.0045, col, M['brass'], segs=12))
    anchor('carb_fuel_in', cb + Vector((0.046, 0.0, -0.012)))
    obs.append(cyl_obj("Engine_Carb_Throat", cb + Vector((-0.03, 0, 0)), cb + Vector((-0.07, 0, 0)), 0.027, col, M['alu_cast'], segs=32))
    filt_c = cb + Vector((-0.07, 0, 0))
    prof = [(0.0, 0.0), (0.0, 0.05), (-0.006, 0.058)]
    for i in range(14):
        a = -0.008 - i * 0.0045
        prof += [(a, 0.056), (a - 0.00225, 0.06)]
    prof += [(-0.075, 0.058), (-0.08, 0.05), (-0.082, 0.0)]
    obs.append(revolve_obj("Engine_AirFilter", prof, col, M['steel_rusty'], mat4(filt_c), segs=64, axis='X', angle=50))
    obs.append(revolve_obj("Engine_AirFilter_EndCap", [(-0.079, 0.0), (-0.079, 0.059), (-0.086, 0.058), (-0.09, 0.03), (-0.091, 0.0)],
                           col, M['chrome'], mat4(filt_c), segs=48, axis='X'))
    obs.append(cyl_obj("Engine_AirFilter_Nut", filt_c + Vector((-0.09, 0, 0)), filt_c + Vector((-0.1, 0, 0)), 0.01, col, M['steel_bright'], segs=6))
    obs.append(cyl_obj("Engine_AirFilter_Clamp", filt_c + Vector((0.004, 0, 0)), filt_c + Vector((-0.004, 0, 0)), 0.0295, col, M['zinc'], segs=32))
    # twin-tower ignition coil under the tank, left of the backbone
    coil = Vector((0.062, -0.18, 0.862))
    obs.append(cyl_obj("Elec_IgnitionCoil", coil, coil + Vector((0, 0.12, 0.0)), 0.024, col, M['paint_black'], segs=24, bevel=0.003))
    obs.append(box_obj("Elec_CoilBracket", Vector((0.04, coil.y + 0.06, 0.885)), (0.004, 0.08, 0.03), col, M['steel_rusty'], bevel=0.001))
    for k, y in enumerate((0.015, 0.105)):
        tw = coil + Vector((0.0, y, -0.024))
        obs.append(cyl_obj("Elec_CoilTower", tw, tw + Vector((0, 0, -0.012)), 0.008, col, M['plastic_black'], segs=16))
        anchor('coil_tower_%d' % k, tw + Vector((0, 0, -0.014)))
    anchor('coil_lv', coil + Vector((0.0, 0.06, 0.026)))
    # purposeful plumbing: external rocker oil feeds with banjo fittings, crankcase breather, fuel line
    gal = Vector((-0.1, -0.05, 0.50))
    obs.append(cyl_obj("Engine_OilGalleryBoss", Vector((-0.095, -0.05, 0.50)), Vector((-0.104, -0.05, 0.50)), 0.012, col, M['alu_cast'], segs=6))
    obs += banjo("Engine_OilFeedBanjo_Case", gal + Vector((-0.008, 0, 0)), (-1, 0, 0), col, M)
    for nm, bj in (("F", f_banjo), ("R", r_banjo)):
        obs += banjo("Engine_OilFeedBanjo_" + nm, bj, (-1, 0, 0), col, M, r=0.0065)
    for nm, BMc, bj, ly, dy in (("F", fBM, f_banjo, 0.03, -0.007), ("R", rBM, r_banjo, -0.07, 0.007)):
        pts = [gal + Vector((-0.011, dy, 0.0)), gal + Vector((-0.035, dy * 2, 0.01)), BMc @ Vector((-0.14, ly, 0.06)), BMc @ Vector((-0.14, ly, 0.24)),
               BMc @ Vector((-0.12, ly * 0.6, 0.272)), bj + (BMc.to_3x3() @ Vector((-0.009, 0, 0)))]
        obs.append(cable_obj("Hose_OilFeed_" + nm, pts, 0.0042, col, M['copper_rust']))
    br0 = Vector((0.0, 0.17, 0.525))
    obs.append(cyl_obj("Engine_BreatherSpigot", br0 - Vector((0, 0, 0.005)), br0 + Vector((0, 0.008, 0.012)), 0.007, col, M['alu_cast'], segs=16))
    obs.append(cable_obj("Hose_Breather", [br0 + Vector((0, 0.008, 0.012)), Vector((0.05, 0.19, 0.538)), Vector((0.1, 0.19, 0.47)), Vector((0.103, 0.185, 0.37)),
                                           Vector((0.104, 0.17, 0.30))], 0.0072, col, M['hose']))
    for p in (br0 + Vector((0, 0.005, 0.008)), Vector((0.103, 0.188, 0.40))):
        obs.append(cyl_obj("Engine_HoseClamp", p + Vector((0, 0, -0.004)), p + Vector((0, 0, 0.004)), 0.0102, col, M['zinc'], segs=16))
    return obs, (f_port, f_dir), (r_port, r_dir)


# ---------------------------------------------------------------------------------------------
# EXHAUST: front header runs down the engine front between the cradle tubes and under the
# crankcase; rear header leaves the right flank of the rear head. Twin mufflers on the right.
# ---------------------------------------------------------------------------------------------

def build_exhaust(M, root, fport, rport):
    col = coll('Exhaust', COLL['Bike'])
    obs = []
    fp, fd = fport
    rp, rd = rport
    MX = -0.235
    lower = dict(c=Vector((MX, 0.0, 0.705)), r=0.039, y0=0.50, y1=0.95)
    upper = dict(c=Vector((MX, 0.0, 0.79)), r=0.043, y0=0.50, y1=0.975)
    hdr_f = [fp, fp + fd * 0.035, Vector((-0.033, -0.312, 0.585)), Vector((-0.027, -0.305, 0.42)), Vector((-0.024, -0.29, 0.25)),
             Vector((-0.024, -0.225, 0.174)), Vector((-0.026, 0.07, 0.172)), Vector((-0.09, 0.19, 0.182)), Vector((-0.2, 0.265, 0.225)),
             Vector((-0.25, 0.33, 0.33)), Vector((-0.25, 0.40, 0.55)), Vector((-0.24, 0.462, 0.672)), Vector((MX, 0.505, 0.705))]
    obs.append(cable_obj("Exhaust_Header_Front", hdr_f, 0.0215, col, M['steel_heavy_rust'], res=20, bevel_res=6))
    hdr_r = [rp, rp + rd * 0.03, Vector((-0.15, 0.20, 0.738)), Vector((-0.178, 0.30, 0.764)), Vector((-0.215, 0.42, 0.789)), Vector((MX, 0.505, 0.79))]
    obs.append(cable_obj("Exhaust_Header_Rear", hdr_r, 0.0215, col, M['steel_heavy_rust'], res=20, bevel_res=6))
    obs.append(cable_obj("Exhaust_HeaderWrap", hdr_f[2:5], 0.0235, col,
                         mat_cache('wrap', lambda: mat_metal("M_HeatWrap", (0.22, 0.2, 0.17), 0.85, rust=0.5, grime=0.7, scale=30,
                                                             rust_cols=((0.06, 0.05, 0.04), (0.15, 0.10, 0.06), (0.25, 0.18, 0.10)))),
                         res=20, bevel_res=6))
    for t, nm in ((hdr_f[2], "A"), (hdr_f[4], "B")):
        obs.append(cyl_obj("Exhaust_WrapClamp" + nm, t + Vector((0, -0.002, -0.004)), t + Vector((0, 0.002, 0.004)), 0.0255, col, M['zinc'], segs=24))
    for nm, sp_ in (("Lower", lower), ("Upper", upper)):
        L_ = sp_['y1'] - sp_['y0']
        r = sp_['r']
        prof = [(0.0, 0.0), (0.0, 0.0225), (0.03, 0.026), (0.07, r), (0.075, r + 0.002), (0.08, r)]
        prof += [(L_ - 0.045, r), (L_ - 0.04, r + 0.002), (L_ - 0.035, r), (L_ - 0.01, r * 0.97), (L_, r * 0.86), (L_, r * 0.62),
                 (L_ - 0.006, r * 0.6), (L_ - 0.006, 0.0)]
        Mm = mat4(sp_['c'] + Vector((0, sp_['y0'], 0)), Matrix.Rotation(rad(-6), 3, 'X'))
        obs.append(revolve_obj("Exhaust_Muffler_" + nm, prof, col, M['chrome'], Mm, segs=48, axis='Y', angle=40))
        obs.append(revolve_obj("Exhaust_Muffler_" + nm + "_Outlet", [(L_ - 0.005, r * 0.45), (L_ + 0.018, r * 0.45), (L_ + 0.018, r * 0.38), (L_ - 0.03, r * 0.38)],
                               col, M['steel_heavy_rust'], Mm, segs=32, axis='Y', closed=True))
        for t in (0.13, L_ - 0.12):
            obs.append(revolve_obj("Exhaust_Strap_" + nm, [(t - 0.009, r + 0.0005), (t - 0.009, r + 0.003), (t + 0.009, r + 0.003), (t + 0.009, r + 0.0005)],
                                   col, M['steel_rusty'], Mm, segs=48, axis='Y', closed=True))
    # hangers: upper muffler strap -> seat rail, lower muffler -> upper muffler
    obs.append(beam_obj("Exhaust_Hanger_Upper", Vector((MX + 0.03, 0.64, 0.84)), Vector((-0.088, 0.64, 0.862)), 0.012, 0.025, 0.012, 0.025, col,
                        M['steel_rusty'], up=(0, 1, 0), bevel=0.002, segs=1))
    obs.append(beam_obj("Exhaust_Hanger_Link", Vector((MX - 0.0, 0.81, 0.742)), Vector((MX - 0.0, 0.81, 0.75)), 0.02, 0.012, 0.02, 0.012, col,
                        M['steel_rusty'], up=(0, 1, 0), bevel=0.002, segs=1))
    obs.append(box_obj("Exhaust_HeatShield", Vector((MX - 0.046, 0.75, 0.79)), (0.006, 0.22, 0.06), col, M['steel_heavy_rust'], rot=(-6, 0, 0), bevel=0.003))
    for y in (0.67, 0.83):
        obs.append(bolt(Vector((MX - 0.05, y, 0.79 + (y - 0.75) * 0.1)), (-1, 0, 0), col, 'bolt', 0.8, "Exhaust_HeatShieldBolt"))
    obs.append(box_obj("Exhaust_HeaderHanger", Vector((-0.026, 0.02, 0.235)), (0.012, 0.012, 0.09), col, M['steel_rusty'], bevel=0.0015))
    return obs


# ---------------------------------------------------------------------------------------------
# REAR END (swingarm, monoshock, chain, rear wheel, brake)
# ---------------------------------------------------------------------------------------------

def build_rear(M, root):
    col = coll('RearEnd', COLL['Bike'])
    wcol = coll('Wheels', COLL['Bike'])
    sp = P['swing_pivot']
    RA = P['rear_axle']
    ax_ = P['arm_x']
    cx = P['chain_x']
    piv = make_empty("RearEnd_SwingarmPivot", mat4(sp), col, 'SINGLE_ARROW', 0.2)
    parent_later(piv, root)
    obs = []
    Yh = M['paint_yellow_heavy']
    arm = beam_obj("Swingarm_Arms", Vector((ax_, sp.y + 0.02, sp.z)), Vector((ax_, RA.y + 0.05, RA.z)), 0.036, 0.082, 0.032, 0.056, col, Yh, bevel=0.008, segs=3)
    mod_mirror(arm, mirror_obj=root)
    obs.append(arm)
    boss = cyl_obj("Swingarm_PivotBoss", Vector((0.137, sp.y, sp.z)), Vector((ax_ + 0.02, sp.y, sp.z)), 0.03, col, Yh, segs=32, bevel=0.003)
    mod_mirror(boss, mirror_obj=root)
    obs.append(boss)
    # cross brace sits between the pivot plates' rear edge and the tyre (clears both through the travel)
    by = sp.y + 0.085
    obs.append(beam_obj("Swingarm_CrossBrace", Vector((-ax_ + 0.017, by, sp.z + 0.01)), Vector((ax_ - 0.017, by, sp.z + 0.01)),
                        0.044, 0.055, 0.044, 0.055, col, Yh, bevel=0.008))
    for sx in (-1, 1):
        obs.append(box_obj("Swingarm_ShockClevis", Vector((sx * 0.02, by, sp.z + 0.06)), (0.007, 0.04, 0.05), col, Yh, bevel=0.002))
    obs.append(beam_obj("Swingarm_BoxBrace_R", Vector((-ax_, sp.y + 0.09, sp.z + 0.045)), Vector((-ax_, RA.y - 0.09, RA.z + 0.04)),
                        0.036, 0.05, 0.032, 0.034, col, Yh, bevel=0.007))
    for t in (0.25, 0.5, 0.75):
        obs.append(bolt(Vector((-ax_ - 0.018, lerp(sp.y + 0.09, RA.y - 0.09, t), lerp(sp.z + 0.045, RA.z + 0.04, t))), (-1, 0, 0), col, 'bolt', 1.0, "Swingarm_BraceBolt"))
    for sx in (-1, 1):
        obs.append(box_obj("Swingarm_AxleBlock", Vector((sx * ax_, RA.y, RA.z)), (0.04, 0.05, 0.04), col, M['alu_cast'], bevel=0.004))
        obs.append(cyl_obj("Swingarm_Adjuster", Vector((sx * ax_, RA.y + 0.025, RA.z + 0.008)), Vector((sx * ax_, RA.y + 0.075, RA.z + 0.008)),
                           0.004, col, M['steel_bright'], segs=12))
        obs.append(bolt(Vector((sx * ax_, RA.y + 0.072, RA.z + 0.008)), (0, 1, 0), col, 'nut', 0.55, "Swingarm_AdjusterNut"))
    obs.append(box_obj("Swingarm_ChainSlider", Vector((ax_ - 0.0215, sp.y + 0.12, sp.z + 0.02)), (0.005, 0.16, 0.05), col, M['plastic_black'], bevel=0.0015))
    # chain guard over the top run
    obs.append(beam_obj("Swingarm_ChainGuard", Vector((cx + 0.004, 0.27, 0.505)), Vector((cx + 0.004, 0.62, 0.522)), 0.05, 0.004, 0.05, 0.004, col,
                        M['paint_black'], bevel=0.0012, segs=1))
    obs.append(beam_obj("Swingarm_ChainGuardLip", Vector((cx + 0.029, 0.27, 0.49)), Vector((cx + 0.029, 0.62, 0.507)), 0.003, 0.03, 0.003, 0.03, col,
                        M['paint_black'], bevel=0.001, segs=1))
    for y in (0.33, 0.56):
        z_arm = lerp(sp.z, RA.z, (y - sp.y) / (RA.y - sp.y))
        obs.append(box_obj("Swingarm_ChainGuardTab", Vector((ax_ - 0.012, y, (z_arm + 0.038 + 0.505) / 2 + 0.005)), (0.006, 0.02, 0.505 - z_arm - 0.02),
                           col, M['paint_black'], bevel=0.001))
    # rear hugger (front-upper quadrant only so it clears the battery/electrics at full bump)
    Rh = P['rear_tire']['R'] + 0.028
    secx = [(-0.12, -0.022), (-0.115, 0.0), (0.115, 0.0), (0.12, -0.022)]
    bm = bmesh.new()
    rows = []
    for i in range(13):
        b = rad(116 + 44 * i / 12)
        rows.append([bm.verts.new(Vector((x, cos(b) * (Rh + dz), sin(b) * (Rh + dz)))) for x, dz in secx])
    for i in range(12):
        for j in range(3):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    hug = obj_from_bm("Fender_Rear_Hugger", bm, col, M['paint_yellow'], mat4(RA), angle=50)
    mod_solidify(hug, 0.003, 1)
    mod_subsurf(hug, 1, 2)
    obs.append(hug)
    for sx in (-1, 1):       # short stays from the hugger's leading edge down onto the cross brace
        hb = RA + Vector((sx * 0.105, cos(rad(158)) * (Rh - 0.004), sin(rad(158)) * (Rh - 0.004)))
        obs.append(beam_obj("Fender_Rear_HuggerStay", hb, Vector((sx * 0.105, by + 0.004, sp.z + 0.01 + 0.027)),
                            0.004, 0.014, 0.004, 0.014, col, M['steel_rusty'], bevel=0.001, segs=1))
    # rear brake: caliper below the right arm, bracket + torque arm, banjo
    ra = -0.09
    g = rad(205)
    cc = RA + Vector((ra, 0.09 * cos(g), 0.09 * sin(g)))
    obs += caliper("Brake_Rear", RA, ra, 0.112, g, col, M, r_c=0.09, length=0.07)
    brk = Vector((-(ax_ + ra * -1) / 2 - 0.0, cc.y + 0.03, cc.z + 0.02))
    obs.append(beam_obj("Brake_Rear_Bracket", Vector((ra - 0.022, cc.y + 0.035, cc.z + 0.012)), Vector((-ax_ + 0.018, RA.y - 0.01, RA.z)),
                        0.035, 0.008, 0.035, 0.008, col, M['steel_rusty'], up=(0, 1, 0), bevel=0.002, segs=1))
    ta0 = Vector((-(ax_ - 0.026), cc.y - 0.005, cc.z - 0.006))
    ta1 = Vector((-(ax_ - 0.026), 0.42, lerp(sp.z, RA.z, (0.42 - sp.y) / (RA.y - sp.y)) - 0.05))
    obs.append(curve_obj("Brake_Rear_TorqueArm", [ta0, ta1], 0.0065, col, M['steel_bright']))
    obs.append(box_obj("Brake_Rear_TorqueTab", ta1 + Vector((0, 0, 0.02)), (0.006, 0.02, 0.04), col, Yh, bevel=0.0015))
    for p in (ta0, ta1):
        obs.append(bolt(p + Vector((-0.008, 0, 0)), (-1, 0, 0), col, 'nut', 0.6, "Brake_Rear_TorqueArmNut"))
    tgr = Vector((0, -sin(g), cos(g)))
    bj = RA + Vector((ra - 0.025, 0, 0)) + Vector((0, cos(g), sin(g))) * 0.09 + tgr * 0.024
    obs += banjo_fitting("Brake_Rear_CaliperBanjo", bj, (-1, 0, 0), col, M)
    anchor('caliper_banjo_R', bj + Vector((-0.012, 0.0, 0.004)), 'swing')
    rw = build_wheel("RearWheel", RA, P['rear_tire'], wcol, M, rotor=dict(a=ra, r_out=0.112, r_in=0.05, holes=16, windows=6),
                     sprocket=dict(a=cx, N=42, r_p=0.1062), axle_half=ax_ + 0.022)
    parent_later(rw, piv)

    # chain: a 2-pitch link module arrayed along a closed path (Curve modifier) on the chain line
    p = 0.015875
    c1, r1 = Vector((0.11, 0.42)), 0.0383
    c2, r2 = Vector((RA.y, RA.z)), 0.1062
    dv = c2 - c1
    Ld = dv.length
    alpha = math.atan2(dv.y, dv.x)
    phi = math.acos((r1 - r2) / Ld)
    path = []

    def arc(c, r, a0, a1, n):
        for i in range(n + 1):
            a = a0 + (a1 - a0) * i / n
            path.append(Vector((c.x + r * cos(a), c.y + r * sin(a))))
    up, lo = alpha + phi, alpha - phi
    arc(c2, r2, up, lo, 40)
    arc(c1, r1, lo, up - 2 * pi, 24)
    cur = poly_curve_obj("Chain_Path", [(q.x, q.y, 0.0) for q in path], col, closed=True)
    cur.data.dimensions = '2D'
    cur.data.fill_mode = 'NONE'
    Mc = mat4(Vector((cx, 0, 0)), Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0))))
    cur.matrix_world = Mc
    cur.hide_render = True
    tot = sum((path[(i + 1) % len(path)] - path[i]).length for i in range(len(path)))
    n_mod = int(round(tot / (2 * p)))
    pe = tot / (2 * n_mod)
    bm = bmesh.new()

    def plate(z, w, x0):
        pts = []
        for i in range(10):
            a = pi / 2 + pi * i / 9
            pts.append((x0 + w * cos(a), w * sin(a)))
        pts.append((x0 + pe / 2, -w * 0.78))
        for i in range(10):
            a = -pi / 2 + pi * i / 9
            pts.append((x0 + pe + w * cos(a), w * sin(a)))
        pts.append((x0 + pe / 2, w * 0.78))
        extrude_bm(bm, pts, 0.0014, Matrix.Translation((0, 0, z)))
    for z in (-0.0062, 0.0062):
        plate(z, 0.0074, 0.0)
    for z in (-0.0045, 0.0045):
        plate(z, 0.0072, pe)
    for x in (0.0, pe):
        bm_cyl(bm, 0.0049, 0.0049, 0.0078, 12, Matrix.Translation((x, 0, -0.0039)))
        bm_cyl(bm, 0.0025, 0.0025, 0.0158, 8, Matrix.Translation((x, 0, -0.0079)))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    chain = obj_from_bm("Chain_Links", bm, col, M['steel_rusty'], Mc, angle=40)
    a = chain.modifiers.new("ChainArray", 'ARRAY')
    a.fit_type = 'FIXED_COUNT'
    a.count = n_mod
    a.use_relative_offset = False
    a.use_constant_offset = True
    a.constant_offset_displace = (2 * pe, 0, 0)
    cm = chain.modifiers.new("FollowPath", 'CURVE')
    cm.object = cur
    cm.deform_axis = 'POS_X'
    obs += [cur, chain]

    # monoshock (constraints keep it attached through the travel)
    top = Vector((0.0, 0.345, 0.835))
    bot = Vector((0.0, by, sp.z + 0.075))
    axd = (bot - top).normalized()
    Lsh = (bot - top).length
    Rt = frame_from_z(axd)
    upper = make_empty("Shock_Upper", mat4(top, Rt), col, 'PLAIN_AXES', 0.03)
    lower = make_empty("Shock_Lower", mat4(bot, frame_from_z(-axd)), col, 'PLAIN_AXES', 0.03)
    top_m = make_empty("Shock_TopMount", mat4(top), col, 'SPHERE', 0.015)
    bot_m = make_empty("Shock_BottomMount", mat4(bot), col, 'SPHERE', 0.015)
    parent_later(upper, root)
    parent_later(lower, piv)
    parent_later(top_m, root)
    parent_later(bot_m, piv)
    c = upper.constraints.new('DAMPED_TRACK')
    c.target = bot_m
    c.track_axis = 'TRACK_Z'
    c = lower.constraints.new('DAMPED_TRACK')
    c.target = top_m
    c.track_axis = 'TRACK_Z'
    sh_up, sh_lo = [], []
    body_prof = [(0.0, 0.0), (0.0, 0.016), (0.012, 0.016), (0.016, 0.024), (0.15, 0.024), (0.155, 0.021), (0.16, 0.012), (0.16, 0.0)]
    sh_up.append(revolve_obj("Shock_Body", body_prof, col, M['alu_cast'], mat4(top, Rt), segs=32, axis='Z'))
    sh_up.append(cyl_obj("Shock_TopEye", top + Vector((-0.012, 0, 0)), top + Vector((0.012, 0, 0)), 0.014, col, M['alu_cast'], segs=24))
    res0 = top + axd * 0.04 + Vector((0.043, 0.0, 0.0))
    sh_up.append(cyl_obj("Shock_Reservoir", res0, res0 + axd * 0.11, 0.018, col, M['paint_red'], segs=32, bevel=0.003))
    sh_up.append(cable_obj("Shock_ReservoirHose", [top + axd * 0.05 + Vector((0.022, 0, 0)), top + axd * 0.036 + Vector((0.034, 0, 0.0)), res0 + Vector((0, 0, 0.004))],
                           0.004, col, M['braided']))
    sh_up.append(revolve_obj("Shock_PreloadCollar", [(0.04, 0.025), (0.04, 0.033), (0.052, 0.033), (0.052, 0.025)], col, M['paint_red'],
                             mat4(top, Rt), segs=32, axis='Z', closed=True))
    Rb = frame_from_z(-axd)
    sh_lo.append(cyl_obj("Shock_Shaft", bot, bot - axd * (Lsh - 0.14), 0.0075, col, M['chrome'], segs=20))
    sh_lo.append(cyl_obj("Shock_BottomEye", bot + Vector((-0.012, 0, 0)), bot + Vector((0.012, 0, 0)), 0.013, col, M['alu_cast'], segs=24))
    sh_lo.append(revolve_obj("Shock_LowerSeat", [(0.018, 0.0), (0.018, 0.03), (0.026, 0.034), (0.03, 0.034), (0.03, 0.0)], col, M['alu_cast'],
                             mat4(bot, Rb), segs=32, axis='Z'))
    s_len = Lsh - 0.03 - 0.052
    turns = 7
    up_dir = -axd
    xa = Vector((1, 0, 0))
    za = xa.cross(up_dir).normalized()
    Rs = Matrix((xa, up_dir, za)).transposed()
    bm = bmesh.new()
    circ = [bm.verts.new((0.031 + 0.0055 * cos(a), 0.0055 * sin(a), 0.0)) for a in [2 * pi * i / 10 for i in range(10)]]
    for i in range(10):
        bm.edges.new((circ[i], circ[(i + 1) % 10]))
    spring = obj_from_bm("Shock_Spring", bm, col, M['paint_red'], mat4(bot - axd * 0.03, Rs), uv=False)
    scr = spring.modifiers.new("Coil", 'SCREW')
    scr.axis = 'Y'
    scr.angle = 2 * pi * turns
    scr.screw_offset = s_len
    scr.steps = 16 * turns
    scr.render_steps = 24 * turns
    scr.use_merge_vertices = False
    scr.use_normal_calculate = True
    scr.use_smooth_shade = True
    spring_top = make_empty("Shock_SpringTopTarget", mat4(top + axd * 0.052, Rt), col, 'PLAIN_AXES', 0.02)
    parent_later(spring_top, upper)
    st = spring.constraints.new('STRETCH_TO')
    st.target = spring_top
    st.rest_length = s_len
    st.volume = 'NO_VOLUME'
    parent_later(spring, lower)
    for o in sh_up:
        parent_later(o, upper)
    for o in sh_lo:
        parent_later(o, lower)
    for o in obs:
        if o.parent is None and not any(o is c_ for c_, _ in PARENT_JOBS):
            parent_later(o, piv)
    return piv


# ---------------------------------------------------------------------------------------------
# CONTROLS (pegs, brake pedal + rear master cylinder, shifter, kick start, side stand)
# ---------------------------------------------------------------------------------------------

def build_controls(M, root):
    col = coll('Controls', COLL['Bike'])
    obs = []
    for sx in (-1, 1):
        base = Vector((sx * 0.143, 0.15, 0.322))
        obs.append(box_obj("Peg_Bracket", base, (0.016, 0.05, 0.06), col, M['steel_rusty'], bevel=0.003))
        pc = Vector((sx * 0.205, 0.15, 0.31))
        obs.append(box_obj("Peg_Body", pc, (0.11, 0.034, 0.016), col, M['steel_rusty'], bevel=0.003))
        bm = bmesh.new()
        for i in range(7):
            for sy in (-1, 1):
                bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.0045, radius2=0.0005, depth=0.008,
                                      matrix=Matrix.Translation((-0.045 + i * 0.015, sy * 0.014, 0.012)) @ Matrix.Rotation(pi / 4, 4, 'Z'))
        obs.append(obj_from_bm("Peg_Teeth", bm, col, M['steel_rusty'], mat4(pc), smooth=False))
        obs.append(cyl_obj("Peg_PivotPin", base + Vector((sx * 0.01, -0.02, -0.012)), base + Vector((sx * 0.01, 0.02, -0.012)), 0.004, col, M['steel_bright'], segs=12))
        obs.append(cyl_obj("Peg_ReturnSpring", base + Vector((sx * 0.018, -0.012, -0.012)), base + Vector((sx * 0.018, 0.012, -0.012)), 0.0055, col, M['steel_rusty'], segs=10))
    # brake pedal (right, red): pivot boss under the peg; a short rear arm pushes a horizontal master
    # cylinder tucked under the pivot plate; remote reservoir higher up outside the plate
    pv = Vector((-0.122, 0.165, 0.27))
    obs.append(cyl_obj("Pedal_Brake_PivotBoss", Vector((-0.104, 0.165, 0.27)), pv, 0.011, col, M['steel_rusty'], segs=20))
    obs.append(cyl_obj("Pedal_Brake_Pivot", pv, pv + Vector((-0.016, 0, 0)), 0.012, col, M['steel_bright'], segs=24))
    obs.append(curve_obj("Pedal_Brake_Arm", [pv + Vector((-0.012, -0.004, -0.004)), Vector((-0.142, 0.10, 0.262)), Vector((-0.176, -0.04, 0.258)),
                                             Vector((-0.192, -0.088, 0.278))], 0.0085, col, M['paint_red'], mode='bend', bend=0.04))
    obs.append(box_obj("Pedal_Brake_Pad", Vector((-0.207, -0.088, 0.282)), (0.045, 0.03, 0.012), col, M['steel_rusty'], rot=(10, 0, 0), bevel=0.003))
    lever = pv + Vector((-0.008, 0.02, 0.03))
    obs.append(curve_obj("Pedal_Brake_Lever", [pv + Vector((-0.008, 0.0, 0.0)), lever], 0.006, col, M['paint_red']))
    obs.append(cyl_obj("Pedal_Brake_Clevis", lever + Vector((0.006, 0, 0)), lever + Vector((-0.006, 0, 0)), 0.006, col, M['steel_bright'], segs=12))
    mc_f = Vector((-0.128, 0.205, 0.303))
    mc_r = Vector((-0.128, 0.262, 0.308))
    obs.append(curve_obj("Pedal_Brake_Pushrod", [lever, mc_f], 0.0035, col, M['steel_bright']))
    obs.append(cyl_obj("Brake_Rear_PushrodBoot", mc_f + Vector((0, -0.006, 0)), mc_f + Vector((0, 0.008, 0.0)), 0.007, col, M['rubber'], r2=0.0095, segs=16))
    obs.append(cyl_obj("Brake_Rear_MasterCyl", mc_f + Vector((0, 0.006, 0.0)), mc_r, 0.011, col, M['alu_cast'], segs=24, bevel=0.002))
    obs.append(box_obj("Brake_Rear_MasterMount", Vector((-0.128, 0.235, 0.322)), (0.012, 0.03, 0.014), col, M['steel_rusty'], bevel=0.0015))
    obs.append(cyl_obj("Brake_Rear_ReservoirInlet", mc_r + Vector((0, -0.01, 0.008)), mc_r + Vector((0, -0.01, 0.016)), 0.004, col, M['alu_cast'], segs=12))
    obs.append(cyl_obj("Brake_Rear_Reservoir", Vector((-0.152, 0.27, 0.565)), Vector((-0.152, 0.27, 0.61)), 0.015, col, M['plastic_black'], segs=24, bevel=0.003))
    obs.append(box_obj("Brake_Rear_ReservoirBracket", Vector((-0.141, 0.27, 0.585)), (0.008, 0.022, 0.03), col, M['steel_rusty'], bevel=0.0015))
    obs.append(cable_obj("Hose_Rear_Reservoir", [Vector((-0.152, 0.27, 0.565)), Vector((-0.152, 0.272, 0.48)), Vector((-0.15, 0.27, 0.36)),
                                                 Vector((-0.136, 0.255, 0.33)), mc_r + Vector((0, -0.01, 0.016))], 0.0032, col, M['hose']))
    anchor('rear_master_out', mc_r + Vector((0.0, 0.006, 0.0)))
    sw_ = Vector((-0.11, 0.19, 0.27))
    obs.append(box_obj("Elec_RearBrakeSwitch", sw_, (0.008, 0.02, 0.01), col, M['plastic_black'], bevel=0.0015))
    obs.append(cable_obj("Elec_RearBrakeSwitchSpring", [sw_ + Vector((0, -0.011, 0.0)), lever + Vector((0.006, -0.002, -0.004))], 0.0016, col, M['steel_rusty']))
    anchor('rear_brake_sw', sw_ + Vector((0.0, 0.012, 0.004)))
    # gear shifter (left)
    sv = Vector((0.105, 0.035, 0.37))
    obs.append(cyl_obj("Shifter_Shaft", Vector((0.095, 0.035, 0.37)), sv + Vector((0.035, 0, 0)), 0.006, col, M['steel_bright'], segs=12))
    obs.append(box_obj("Shifter_Clamp", sv + Vector((0.03, 0, 0)), (0.012, 0.022, 0.022), col, M['steel_rusty'], bevel=0.002))
    obs.append(curve_obj("Shifter_Lever", [sv + Vector((0.034, -0.006, -0.004)), Vector((0.165, -0.015, 0.345)), Vector((0.168, -0.04, 0.335))], 0.0062, col,
                         M['steel_rusty'], mode='bend', bend=0.02))
    obs.append(cyl_obj("Shifter_Toe", Vector((0.165, -0.04, 0.335)), Vector((0.215, -0.04, 0.335)), 0.008, col, M['rubber'], segs=16, bevel=0.002))
    # kick start lever (right, folded back along the gearbox)
    k0 = Vector((-0.146, 0.09, 0.37))
    obs.append(cyl_obj("Kickstart_Shaft", Vector((-0.12, 0.09, 0.37)), k0 + Vector((-0.02, 0, 0)), 0.009, col, M['steel_bright'], segs=16))
    obs.append(box_obj("Kickstart_Boss", k0 + Vector((-0.024, 0, 0)), (0.016, 0.03, 0.03), col, M['steel_rusty'], bevel=0.003))
    obs.append(curve_obj("Kickstart_Lever", [k0 + Vector((-0.028, 0.0, 0.004)), Vector((-0.176, 0.15, 0.47)), Vector((-0.178, 0.19, 0.51))], 0.0085, col,
                         M['steel_rusty'], mode='bend', bend=0.03))
    obs.append(cyl_obj("Kickstart_PedalPivot", Vector((-0.176, 0.19, 0.51)), Vector((-0.19, 0.19, 0.51)), 0.006, col, M['steel_bright'], segs=12))
    obs.append(box_obj("Kickstart_Pedal", Vector((-0.184, 0.225, 0.516)), (0.014, 0.07, 0.014), col, M['rubber'], rot=(8, 0, 0), bevel=0.003))
    # side stand (left, retracted)
    ks = Vector((0.095, 0.10, 0.205))
    obs.append(cyl_obj("Kickstand_Pivot", ks + Vector((-0.012, 0, 0)), ks + Vector((0.02, 0, 0)), 0.012, col, M['steel_bright'], segs=20))
    obs.append(curve_obj("Kickstand_Leg", [ks + Vector((0.015, 0, 0)), Vector((0.13, 0.30, 0.225)), Vector((0.135, 0.36, 0.24))], 0.011, col,
                         M['paint_black'], mode='bend', bend=0.05))
    obs.append(box_obj("Kickstand_Foot", Vector((0.135, 0.37, 0.24)), (0.03, 0.04, 0.01), col, M['paint_black'], rot=(20, 0, 0), bevel=0.003))
    obs.append(cable_obj("Kickstand_Spring", [ks + Vector((0.02, 0.02, -0.005)), Vector((0.10, 0.18, 0.212))], 0.003, col, M['steel_rusty']))
    return obs


# ---------------------------------------------------------------------------------------------
# CABLES: Catmull-Rom splines -> exact Bezier curves, hooks for flex, ties along the run
# ---------------------------------------------------------------------------------------------

HOOK_JOBS = []       # (curve object, [point indices], target object name)
SPLINES = {}


class Spline:
    def __init__(self, pts):
        self.p = [Vector(q) for q in pts]
        n = len(self.p)
        self.hl, self.hr = [], []
        for i in range(n):
            a = self.p[max(0, i - 1)]
            b = self.p[min(n - 1, i + 1)]
            m = (b - a) / (2.0 if 0 < i < n - 1 else 1.0)
            self.hl.append(self.p[i] - m / 3)
            self.hr.append(self.p[i] + m / 3)
        self.samples = []
        for i in range(n - 1):
            for k in range(24):
                self.samples.append(self.eval(i, k / 24))
        self.samples.append(self.p[-1])
        self.cum = [0.0]
        for a, b in zip(self.samples, self.samples[1:]):
            self.cum.append(self.cum[-1] + (b - a).length)
        self.length = self.cum[-1]

    def eval(self, i, t):
        u = 1 - t
        return self.p[i] * u ** 3 + self.hr[i] * 3 * u * u * t + self.hl[i + 1] * 3 * u * t * t + self.p[i + 1] * t ** 3

    def at(self, s):
        s = max(0.0, min(self.length, s))
        lo, hi = 0, len(self.cum) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.cum[mid] < s:
                lo = mid
            else:
                hi = mid
        seg = self.cum[hi] - self.cum[lo]
        f = 0.0 if seg == 0 else (s - self.cum[lo]) / seg
        return self.samples[lo].lerp(self.samples[hi], f), (self.samples[hi] - self.samples[lo]).normalized()


def cable_obj(name, pts, radius, col, mat, res=12, bevel_res=3, caps=True):
    sp = Spline(pts)
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.resolution_u = res
    cu.bevel_mode = 'ROUND'
    cu.bevel_depth = radius
    cu.bevel_resolution = bevel_res
    cu.use_fill_caps = caps
    cu.twist_mode = 'MINIMUM'
    s = cu.splines.new('BEZIER')
    s.bezier_points.add(len(sp.p) - 1)
    for bp, co, hl, hr in zip(s.bezier_points, sp.p, sp.hl, sp.hr):
        bp.handle_left_type = bp.handle_right_type = 'FREE'
        bp.co, bp.handle_left, bp.handle_right = co, hl, hr
    ob = bpy.data.objects.new(name, cu)
    col.objects.link(ob)
    if mat:
        cu.materials.append(mat)
    SPLINES[ob.name] = sp
    return ob


TIE_MESH = {}


def tie_mesh(r, M):
    key = round(r, 4)
    if key not in TIE_MESH:
        bm = bmesh.new()
        prof = [(-0.0018, r + 0.0002), (-0.0018, r + 0.0013), (0.0018, r + 0.0013), (0.0018, r + 0.0002)]
        rb = revolve_bm(prof, 16, closed=True, axis='Z')
        me = bpy.data.meshes.new("HW_CableTie_%dmm" % int(round(r * 2000)))
        bm2 = rb
        bm_box(bm2, (0.0045, 0.004, 0.0045), Matrix.Translation((0.0, r + 0.0025, 0.0)))
        bm2.to_mesh(me)
        bm2.free()
        bm.free()
        me.materials.append(M['plastic_black'])
        finish_mesh(me, True, 40)
        TIE_MESH[key] = me
    return TIE_MESH[key]


def add_ties(cable, r, col, M, fractions, parent=None, name="HW_CableTie"):
    sp = SPLINES[cable.name]
    out = []
    for f in fractions:
        p, t = sp.at(sp.length * f)
        ob = bpy.data.objects.new(name, tie_mesh(r, M))
        col.objects.link(ob)
        ob.matrix_world = mat4(p, frame_from_z(t) @ Matrix.Rotation(random.uniform(0, 2 * pi), 3, 'Z'))
        out.append(ob)
        if parent is not None:
            parent_later(ob, parent)
    return out


def hook_points(cable, idxs, target_name):
    HOOK_JOBS.append((cable, list(idxs), target_name))


def apply_hooks():
    bpy.context.view_layer.update()
    for cu_ob, idxs, tname in HOOK_JOBS:
        tgt = bpy.data.objects[tname]
        m = cu_ob.modifiers.new("Flex_" + tname, 'HOOK')
        m.object = tgt
        vi = []
        for i in idxs:
            vi += [3 * i, 3 * i + 1, 3 * i + 2]
        m.vertex_indices_set(vi)
        m.matrix_inverse = (cu_ob.matrix_world.inverted() @ tgt.matrix_world).inverted()
        m.use_apply_on_spline = True
    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------------------------
# ELECTRICS + HARNESS: battery -> fuse box -> main loom along the backbone -> head stock ->
# headlight shell (junction) -> switchgear/gauges; rear loom to the tail; engine sensors;
# control cables and brake lines with hooks so they flex with steering and suspension.
# ---------------------------------------------------------------------------------------------

def build_electrics(M, root):
    col = coll('Electrics', COLL['Bike'])
    obs = []
    # battery in a steel holder between the seat rails (ahead of the rear tyre's bump envelope)
    bc = Vector((0.0, 0.414, 0.77))
    obs.append(box_obj("Elec_Battery", bc, (0.118, 0.058, 0.098), col, M['battery_case'], bevel=0.003))
    obs.append(box_obj("Elec_BatteryLid", bc + Vector((0, 0, 0.051)), (0.12, 0.06, 0.006), col, M['plastic_black'], bevel=0.0015))
    obs.append(box_obj("Elec_BatteryHolder", bc + Vector((0, 0, -0.044)), (0.126, 0.066, 0.012), col, M['paint_black'], bevel=0.0015))
    for sx in (-1, 1):
        obs.append(box_obj("Elec_BatteryHolderStrap", Vector((sx * 0.066, bc.y, bc.z + 0.012)), (0.004, 0.02, 0.12), col, M['paint_black'], bevel=0.0008))
    obs.append(box_obj("Elec_BatteryStrap", bc + Vector((0, 0, 0.057)), (0.135, 0.018, 0.004), col, M['rubber'], bevel=0.001))
    for nm, tx, mat in (("Pos", -0.012, 'wire_red'), ("Neg", -0.042, 'wire_black')):
        t = bc + Vector((tx, -0.016, 0.054))
        obs.append(cyl_obj("Elec_BatteryTerminal_" + nm, t, t + Vector((0, 0, 0.008)), 0.0055, col, M['brass'], segs=12))
        obs.append(cyl_obj("Elec_TerminalBoot_" + nm, t + Vector((0, 0, 0.004)), t + Vector((0, 0.0, 0.016)), 0.0085, col, M[mat if nm == 'Pos' else 'rubber'], r2=0.006, segs=16))
        anchor('batt_' + nm.lower(), t + Vector((0, 0.004, 0.017)))
    # components on the tray behind the battery
    fb = Vector((0.04, 0.475, 0.812))
    obs.append(box_obj("Elec_FuseBox", fb, (0.042, 0.032, 0.026), col, M['plastic_black'], bevel=0.003))
    obs.append(box_obj("Elec_FuseBoxLid", fb + Vector((0, 0, 0.0145)), (0.044, 0.034, 0.004), col, M['plastic_black'], bevel=0.001))
    anchor('fusebox', fb + Vector((0.0, -0.018, 0.0)))
    cdi = Vector((-0.04, 0.48, 0.81))
    obs.append(box_obj("Elec_CDI", cdi, (0.05, 0.04, 0.022), col, M['dark_metal'], bevel=0.003))
    obs.append(box_obj("Elec_CDIConnector", cdi + Vector((0.0, -0.023, 0.0)), (0.03, 0.008, 0.014), col, M['plastic_black'], bevel=0.001))
    anchor('cdi', cdi + Vector((0.0, -0.03, 0.0)))
    rr = Vector((0.0, 0.53, 0.812))
    obs.append(box_obj("Elec_RegRect", rr, (0.06, 0.042, 0.022), col, M['alu_fins'], bevel=0.002))
    bm = bmesh.new()
    bm_box(bm, (0.058, 0.0015, 0.008))
    fin = obj_from_bm("Elec_RegRectFins", bm, col, M['alu_fins'], mat4(rr + Vector((0, -0.018, 0.014))))
    a = fin.modifiers.new("FinArray", 'ARRAY')
    a.count = 8
    a.use_relative_offset = False
    a.use_constant_offset = True
    a.constant_offset_displace = (0, 0.005, 0)
    obs.append(fin)
    anchor('regrect', rr + Vector((0.0, -0.024, 0.0)))
    obs.append(cyl_obj("Elec_FlasherRelay", Vector((-0.05, 0.535, 0.80)), Vector((-0.05, 0.535, 0.828)), 0.011, col, M['plastic_black'], segs=20, bevel=0.002))
    anchor('flasher', Vector((-0.05, 0.522, 0.81)))
    for p in (fb, cdi, rr):
        obs.append(bolt(p + Vector((0, 0, -0.012)), (0, 0, 1), col, 'cap', 0.5, "Elec_MountScrew"))
    # ground lug on the shock cross tube
    obs.append(box_obj("Elec_GroundLug", Vector((-0.03, 0.345, 0.848)), (0.012, 0.016, 0.003), col, M['brass'], bevel=0.0006))
    obs.append(bolt(Vector((-0.03, 0.345, 0.8495)), (0, 0, 1), col, 'bolt', 0.6, "Elec_GroundBolt"))
    anchor('ground', Vector((-0.03, 0.353, 0.849)))
    return obs


def build_harness(M, root, groups):
    col = coll('Harness', COLL['Bike'])
    parent_of = {}

    def run(name, pts, r, mat, group='static', hooks=None, ties=None, tie_group=None):
        ob = cable_obj(name, pts, r, col, M[mat], res=max(6, min(16, int(len(pts) * 3))))
        parent_of[ob.name] = group
        for g, idxs in (hooks or {}).items():
            hook_points(ob, idxs, groups[g].name)
        if ties:
            add_ties(ob, r, col, M, ties, parent=groups[tie_group or group])
        return ob

    def a(name, off=(0, 0, 0)):
        return A[name][0] + Vector(off)
    bb = FRAME['backbone']
    # ---- main loom: fuse box -> along the left seat rail -> under the tank along the backbone -> head stock ----
    loom = [a('fusebox'), Vector((0.045, 0.44, 0.838)), Vector((0.045, 0.345, 0.836)), Vector((0.045, 0.22, 0.832)), Vector((0.04, 0.12, 0.83)),
            Vector((0.026, 0.02, 0.858)), Vector((0.024, -0.12, 0.885)), Vector((0.022, -0.26, 0.922)), Vector((0.02, -0.36, 0.945)),
            Vector((0.045, -0.44, 0.982)), Vector((0.048, -0.50, 0.982)), Vector((0.035, -0.545, 0.98)), a('headlight_grommet_low', (0, 0.004, -0.004))]
    L = run("Loom_Main", loom, 0.0068, 'loom', hooks={'steer': [11, 12]}, ties=(0.06, 0.15, 0.25, 0.34, 0.43, 0.52, 0.61, 0.70), tie_group='static')
    # ---- rear loom: fuse box -> left rail -> tail light / rear indicators ----
    rear = [a('fusebox', (0.004, 0.018, 0.004)), Vector((0.05, 0.56, 0.848)), Vector((0.05, 0.70, 0.85)), Vector((0.03, 0.80, 0.846)), a('tail_light')]
    run("Loom_Rear", rear, 0.0045, 'loom', ties=(0.25, 0.5, 0.72))
    for sx, k in ((1, 'L'), (-1, 'R')):
        run("Wire_RearIndicator_" + k, [a('tail_light', (0, -0.02, -0.004)), Vector((sx * 0.03, 0.84, 0.858)), a('ind_rear_' + k)], 0.0022, 'wire_black')
    # ---- battery, power and charging ----
    run("Wire_BatteryPos", [a('batt_pos'), a('batt_pos', (0.004, 0.02, 0.004)), Vector((0.012, 0.44, 0.842)), Vector((0.045, 0.457, 0.818)), a('fusebox', (0.01, 0.002, -0.004))], 0.0042, 'wire_red')
    run("Wire_BatteryNeg", [a('batt_neg'), a('batt_neg', (0.0, -0.022, 0.004)), Vector((-0.034, 0.368, 0.848)), a('ground')], 0.0042, 'wire_black')
    run("Wire_RegRect", [a('regrect', (0.022, 0.0, 0.0)), Vector((0.032, 0.5, 0.818)), a('fusebox', (0.0, 0.034, 0.0))], 0.003, 'wire_yellow')
    run("Wire_CDI", [a('cdi'), Vector((-0.04, 0.456, 0.842)), Vector((0.02, 0.462, 0.842)), a('fusebox', (-0.01, 0.002, 0.012))], 0.0035, 'wire_black')
    run("Wire_Flasher", [a('flasher', (-0.008, 0, 0)), Vector((-0.064, 0.52, 0.826)), Vector((-0.064, 0.455, 0.83)), a('fusebox', (-0.01, 0.0, 0.0))], 0.0022, 'wire_orange')
    # alternator -> down the left side of the gearbox -> regulator (3 yellow wires in one sleeve)
    alt = [a('alt_out'), Vector((0.113, -0.045, 0.53)), Vector((0.111, 0.04, 0.536)), Vector((0.111, 0.12, 0.546)), Vector((0.111, 0.18, 0.60)),
           Vector((0.111, 0.23, 0.636)), Vector((0.113, 0.29, 0.662)), Vector((0.113, 0.305, 0.70)), Vector((0.10, 0.36, 0.80)), Vector((0.085, 0.40, 0.825)), Vector((0.085, 0.47, 0.832)),
           Vector((0.05, 0.522, 0.842)), a('regrect', (0.034, 0.0, 0.002))]
    run("Loom_Alternator", alt, 0.0045, 'loom', ties=(0.25, 0.42, 0.58))
    run("Wire_NeutralSwitch", [a('neutral_sw'), Vector((0.11, 0.163, 0.46)), Vector((0.11, 0.165, 0.56)), Vector((0.111, 0.17, 0.635))], 0.0018, 'wire_green')
    # ignition: CDI -> coil (low tension) and HT leads to the plugs (left side), clipped to the coil bracket
    run("Wire_CoilLT", [Vector((0.032, 0.0, 0.875)), Vector((0.05, -0.08, 0.888)), a('coil_lv')], 0.0022, 'wire_black')
    for k, plug in ((0, 'plug_F'), (1, 'plug_R')):
        t0 = a('coil_tower_%d' % k)
        p1 = a(plug)
        mid = (t0 + p1) / 2 + Vector((0.05, 0.0, 0.0))
        run("Wire_HTLead_%d" % k, [t0, t0 + Vector((0.01, 0.0, -0.02)), mid, p1 + Vector((0.012, 0, 0.012)), p1], 0.0042, 'wire_red')
    run("Wire_OilPressure", [a('oilp_sw'), Vector((-0.112, -0.24, 0.55)), Vector((-0.07, -0.34, 0.70)), Vector((-0.03, -0.39, 0.86)),
                             Vector((0.01, -0.40, 0.93)), Vector((0.02, -0.40, 0.955))], 0.0018, 'wire_blue')
    run("Wire_Horn", [a('horn'), a('horn', (-0.01, 0.02, 0.06)), Vector((0.04, -0.45, 0.95)), Vector((0.032, -0.46, 0.982))], 0.002, 'wire_green')
    run("Wire_RearBrakeSwitch", [a('rear_brake_sw'), Vector((-0.111, 0.202, 0.30)), Vector((-0.111, 0.2, 0.36)), Vector((-0.111, 0.198, 0.43)), Vector((-0.11, 0.205, 0.50)), Vector((-0.105, 0.24, 0.66)),
                                 Vector((-0.06, 0.27, 0.72)), Vector((-0.055, 0.30, 0.78)), Vector((-0.05, 0.35, 0.815)), Vector((0.0, 0.378, 0.822)),
                                 Vector((0.045, 0.38, 0.836)), Vector((0.045, 0.40, 0.838))],
        0.0018, 'wire_brown')
    # ---- handlebar harness (all steer group): switches -> along the bar -> into the headlight shell ----
    gt = a('headlight_grommet_top')
    for k in ('L', 'R'):
        sx = 1 if k == 'L' else -1
        e = a('switch_exit_' + k)
        T0_ = A['bar_center'][0]
        pts = [e, e + Vector((-sx * 0.02, 0.002, -0.006)), Vector((sx * 0.20, T0_.y + 0.022, T0_.z + 0.0)), Vector((sx * 0.13, T0_.y + 0.005, T0_.z - 0.02)),
               Vector((sx * 0.06, T0_.y + 0.009, T0_.z - 0.019)), Vector((sx * 0.046, T0_.y - 0.07, T0_.z - 0.004)), Vector((sx * 0.046, -0.553, T0_.z - 0.008)),
               Vector((sx * 0.022, gt.y + 0.018, gt.z + 0.026)), gt + Vector((sx * 0.004, 0.004, 0.004))]
        run("Loom_Switch_" + k, pts, 0.004, 'loom', group='steer', ties=(0.25, 0.4, 0.52), tie_group='steer')
    bs = a('brake_switch_F')
    run("Wire_FrontBrakeSwitch", [bs, bs + Vector((0.02, -0.01, -0.02)), a('switch_exit_R', (0.01, 0, -0.012))], 0.0016, 'wire_white', group='steer')
    gc = a('gauge_conn')
    run("Loom_Gauge", [gc, gc + Vector((0.0, -0.01, -0.03)), Vector((0.0, gt.y - 0.0, gt.z + 0.07)), gt + Vector((0.0, 0.004, 0.006))], 0.0035, 'loom', group='steer')
    ib = a('ignition_back')
    run("Wire_Ignition", [ib, ib + Vector((0.0, -0.01, -0.03)), Vector((0.05, gt.y + 0.01, gt.z + 0.01)), gt + Vector((0.008, 0.004, 0.002))], 0.0025, 'wire_red', group='steer')
    hc = a('headlight_c')
    for k in ('L', 'R'):
        sx = 1 if k == 'L' else -1
        i0 = a('ind_front_' + k)
        ear_mid = (A['ear_a_' + k][0] + A['ear_b_' + k][0]) / 2 + Vector((sx * 0.012, 0, 0.012))
        run("Wire_FrontIndicator_" + k, [i0, i0 + Vector((0.0, -0.03, -0.004)), ear_mid + Vector((0, 0, -0.017)), hc + Vector((sx * 0.098, 0.014, -0.018))],
            0.0018, 'wire_black', group='steer')
    # ---- control cables ----
    t0 = a('throttle_out_0')
    t1 = a('throttle_out_1')
    for k, (tp, carb) in enumerate(((t0, 'carb_throttle_0'), (t1, 'carb_throttle_1'))):
        dz = 0.006 * k
        o = Vector((0.0125 * k, 0.0, 0.005 * k))
        pts = [tp, tp + Vector((0.0, -0.03, -0.05)), Vector((-0.17, tp.y - 0.05, tp.z - 0.15)) + o, Vector((-0.08, -0.46, 0.99)) + o,
               Vector((-0.06, -0.40, 0.93)) + o, Vector((-0.062, -0.26, 0.865)) + o, Vector((-0.048 + 0.016 * k, -0.10, 0.815)), a(carb, (0, 0, 0.03)), a(carb)]
        run("Cable_Throttle_%d" % k, pts, 0.0033, 'wire_black', hooks={'steer': [0, 1, 2]}, ties=(0.62,), tie_group='static')
    c0 = a('clutch_out')
    tpm = A['throttle_out_0'][0].copy()
    tpm.x = -tpm.x
    pts = [c0, c0 + Vector((0.0, -0.03, -0.05)), Vector((0.17, tpm.y - 0.05, tpm.z - 0.15)), Vector((0.08, -0.46, 0.99)), Vector((0.076, -0.42, 0.93)),
           Vector((0.10, -0.37, 0.80)), Vector((0.135, -0.22, 0.728)), Vector((0.135, -0.06, 0.63)), a('clutch_arm', (0, 0, 0.03)), a('clutch_arm')]
    run("Cable_Clutch", pts, 0.0042, 'wire_black', hooks={'steer': [0, 1, 2]}, ties=(0.55, 0.7), tie_group='static')
    run("Cable_ClutchInner", [a('clutch_arm'), a('clutch_arm_end')], 0.0012, 'steel_bright')
    # front brake: master (steer) -> down in front of the yokes -> hose guide (travel) -> caliper banjo (travel)
    bj = a('brake_banjo_F')
    hg = a('hose_guide_F')
    cb = a('caliper_banjo_F')
    F_ = ForkFrame()
    fx = P['fork_half_span']
    yb, yt = P['yokes']
    pts = [bj, bj + Vector((0.03, -0.012, -0.008)), Vector((-0.135, bj.y - 0.09, bj.z + 0.004)), F_.p(0.93, -0.08, -fx + 0.012), F_.p(yt + 0.05, -0.085, -fx), F_.p(yt - 0.05, -0.07, -fx - 0.004), F_.p(yb + 0.04, -0.068, -fx - 0.006),
           F_.p(yb - 0.08, -0.04, -(fx + 0.048)), hg, hg + Vector((0.004, 0.022, -0.06)), cb + Vector((-0.004, 0.012, 0.02)), cb]
    run("Hose_Front_Brake", pts, 0.0048, 'braided', group='steer', hooks={'travel': [8, 9, 10, 11]})
    # speed sensor wire: sensor (travel) -> up the left slider -> into the headlight (steer)
    ss = a('speed_sensor')
    F_ = ForkFrame()
    fx = P['fork_half_span']
    gb = a('headlight_grommet_bottom')
    pts = [ss, ss + Vector((0.006, -0.012, 0.03)), F_.p(0.10, -0.047, fx - 0.03), F_.p(0.16, -0.052, fx + 0.046), F_.p(0.30, -0.053, fx + 0.046),
           F_.p(0.46, -0.054, fx + 0.04), F_.p(yb + 0.02, -0.07, fx - 0.03), Vector((0.05, gb.y - 0.03, gb.z - 0.03)), gb]
    run("Wire_SpeedSensor", pts, 0.0022, 'wire_black', group='steer', hooks={'travel': [0, 1, 2, 3, 4, 5]})
    # rear brake hose: master (static) -> along the right arm (swing) -> caliper banjo
    rm = a('rear_master_out')
    cr = a('caliper_banjo_R')
    ax_ = P['arm_x']
    sp = P['swing_pivot']
    RA = P['rear_axle']

    def armz(y):
        return lerp(sp.z, RA.z, (y - sp.y) / (RA.y - sp.y))
    pts = [rm, rm + Vector((-0.012, 0.016, 0.004)), Vector((-0.15, 0.338, 0.368)), Vector((-0.16, 0.38, 0.447)), Vector((-0.163, 0.46, armz(0.46) + 0.05)),
           Vector((-0.163, 0.56, armz(0.56) + 0.045)), Vector((-0.135, 0.622, 0.39)), cr + Vector((-0.004, 0.0, 0.02)), cr]
    run("Hose_Rear_Brake", pts, 0.0045, 'braided', hooks={'swing': [2, 3, 4, 5, 6, 7, 8]})
    # fuel line: petcock (left) -> over the top of the V -> carb inlet
    pc = a('petcock_out')
    fi = a('carb_fuel_in')
    run("Hose_Fuel", [pc, pc + Vector((0.0, 0.0, -0.03)), Vector((0.03, -0.02, 0.78)), Vector((0.0, -0.012, 0.76)), fi + Vector((0.015, 0, 0.0)), fi],
        0.0045, 'hose')
    return parent_of


# ---------------------------------------------------------------------------------------------
# RIG: custom properties on Bike_Root drive the pivots
# ---------------------------------------------------------------------------------------------

def add_prop(ob, name, val, lo, hi, desc, subtype='NONE'):
    ob[name] = val
    ui = ob.id_properties_ui(name)
    ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=desc, subtype=subtype)


def drive(target, path, index, root, prop, expr):
    fc = target.driver_add(path, index)
    d = fc.driver
    d.type = 'SCRIPTED'
    v = d.variables.new()
    v.name = 'v'
    v.type = 'SINGLE_PROP'
    v.targets[0].id_type = 'OBJECT'
    v.targets[0].id = root
    v.targets[0].data_path = '["%s"]' % prop
    d.expression = expr
    return fc


def build_rig(root):
    lock = P['steer_lock']
    add_prop(root, "steer", 0.0, -lock, lock, "Handlebar steering angle (degrees), limited by the steering stops")
    add_prop(root, "fork_travel", 0.0, 0.0, P['fork_travel'], "Front fork compression (m)")
    add_prop(root, "swingarm_angle", 0.0, P['swing_droop'], P['swing_bump'], "Swingarm rotation (degrees, + = bump)")
    add_prop(root, "distance", 0.0, -1000.0, 1000.0, "Distance rolled forward (m) - spins both wheels")
    add_prop(root, "speedo_kmh", 0.0, 0.0, 180.0, "Speedometer needle (km/h)")
    add_prop(root, "tacho_rpm", 0.0, 0.0, 10000.0, "Tachometer needle (rpm)")
    ob = bpy.data.objects
    drive(ob["FrontEnd_Steer"], 'rotation_euler', 2, root, "steer", "v*pi/180")
    tr = ob["FrontEnd_ForkTravel"]
    drive(tr, 'location', 2, root, "fork_travel", "%.6f+v" % tr.location.z)
    drive(ob["RearEnd_SwingarmPivot"], 'rotation_euler', 0, root, "swingarm_angle", "v*pi/180")
    drive(ob["FrontWheel_Pivot"], 'rotation_euler', 0, root, "distance", "v/%.4f" % P['front_tire']['R'])
    drive(ob["RearWheel_Pivot"], 'rotation_euler', 0, root, "distance", "v/%.4f" % P['rear_tire']['R'])
    # needles sweep 270 deg clockwise from the 0 mark (225 deg)
    drive(ob["Gauge_Speedo_Needle"], 'rotation_euler', 2, root, "speedo_kmh", "(225-v*1.5)*pi/180")
    drive(ob["Gauge_Tacho_Needle"], 'rotation_euler', 2, root, "tacho_rpm", "(225-v*0.027)*pi/180")


# ---------------------------------------------------------------------------------------------
# STUDIO (not part of the asset)
# ---------------------------------------------------------------------------------------------

def build_studio():
    col = coll('Studio')
    # cyclorama
    bm = bmesh.new()
    rows = []
    prof = [(-8.0, 0.0), (3.2, 0.0)]
    for i in range(1, 13):
        a = (pi / 2) * i / 12
        prof.append((3.2 + 1.6 * sin(a), 1.6 * (1 - cos(a))))
    prof.append((4.8, 7.0))
    for x in (-10.0, 10.0):
        rows.append([bm.verts.new((x, y, z)) for y, z in prof])
    for j in range(len(prof) - 1):
        bm.faces.new((rows[0][j], rows[1][j], rows[1][j + 1], rows[0][j + 1]))
    bm.normal_update()
    m = mat_simple("M_Studio_Backdrop", (0.85, 0.85, 0.85), rough=0.9)
    cyc = obj_from_bm("Studio_Cyclorama", bm, col, m, mat4(Vector((0, 0, 0)), Matrix.Rotation(rad(-40), 3, 'Z')), angle=89)
    for v in cyc.data.polygons:
        v.use_smooth = True
    # lights
    def area(name, loc, target, power, size, color=(1, 1, 1), shape='RECTANGLE', size_y=None):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = power
        ld.shape = shape
        ld.size = size
        if size_y:
            ld.size_y = size_y
        ld.color = color
        o = bpy.data.objects.new(name, ld)
        col.objects.link(o)
        o.location = loc
        d = (Vector(target) - Vector(loc))
        o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return o
    area("Studio_Key", (-2.8, -2.2, 3.0), (0, 0, 0.5), 380, 2.5, (1.0, 0.97, 0.93))
    area("Studio_Fill", (-3.2, 1.2, 1.4), (0, 0, 0.5), 110, 3.0, (0.93, 0.96, 1.0))
    area("Studio_Top", (0.0, 0.0, 3.6), (0, 0, 0.0), 260, 3.5, size_y=2.0, shape='RECTANGLE')
    area("Studio_Rim", (1.6, 2.6, 2.0), (0, 0, 0.6), 220, 1.5)
    w = bpy.data.worlds.new("Studio_World")
    w.use_nodes = True
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.9, 0.9, 0.9, 1)
    bg.inputs['Strength'].default_value = 0.35
    bpy.context.scene.world = w
    # cameras
    def cam(name, loc, target, lens):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.sensor_width = 36
        o = bpy.data.objects.new(name, cd)
        col.objects.link(o)
        o.location = loc
        o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return o
    hero = cam("Cam_Hero", (-2.55, -3.05, 1.18), (0.0, -0.06, 0.55), 55)
    cam("Cam_SideRight", (-4.2, 0.0, 0.62), (0, 0.0, 0.6), 50)
    cam("Cam_SideLeft", (4.2, 0.0, 0.62), (0, 0.0, 0.6), 50)
    cam("Cam_FrontThreeQuarterLeft", (2.5, -3.0, 1.25), (0.0, -0.06, 0.55), 55)
    cam("Cam_RearThreeQuarter", (-2.4, 3.0, 1.35), (0.0, 0.1, 0.55), 55)
    cam("Cam_Front", (0.0, -4.2, 0.75), (0, 0, 0.62), 60)
    cam("Cam_Top", (0.0, 0.0, 4.5), (0, 0, 0.0), 50)
    cam("Cam_EngineDetail", (-1.05, -0.7, 0.75), (-0.05, -0.05, 0.52), 45)
    bpy.context.scene.camera = hero


def render_settings(samples=96, res=(1400, 1260)):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = 'OPENIMAGEDENOISE'
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 8
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Punchy'
    sc.view_settings.exposure = -0.25
    sc.render.image_settings.file_format = 'PNG'


# ---------------------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------------------

def build(out_path=None):
    reset_scene()
    COLL.clear()
    PARENT_JOBS.clear()
    MAT_CACHE.clear()
    HW.clear()
    bike = coll('Bike')
    COLL['Bike'] = bike
    cut = coll('Cutters', bike)
    COLL['Cutters'] = cut
    M = build_materials()
    hardware_meshes(M)
    root = make_empty("Bike_Root", Matrix.Identity(4), bike, 'ARROWS', 0.5)
    root["asset_name"] = "Wasteland Scrambler"
    A.clear()
    HOOK_JOBS.clear()
    SPLINES.clear()
    TIE_MESH.clear()
    FRAME.clear()
    F, steer, travel = build_front(M, root)
    build_cockpit(M, F, steer)
    frame = build_frame(M, F, root)
    body, tank, seat, tail = build_body(M, root)
    eng, fport, rport = build_engine(M, root)
    exh = build_exhaust(M, root, fport, rport)
    swing = build_rear(M, root)
    ctrl = build_controls(M, root)
    build_electrics(M, root)
    groups = dict(static=root, steer=steer, travel=travel, swing=swing)
    harness_parent = build_harness(M, root, groups)
    for name, g in harness_parent.items():
        parent_later(bpy.data.objects[name], groups[g])
    # everything without a parent hangs off the root
    for o in bike.all_objects:
        if o is root or o.parent is not None:
            continue
        if any(o is c for c, _ in PARENT_JOBS):
            continue
        parent_later(o, root)
    resolve_parents()
    apply_hooks()
    build_rig(root)
    # cutters: hidden in viewport too
    for o in cut.all_objects:
        o.hide_set(True)
    build_studio()
    render_settings()
    bpy.context.view_layer.update()
    if out_path:
        bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
    return root


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = None
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    build(out)
    stats = dict(objects=len(bpy.data.objects), meshes=len(bpy.data.meshes), materials=len(bpy.data.materials))
    print("BUILD_OK", stats)
