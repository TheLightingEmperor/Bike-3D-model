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
import random
import sys
from math import pi, sin, cos, radians as rad
from mathutils import Vector, Matrix, Euler

random.seed(11)

# ---------------------------------------------------------------------------------------------
# Global layout parameters (edit these to re-proportion the bike, then re-run)
# ---------------------------------------------------------------------------------------------
P = dict(
    front_axle=Vector((0.0, -0.74, 0.37)),
    rear_axle=Vector((0.0, 0.72, 0.375)),
    rake=25.0,                 # steering head angle from vertical (deg)
    fork_offset=0.05,          # steering axis behind fork tubes
    fork_half_span=0.14,       # fork tube lateral offset
    front_tire=dict(R=0.37, half_w=0.108, R_rim=0.215, lug_h=0.024, lugs=21),
    rear_tire=dict(R=0.375, half_w=0.122, R_rim=0.215, lug_h=0.026, lugs=23),
    swing_pivot=Vector((0.0, 0.22, 0.43)),
    yellow=(0.80, 0.43, 0.006),
)

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
              primer=(0.20, 0.19, 0.17), metal=(0.22, 0.21, 0.2), scratches=0.6, streak=0.5, dust=0.5, smudge=0.6):
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


def build_wheel(prefix, axle, spec, col, M, rotor=None, sprocket=None, axle_half=0.16):
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
    if sprocket:
        build_sprocket(prefix + "_Sprocket", axle + Vector((sprocket['a'], 0, 0)), sprocket['N'], sprocket['r_p'],
                       0.006, col, M)
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
# FRONT END (steering, forks, headlight, cockpit)
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


def build_front(M, root):
    col = coll('FrontEnd', COLL['Bike'])
    wcol = coll('Wheels', COLL['Bike'])
    F = ForkFrame()
    off, fx = P['fork_offset'], P['fork_half_span']
    Rf = F.R
    # --- rig empties ---
    ax = make_empty("FrontEnd_SteeringAxis", F.M(0.68, off), col, 'SINGLE_ARROW', 0.35)
    steer = make_empty("FrontEnd_Steer", F.M(0.68, off), col, 'CIRCLE', 0.12)
    travel = make_empty("FrontEnd_ForkTravel", F.M(0.0), col, 'SINGLE_ARROW', 0.15)
    parent_later(ax, root)
    parent_later(steer, ax)
    parent_later(travel, steer)
    S, T = [], []          # objects parented to steer / travel

    # --- stanchions (upper tubes), mirrored ---
    bm = bmesh.new()
    bm_cyl(bm, 0.0265, 0.0265, 0.58, 40, Matrix.Translation((-fx, 0, 0.0)))
    stan = obj_from_bm("Fork_Stanchions", bm, col, M['chrome'], F.M(0.30), angle=50)
    mod_mirror(stan)
    S.append(stan)
    # fork caps + preload adjusters
    prof = [(0.0, 0.0), (0.0, 0.0285), (0.004, 0.0295), (0.022, 0.0295), (0.026, 0.024), (0.03, 0.012), (0.042, 0.012),
            (0.043, 0.006), (0.043, 0.0)]
    cap = revolve_obj("Fork_TopCaps", [(a, r) for a, r in prof], col, M['steel_rusty'],
                      F.M(0.88, 0, 0) @ Matrix.Translation((-fx, 0, 0)), segs=6, closed=False, axis='Z', angle=20)
    cap.matrix_world = F.M(0.875)
    for v in cap.data.vertices:
        v.co.x -= fx
    mod_mirror(cap)
    S.append(cap)

    # --- yokes (triple clamps) ---
    def yoke(name, t, thick, extra_pts=()):
        pts = []
        for cx in (-fx, fx):
            pts += circle_pts(0.046, 24, cx, 0.0)
        pts += circle_pts(0.04, 24, 0.0, off)
        pts += list(extra_pts)
        hull = convex_hull([(round(x, 6), round(y, 6)) for x, y in pts])
        ob = extrude_obj(name, hull, thick, col, M['alu_cast'], F.M(t), bevel=0.003)
        # tube bores
        bm = bmesh.new()
        for cx in (-fx, fx):
            bm_cyl(bm, 0.0268, 0.0268, thick * 3, 32, Matrix.Translation((cx, 0, -thick * 1.5)))
        bm_cyl(bm, 0.018, 0.018, thick * 3, 24, Matrix.Translation((0, off, -thick * 1.5)))
        cut = obj_from_bm(name + "_BoreCutter", bm, COLL['Cutters'], None, F.M(t), smooth=False)
        mod_boolean(ob, cut, "Bores")
        ob.modifiers.move(len(ob.modifiers) - 1, 0)
        # pinch bolts (from the back)
        for cx in (-fx, fx):
            for dz in (-thick * 0.22, thick * 0.22):
                bolt(F.p(t + dz, 0.05, cx * 0.82), F.nb, col, 'cap', 1.0, name + "_PinchBolt")
        return ob
    S.append(yoke("Fork_BottomYoke", 0.56, 0.034))
    S.append(yoke("Fork_TopYoke", 0.80, 0.026, circle_pts(0.03, 12, 0.045, off + 0.035) + circle_pts(0.03, 12, -0.045, off + 0.035)))
    S.append(cyl_obj("Fork_SteeringStemNut", F.p(0.813, off), F.p(0.83, off), 0.02, col, M['steel_rusty'], segs=6, bevel=0.001))
    S.append(cyl_obj("Fork_SteeringStem", F.p(0.54, off), F.p(0.82, off), 0.017, col, M['steel_rusty'], segs=24))

    # --- sliders (lower legs, yellow), axle lugs, dust seals: ride on the fork travel ---
    prof = [(-0.045, 0.0), (-0.045, 0.026), (-0.03, 0.034), (0.25, 0.036), (0.36, 0.038), (0.372, 0.0395), (0.385, 0.0395),
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
    # axle lugs + pinch bolts
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
    # fork brace (yellow arch plate on the front of the sliders)
    arch = [(-0.17, 0.26), (-0.17, 0.36), (-0.11, 0.445), (0.11, 0.445), (0.17, 0.36), (0.17, 0.26),
            (0.115, 0.26), (0.115, 0.35), (0.075, 0.405), (-0.075, 0.405), (-0.115, 0.35), (-0.115, 0.26)]
    br = extrude_obj("Fork_Brace", arch, 0.012, col, M['paint_yellow'], F.M(0.0, 0, 0, Matrix.Rotation(pi / 2, 3, 'X')) @
                     Matrix.Translation((0, 0, 0.044)), bevel=0.003)
    T.append(br)
    for sx in (-1, 1):
        for z in (0.28, 0.335):
            T.append(bolt(F.p(z, -0.05, sx * 0.143), -F.nb, col, 'bolt', 1.0, "Fork_BraceBolt"))
    # front fender (U-section swept around the wheel), solidified
    A = F.A
    Rfd = P['front_tire']['R'] + 0.045
    sec = [(-0.118, -0.034), (-0.1165, -0.012), (-0.108, 0.002), (-0.07, 0.010), (0.0, 0.013), (0.07, 0.010),
           (0.108, 0.002), (0.1165, -0.012), (0.118, -0.034)]
    bm = bmesh.new()
    nb_ = 30
    rows = []
    for i in range(nb_ + 1):
        beta = rad(38 + (124 - 38) * i / nb_)
        taper = 1.0 - 0.18 * (1 - i / nb_) ** 2
        row = []
        for x, ro in sec:
            r = Rfd + ro
            row.append(bm.verts.new(A + Vector((x * taper, -cos(beta) * r, sin(beta) * r)) - A))
        rows.append(row)
    for i in range(nb_):
        for j in range(len(sec) - 1):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    fender = obj_from_bm("Fender_Front", bm, col, M['paint_yellow'], mat4(A), angle=60)
    mod_solidify(fender, 0.0035, 1)
    mod_subsurf(fender, 1, 2)
    mod_weighted_normals(fender)
    T.append(fender)

    # --- front brake caliper (right side) ---
    rot_a = -0.078
    g = rad(32)
    cc = A + Vector((rot_a, 0.128 * cos(g), 0.128 * sin(g)))
    cal = box_obj("Brake_Front_Caliper", cc, (0.046, 0.088, 0.046), col, M['alu_cast'], rot=(math.degrees(g) + 90, 0, 0),
                  bevel=0.009, segs=3)
    T.append(cal)
    for k in (-1, 1):
        pc = cc + Vector((-0.024, 0, 0)) + Vector((0, -sin(g), cos(g))) * 0.022 * k
        T.append(cyl_obj("Brake_Front_CaliperPiston", pc, pc + Vector((-0.008, 0, 0)), 0.016, col, M['alu_cast'], segs=24, bevel=0.002))
    for k in (-1, 1):
        pb = cc + Vector((-0.024, 0, 0)) + Vector((0, -sin(g), cos(g))) * 0.036 * k + Vector((0, cos(g), sin(g))) * 0.012
        T.append(bolt(pb, (-1, 0, 0), col, 'cap', 1.0, "Brake_Front_BridgeBolt"))
    T.append(cyl_obj("Brake_Front_BleedNipple", cc + Vector((-0.012, 0, 0)) + Vector((0, cos(g), sin(g))) * 0.022,
                     cc + Vector((-0.012, 0, 0)) + Vector((0, cos(g), sin(g))) * 0.034, 0.003, col, M['steel_bright'], segs=6))
    # caliper hanger bracket to the slider
    T.append(box_obj("Brake_Front_CaliperBracket", F.p(0.10, 0.05, -0.118), (0.014, 0.06, 0.075), col, M['alu_cast'],
                     rot=(-P['rake'], 0, 0), bevel=0.004))
    T.append(bolt(F.p(0.085, 0.07, -0.128), (-1, 0, 0), col, 'bolt', 1.0, "Brake_Front_MountBolt"))
    T.append(bolt(F.p(0.125, 0.07, -0.128), (-1, 0, 0), col, 'bolt', 1.0, "Brake_Front_MountBolt"))

    # --- front wheel on the fork travel ---
    fw = build_wheel("FrontWheel", A, P['front_tire'], wcol, M,
                     rotor=dict(a=rot_a, r_out=0.152, r_in=0.062, holes=20, windows=8), axle_half=0.165)
    parent_later(fw, travel)

    # --- headlight (bucket, chrome rim, lens, reflector, bulb) ---
    hc = F.p(0.645, -0.13)
    hc = Vector((0, hc.y, hc.z))
    HL = mat4(hc) @ Matrix.Scale(1.14, 4)
    bucket = [(0.085, 0.014), (0.083, 0.03), (0.075, 0.05), (0.058, 0.068), (0.035, 0.08), (0.005, 0.0865),
              (-0.012, 0.0875), (-0.012, 0.0835), (0.004, 0.0825)]
    S.append(revolve_obj("Headlight_Bucket", bucket, col, M['chrome'], HL, segs=64, axis='Y', angle=40))
    rim = [(-0.008, 0.0835), (-0.010, 0.091), (-0.02, 0.094), (-0.03, 0.093), (-0.036, 0.088), (-0.036, 0.081),
           (-0.03, 0.08), (-0.025, 0.082)]
    S.append(revolve_obj("Headlight_Rim", rim, col, M['chrome'], HL, segs=64, axis='Y', closed=True, angle=40))
    lens = [(-0.045, 0.0), (-0.0445, 0.02)]
    for i in range(1, 9):   # fresnel steps
        r = 0.02 + 0.0075 * i
        lens += [(-0.0445 + 0.0015 * i, r - 0.0045), (-0.0430 + 0.0015 * i, r)]
    lens += [(-0.031, 0.083)]
    S.append(revolve_obj("Headlight_Lens", lens, col, M['glass'], HL, segs=64, axis='Y', angle=25))
    refl = [(0.05, 0.012)] + [(0.05 - 0.072 * (k / 10) ** 2, 0.012 + 0.068 * (k / 10)) for k in range(1, 11)]
    S.append(revolve_obj("Headlight_Reflector", refl, col, M['reflector'], HL, segs=64, axis='Y', angle=60))
    bulb = bmesh.new()
    bmesh.ops.create_uvsphere(bulb, u_segments=16, v_segments=10, radius=0.011, matrix=Matrix.Translation((0, 0.035, 0)))
    bm_cyl(bulb, 0.009, 0.009, 0.02, 16, Matrix.Translation((0, 0.04, 0)) @ Matrix.Rotation(-pi / 2, 4, 'X'))
    S.append(obj_from_bm("Headlight_Bulb", bulb, col, M['bulb'], HL))
    for o in S[-5:]:
        o.data.transform(Matrix.Scale(1.14, 4))
        o.matrix_world = mat4(hc)
    # headlight ears + clamps to the stanchions
    for sx in (-1, 1):
        clamp_c = F.p(0.645, 0, sx * fx)
        bm = bmesh.new()
        bm_cyl(bm, 0.036, 0.036, 0.034, 32, Matrix.Translation((0, 0, -0.017)))
        cl = obj_from_bm("Headlight_ForkClamp" + ("_L" if sx > 0 else "_R"), bm, col, M['paint_yellow'], mat4(clamp_c, Rf))
        mod_bevel(cl, 0.003, 2, 40)
        S.append(cl)
        ear_a = clamp_c + Vector((-sx * 0.012, -0.03, 0.0))
        ear_b = hc + Vector((sx * 0.1, 0.014, 0.0))
        S.append(curve_obj("Headlight_Ear" + ("_L" if sx > 0 else "_R"), [ear_a, (ear_a + ear_b) / 2 + Vector((sx * 0.012, 0, 0.012)), ear_b],
                           0.009, col, M['paint_yellow'], mode='smooth'))
        S.append(bolt(ear_b + Vector((sx * 0.004, 0, 0)), (sx, 0, 0), col, 'bolt', 1.3, "Headlight_EarBolt"))
        # amber markers on the ears
        mk = clamp_c + Vector((sx * 0.05, -0.02, 0.012))
        S.append(cyl_obj("Indicator_Front_Stalk", clamp_c + Vector((sx * 0.03, -0.01, 0.01)), mk, 0.006, col, M['rubber'], segs=12))
        S.append(box_obj("Indicator_Front_Housing", mk + Vector((sx * 0.012, -0.012, 0)), (0.03, 0.032, 0.026), col,
                         M['plastic_black'], bevel=0.004))
        S.append(box_obj("Indicator_Front_Lens", mk + Vector((sx * 0.012, -0.029, 0)), (0.026, 0.006, 0.022), col,
                         M['lens_orange'], bevel=0.0025))

    # --- handlebar & controls ---
    rp = F.p(0.813, off + 0.03)
    for sx in (-1, 1):
        base = rp + Vector((sx * 0.036, 0, 0))
        S.append(cyl_obj("Bar_Riser", base, base + Vector((0, 0, 0.055)), 0.014, col, M['alu_cast'], segs=24, bevel=0.001))
        S.append(box_obj("Bar_Clamp", base + Vector((0, 0, 0.068)), (0.036, 0.04, 0.026), col, M['alu_cast'], bevel=0.005, segs=3))
        for dy in (-0.012, 0.012):
            S.append(bolt(base + Vector((0, dy, 0.082)), (0, 0, 1), col, 'cap', 0.9, "Bar_ClampBolt"))
    T0 = rp + Vector((0, 0, 0.068))
    ty, tz = T0.y, T0.z
    bar_pts = [(-0.39, ty + 0.115, tz + 0.055), (-0.25, ty + 0.05, tz + 0.035), (-0.15, ty, tz),
               (0.15, ty, tz), (0.25, ty + 0.05, tz + 0.035), (0.39, ty + 0.115, tz + 0.055)]
    S.append(curve_obj("Bar_Handlebar", bar_pts, 0.0125, col, M['steel_rusty'], mode='bend', bend=0.07, res=16))
    S.append(curve_obj("Bar_CrossBrace", [(-0.14, ty + 0.012, tz + 0.055), (0.14, ty + 0.012, tz + 0.055)], 0.009, col,
                       M['steel_rusty'], mode='smooth'))
    for sx in (-1, 1):
        a_ = Vector((sx * 0.135, ty + 0.004, tz + 0.006))
        S.append(curve_obj("Bar_CrossBraceLeg", [a_, a_ + Vector((0, 0.006, 0.049))], 0.009, col, M['steel_rusty']))
    # gauge box (centre) with dial
    gb_c = Vector((0, ty + 0.004, tz + 0.085))
    S.append(box_obj("Gauge_Box", gb_c, (0.115, 0.07, 0.07), col, M['alu_cast'], rot=(-12, 0, 0), bevel=0.006, segs=3))
    face_c = gb_c + Vector((0, 0.033, 0.006))
    S.append(cyl_obj("Gauge_Bezel", face_c, face_c + Vector((0, 0.006, 0.0015)), 0.026, col, M['chrome'], segs=40, bevel=0.0015))
    S.append(cyl_obj("Gauge_Face", face_c + Vector((0, 0.0035, 0.0007)), face_c + Vector((0, 0.0045, 0.0009)), 0.0225, col,
                     M['plastic_black'], segs=40))
    S.append(cyl_obj("Gauge_Glass", face_c + Vector((0, 0.0055, 0.0011)), face_c + Vector((0, 0.0062, 0.0012)), 0.023, col,
                     M['glass'], segs=40))
    S.append(box_obj("Gauge_Needle", face_c + Vector((0.006, 0.005, 0.006)), (0.002, 0.0006, 0.017), col, M['lens_red'],
                     rot=(0, -35, 0), bevel=0))
    for sx in (-1, 1):
        S.append(bolt(gb_c + Vector((sx * 0.045, 0.0, 0.032)), (0, 0, 1), col, 'cap', 0.8, "Gauge_Bolt"))
    S.append(box_obj("Gauge_Bracket", gb_c + Vector((0, 0.0, -0.038)), (0.06, 0.03, 0.014), col, M['steel_rusty'], bevel=0.002))
    # right-side amber box and fuse box (as on the concept)
    S.append(box_obj("Bar_AuxLight_Housing", Vector((-0.115, ty - 0.012, tz + 0.045)), (0.05, 0.035, 0.035), col,
                     M['dark_metal'], rot=(-10, 0, 8), bevel=0.004))
    S.append(box_obj("Bar_AuxLight_Lens", Vector((-0.115, ty - 0.031, tz + 0.047)), (0.042, 0.006, 0.027), col,
                     M['lens_orange'], rot=(-10, 0, 8), bevel=0.002))
    S.append(box_obj("Bar_FuseBox", Vector((0.10, ty + 0.012, tz + 0.04)), (0.045, 0.04, 0.03), col, M['steel_heavy_rust'],
                     rot=(0, 0, -6), bevel=0.003))

    def bar_frame(sx, s_from_end):
        p0 = Vector(bar_pts[-1]) if sx > 0 else Vector(bar_pts[0])
        p1 = Vector(bar_pts[-2]) if sx > 0 else Vector(bar_pts[1])
        dvec = (p0 - p1).normalized()
        return p0 - dvec * s_from_end, dvec
    grip_prof = [(0.0, 0.0125), (0.0, 0.0215), (0.004, 0.0222), (0.008, 0.0185)]
    for i in range(16):
        z = 0.012 + i * 0.0068
        grip_prof += [(z, 0.0172), (z + 0.0022, 0.0181), (z + 0.0046, 0.0181), (z + 0.0068, 0.0172)]
    grip_prof += [(0.124, 0.0165), (0.126, 0.0125)]
    for sx, nm in ((-1, "R"), (1, "L")):
        g0, dv = bar_frame(sx, 0.128)
        S.append(revolve_obj("Bar_Grip_" + nm, grip_prof, col, M['rubber'], mat4(g0, frame_from_z(dv)), segs=32, axis='Z', angle=50))
        endp, _ = bar_frame(sx, 0.0)
        S.append(revolve_obj("Bar_EndWeight_" + nm, [(0.0, 0.0), (0.0, 0.0175), (0.006, 0.019), (0.02, 0.0185), (0.024, 0.014),
                                                     (0.025, 0.0)], col, M['chrome'], mat4(endp, frame_from_z(dv)), segs=32, axis='Z'))
        # switch housing
        sw, _ = bar_frame(sx, 0.155)
        S.append(box_obj("Bar_SwitchHousing_" + nm, sw + Vector((0, 0.0, 0.0)), (0.035, 0.05, 0.045), col, M['plastic_black'],
                         rot=(0, 0, math.degrees(math.atan2(dv.y, dv.x)) - 90 * 0 + 0), bevel=0.008, segs=3))
        S.append(cyl_obj("Bar_Switch_Button_" + nm, sw + Vector((0, 0.022, 0.012)), sw + Vector((0, 0.03, 0.014)), 0.006, col,
                         M['lens_red'] if sx < 0 else M['plastic_black'], segs=16))
        # lever perch + lever
        pc, _ = bar_frame(sx, 0.185)
        S.append(cyl_obj("Bar_LeverPerch_" + nm, pc - dv * 0.012, pc + dv * 0.012, 0.019, col, M['alu_cast'], segs=24, bevel=0.002))
        piv = pc + Vector((0, -0.035, 0.01))
        lev = [piv, piv + dv * 0.05 + Vector((0, -0.028, -0.004)), piv + dv * 0.13 + Vector((0, -0.032, -0.012)),
               piv + dv * 0.175 + Vector((0, -0.024, -0.016))]
        S.append(curve_obj("Bar_Lever_" + nm, lev, 0.0058, col, M['alu_cast'], radii=[1.3, 1.0, 0.9, 1.2]))
        S.append(cyl_obj("Bar_LeverPivot_" + nm, piv + Vector((0, 0, -0.012)), piv + Vector((0, 0, 0.012)), 0.007, col,
                         M['alu_cast'], segs=16))
        if sx < 0:
            # master cylinder + reservoir
            S.append(cyl_obj("Brake_Front_MasterCyl", pc + Vector((0, -0.018, 0.0)), pc + Vector((0, -0.018, 0.0)) + dv * 0.07,
                             0.012, col, M['alu_cast'], segs=24, bevel=0.0015))
            S.append(box_obj("Brake_Front_Reservoir", pc + dv * 0.03 + Vector((0, -0.02, 0.03)), (0.05, 0.035, 0.03), col,
                             M['dark_metal'], bevel=0.004))
            for k in (-1, 1):
                S.append(bolt(pc + dv * (0.03 + 0.017 * k) + Vector((0, -0.02, 0.046)), (0, 0, 1), col, 'cap', 0.55, "Brake_Front_ResScrew"))
            # throttle tube
            S.append(cyl_obj("Bar_ThrottleHousing", sw - dv * 0.0, sw + dv * 0.03, 0.021, col, M['plastic_black'], segs=24, bevel=0.003))
    # --- cables and hoses (splines) ---
    lp_r, dvr = bar_frame(-1, 0.185)
    lp_l, dvl = bar_frame(1, 0.185)
    cal_top = cc + Vector((-0.012, 0, 0)) + Vector((0, -sin(g), cos(g))) * 0.046
    S.append(curve_obj("Hose_Front_Brake", [lp_r + Vector((0, -0.03, 0.0)), lp_r + Vector((0.02, -0.12, -0.05)),
                                            F.p(0.72, -0.045, -fx - 0.03), F.p(0.45, 0.0, -fx - 0.048),
                                            F.p(0.22, 0.05, -fx - 0.03), cal_top], 0.0048, col, M['braided'], res=24))
    S.append(curve_obj("Cable_Clutch", [lp_l + Vector((0, -0.03, 0.0)), lp_l + Vector((-0.03, -0.14, -0.06)),
                                        Vector((0.13, -0.50, 0.95)), Vector((0.15, -0.36, 0.78)), Vector((0.16, -0.18, 0.62)),
                                        Vector((0.17, -0.06, 0.52))], 0.0042, col, M['wire_black'], res=24))
    th_p, _ = bar_frame(-1, 0.14)
    for k, dz in enumerate((0.0, 0.012)):
        S.append(curve_obj("Cable_Throttle_%d" % k, [th_p + Vector((0, 0.025, dz)), th_p + Vector((0.02, 0.09, -0.05 + dz)),
                                                     Vector((-0.10, -0.38, 0.96 + dz)), Vector((-0.13, -0.24, 0.84)),
                                                     Vector((-0.11, -0.06, 0.74 + dz))], 0.0036, col, M['wire_black'], res=24))
    for k, (sx, mat) in enumerate(((-1, 'wire_black'), (1, 'wire_black'), (1, 'wire_red'))):
        sw, _ = bar_frame(sx, 0.155)
        S.append(curve_obj("Wire_Switch_%d" % k, [sw + Vector((0, 0.02, -0.02)), sw + Vector((-sx * 0.05, 0.02, -0.08)),
                                                  hc + Vector((sx * 0.03, 0.07, 0.045)), hc + Vector((0, 0.09, 0.0))],
                           0.003, col, M[mat], res=20))
    S.append(curve_obj("Cable_Speedo", [gb_c + Vector((0, 0.0, -0.03)), gb_c + Vector((0.0, -0.06, -0.12)),
                                        F.p(0.5, -0.06, 0.06), F.p(0.2, -0.045, fx - 0.06), F.p(0.03, -0.01, fx - 0.04)],
                       0.004, col, M['wire_black'], res=24))

    for ob in S:
        parent_later(ob, steer)
    for ob in T:
        parent_later(ob, travel)
    return F, steer, travel


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


def ribbon_bm(bm, pts2d, width, spacing=0.002):
    """Stroke a 2D polyline into a quad ribbon (for painted emblems)."""
    P2 = [Vector((x, y)) for x, y in pts2d]
    dense = [P2[0]]
    for a, b in zip(P2, P2[1:]):
        n = max(1, int((b - a).length / spacing))
        for i in range(1, n + 1):
            dense.append(a.lerp(b, i / n))
    prev = None
    for i, p in enumerate(dense):
        a = dense[max(0, i - 1)]
        b = dense[min(len(dense) - 1, i + 1)]
        t = (b - a)
        if t.length < 1e-9:
            continue
        t.normalize()
        n = Vector((-t.y, t.x)) * width / 2
        v1 = bm.verts.new((p.x + n.x, p.y + n.y, 0))
        v2 = bm.verts.new((p.x - n.x, p.y - n.y, 0))
        if prev:
            bm.faces.new((prev[0], prev[1], v2, v1))
        prev = (v1, v2)


# ---------------------------------------------------------------------------------------------
# FRAME
# ---------------------------------------------------------------------------------------------

def build_frame(M, F, root):
    col = coll('Frame', COLL['Bike'])
    Yp = M['paint_yellow']
    off = P['fork_offset']
    obs = []
    obs.append(cyl_obj("Frame_HeadTube", F.p(0.585, off), F.p(0.778, off), 0.032, col, Yp, segs=40, bevel=0.002))
    for t in (0.585, 0.778):
        obs.append(cyl_obj("Frame_HeadBearingCup", F.p(t - 0.006, off), F.p(t + 0.006, off), 0.036, col, M['steel_rusty'], segs=40))
    ht_top = F.p(0.745, off + 0.03)
    ht_bot = F.p(0.605, off + 0.02)
    obs.append(curve_obj("Frame_Backbone", [ht_top, Vector((0, -0.22, 0.955)), Vector((0, 0.02, 0.885)), Vector((0, 0.11, 0.862))],
                         0.024, col, Yp, mode='bend', bend=0.08))
    rails = curve_obj("Frame_SeatRails", [Vector((0.075, 0.08, 0.855)), Vector((0.075, 0.74, 0.872)), Vector((0.0, 0.81, 0.872))],
                      0.0175, col, Yp, mode='bend', bend=0.06)
    mod_mirror(rails)
    obs.append(rails)
    obs.append(curve_obj("Frame_RailCross_Front", [Vector((-0.075, 0.11, 0.857)), Vector((0.075, 0.11, 0.857))], 0.016, col, Yp))
    obs.append(curve_obj("Frame_RailCross_Shock", [Vector((-0.075, 0.30, 0.86)), Vector((0.075, 0.30, 0.86))], 0.016, col, Yp))
    obs.append(curve_obj("Frame_DownTube", [ht_bot, Vector((0, -0.445, 0.72))], 0.027, col, Yp))
    cradle = curve_obj("Frame_Cradle", [Vector((0.0, -0.445, 0.735)), Vector((0.07, -0.43, 0.62)), Vector((0.07, -0.365, 0.27)),
                                        Vector((0.07, -0.25, 0.212)), Vector((0.07, 0.12, 0.212)), Vector((0.085, 0.215, 0.30)),
                                        Vector((0.09, 0.215, 0.56)), Vector((0.075, 0.12, 0.855))],
                       0.0195, col, Yp, mode='bend', bend=0.07)
    mod_mirror(cradle)
    obs.append(cradle)
    sub = curve_obj("Frame_SubframeStrut", [Vector((0.10, 0.25, 0.56)), Vector((0.08, 0.56, 0.866))], 0.0145, col, Yp)
    mod_mirror(sub)
    obs.append(sub)
    obs.append(curve_obj("Frame_CradleCross", [Vector((-0.07, -0.30, 0.235)), Vector((0.07, -0.30, 0.235))], 0.015, col, Yp))
    # gussets at the steering head
    g = extrude_obj("Frame_HeadGusset", [(0.0, 0.0), (0.17, -0.05), (0.08, -0.13), (-0.005, -0.14)], 0.006, col, Yp,
                    mat4(Vector((0, -0.405, 0.94)), Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))), bevel=0.0015)
    obs.append(g)
    # swingarm pivot plates (chunky, yellow) - mirrored
    plate = [(0.12, 0.33), (0.30, 0.33), (0.33, 0.40), (0.33, 0.50), (0.28, 0.60), (0.17, 0.62), (0.12, 0.55)]
    pp = extrude_obj("Frame_PivotPlates", plate, 0.018, col, M['paint_yellow_heavy'],
                     mat4(Vector((0.117, 0, 0)), Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))), bevel=0.004)
    mod_mirror(pp, mirror_obj=root)
    obs.append(pp)
    for (y, z) in ((0.15, 0.36), (0.29, 0.37), (0.30, 0.49), (0.18, 0.58), (0.25, 0.58)):
        for sx in (-1, 1):
            obs.append(bolt(Vector((sx * 0.126, y, z)), (sx, 0, 0), col, 'bolt', 1.0, "Frame_PlateBolt"))
    sp = P['swing_pivot']
    obs.append(cyl_obj("Frame_SwingarmPivotBolt", sp + Vector((-0.185, 0, 0)), sp + Vector((0.185, 0, 0)), 0.009, col, M['steel_bright'], segs=16))
    for sx in (-1, 1):
        obs.append(bolt(sp + Vector((sx * 0.185, 0, 0)), (sx, 0, 0), col, 'nut', 1.1, "Frame_PivotNut"))
    # engine mount plates
    for y, z in ((-0.33, 0.33), (-0.38, 0.62)):
        em = box_obj("Frame_EngineMount", Vector((0, y, z)), (0.17, 0.05, 0.06), col, M['steel_rusty'], bevel=0.004)
        obs.append(em)
        for sx in (-1, 1):
            obs.append(bolt(Vector((sx * 0.087, y, z)), (sx, 0, 0), col, 'bolt', 1.0, "Frame_EngineMountBolt"))
    # shock top mount tabs
    for sx in (-1, 1):
        obs.append(box_obj("Frame_ShockTab", Vector((sx * 0.022, 0.30, 0.83)), (0.006, 0.05, 0.06), col, Yp, bevel=0.002))
    # horn + oil cooler
    hc = Vector((-0.085, -0.46, 0.78))
    obs.append(revolve_obj("Elec_Horn", [(0.0, 0.0), (0.0, 0.045), (0.008, 0.048), (0.022, 0.046), (0.03, 0.02), (0.032, 0.0)],
                           col, M['steel_rusty'], mat4(hc, frame_from_z(Vector((-0.6, -1.0, 0.1)))), segs=40, axis='Z'))
    obs.append(cyl_obj("Elec_HornGrille", hc + Vector((0.0, 0.0, 0.0)), hc + Vector((0.003, 0.005, -0.0005)), 0.036, col, M['dark_metal'], segs=40))
    obs.append(box_obj("Elec_HornBracket", hc + Vector((0.035, 0.03, 0.0)), (0.05, 0.012, 0.025), col, M['steel_rusty'], rot=(0, 0, 30), bevel=0.002))
    oc = Vector((0, -0.475, 0.56))
    bm = bmesh.new()
    bm_box(bm, (0.16, 0.035, 0.11))
    ocb = obj_from_bm("Engine_OilCooler_Core", bm, col, M['dark_metal'], mat4(oc, Euler((rad(-12), 0, 0)).to_matrix()))
    mod_bevel(ocb, 0.003, 2)
    obs.append(ocb)
    bm = bmesh.new()
    bm_box(bm, (0.158, 0.038, 0.0016))
    fin = obj_from_bm("Engine_OilCooler_Fins", bm, col, M['alu_fins'], mat4(oc + Vector((0, 0, -0.05)), Euler((rad(-12), 0, 0)).to_matrix()))
    a = fin.modifiers.new("FinArray", 'ARRAY')
    a.count = 22
    a.relative_offset_displace = (0, 0, 3.0)
    obs.append(fin)
    for sx in (-1, 1):
        obs.append(box_obj("Engine_OilCooler_Tank", oc + Vector((sx * 0.087, 0, 0)), (0.016, 0.045, 0.12), col, M['alu_cast'],
                           rot=(-12, 0, 0), bevel=0.004))
    return obs


# ---------------------------------------------------------------------------------------------
# BODYWORK (subdivision cages)
# ---------------------------------------------------------------------------------------------

def build_body(M, root):
    col = coll('Bodywork', COLL['Bike'])
    obs = []

    # ---- fuel tank ----
    U = [-1, -0.5, 0, 0.5, 1]
    YS = [-0.385, -0.26, -0.13, 0.0, 0.135]
    ZT = [1.045, 1.09, 1.085, 1.045, 0.97]
    ZB = [0.85, 0.79, 0.775, 0.785, 0.835]
    HW = [0.125, 0.198, 0.205, 0.182, 0.125]
    WF = [0.62, 0.97, 1.0, 0.9, 0.5]           # width factor over height (w=-1..1)

    def tank(u, v, w):
        y = interp(U, YS, u)
        zt, zb = interp(U, ZT, u), interp(U, ZB, u)
        z = lerp(zb, zt, (w + 1) / 2)
        if w <= -0.99 and abs(v) < 0.6:          # frame tunnel
            z += 0.065 * (1 - (abs(v) / 0.6) ** 2)
        if u <= -0.99:                          # nose leans forward at the top
            y -= 0.035 * (w + 1) / 2
        hw = interp(U, HW, u) * interp(U, WF, w)
        if 0.2 < u < 0.8 and -0.8 < w < 0.2:     # knee recess
            hw *= 0.9
        return Vector((v * hw, y, z))
    tk = cage_obj("Body_FuelTank", col, M["paint_yellow"], tank, cuts=3, view=3, render=3)
    mod_weighted_normals(tk)
    obs.append(tk)
    bpy.context.view_layer.update()
    top, n = surface_hit(tk, (0, -0.2, 1.3), (0, 0, -1))
    if top is not None:
        cap = revolve_obj("Body_FuelCap", [(0, 0.0), (0, 0.034), (0.004, 0.036), (0.012, 0.035), (0.016, 0.03), (0.018, 0.012),
                                           (0.024, 0.01), (0.026, 0.0)], col, M['chrome'], mat4(top - n * 0.002, frame_from_z(n)),
                          segs=40, axis='Z')
        obs.append(cap)
        obs.append(box_obj("Body_FuelCap_Hinge", top + n * 0.015 + Vector((0, 0.04, 0)), (0.016, 0.018, 0.01), col, M['chrome'], bevel=0.002))
        bolt_ring(top + n * 0.001, n, 0.044, 6, col, 'cap', 0.6, name="Body_FuelCapScrew")
    # tank emblem (painted line art) on both sides - ribbons shrink-wrapped onto the tank
    emb = []
    ring = [(0.034 * cos(a), 0.034 * sin(a) + 0.012) for a in [2 * pi * i / 64 for i in range(65)]]
    crown = [(-0.02, -0.004), (-0.023, 0.024), (-0.01, 0.009), (0.0, 0.031), (0.01, 0.009), (0.023, 0.024),
             (0.02, -0.004), (-0.02, -0.004)]
    band = [(-0.021, -0.011), (0.021, -0.011)]
    for sx in (-1, 1):
        bm = bmesh.new()
        ribbon_bm(bm, ring, 0.0045)
        ribbon_bm(bm, crown, 0.004)
        ribbon_bm(bm, band, 0.004)
        for cx, cy in ((-0.023, 0.028), (0.0, 0.035), (0.023, 0.028)):
            pts = circle_pts(0.0018, 12, cx, cy)
            ribbon_bm(bm, pts + [pts[0]], 0.0034)
        hit, nn = surface_hit(tk, (sx * 0.5, -0.16, 0.955), (-sx, 0, 0))
        if hit is None:
            continue
        # local X -> along the bike, local Y -> up, local Z -> out of the tank
        x_ax = Vector((0, -1, 0))
        y_ax = nn.cross(x_ax).normalized()
        if y_ax.z < 0:
            y_ax = -y_ax
        x_ax = y_ax.cross(nn).normalized()
        Rm = Matrix((x_ax, y_ax, nn)).transposed()
        for v in bm.verts:
            v.co.x *= 1.55 * (-1 if sx < 0 else 1)
            v.co.y *= 1.55
        bm.normal_update()
        bmesh.ops.reverse_faces(bm, faces=[f for f in bm.faces if f.normal.z < 0])   # decal faces out of the tank
        e = obj_from_bm("Body_TankEmblem" + ("_L" if sx > 0 else "_R"), bm, col, M['emblem'], mat4(hit + nn * 0.004, Rm), smooth=True)
        sw = e.modifiers.new("Shrinkwrap", 'SHRINKWRAP')
        sw.target = tk
        sw.wrap_method = 'NEAREST_SURFACEPOINT'
        sw.wrap_mode = 'OUTSIDE_SURFACE'
        sw.offset = 0.0008
        emb.append(e)
    obs += emb

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
    # seat pan
    def pan(u, v, w):
        p = seat(u, v, w)
        p.z = lerp(0.858, 0.874, (w + 1) / 2)
        p.x *= 1.02
        return p
    obs.append(cage_obj("Body_SeatPan", col, M['plastic_black'], pan, cuts=2, view=2, render=2))
    # piping: loop around the cushion, snapped on with a shrinkwrap applied to the spline points
    loop = []
    for i in range(48):
        a = 2 * pi * i / 48
        u = math.copysign(abs(cos(a)) ** 0.35, cos(a))
        v = math.copysign(abs(sin(a)) ** 0.35, sin(a))
        p = seat(u * 0.995, v, 0.45)
        loop.append(p * 1.0)
    pip = curve_obj("Body_SeatPiping", loop, 0.0028, col, M['leather'], closed=True, res=4)
    sw = pip.modifiers.new("Shrinkwrap", 'SHRINKWRAP')
    sw.target = st
    sw.wrap_method = 'NEAREST_SURFACEPOINT'
    sw.offset = 0.0012
    sw.wrap_mode = 'OUTSIDE_SURFACE'
    sw.use_apply_on_spline = True
    obs.append(pip)

    # ---- tail cowl ----
    TU = [-1, -0.5, 0, 0.5, 1]
    TY = [0.40, 0.53, 0.66, 0.79, 0.90]
    TT = [0.905, 0.93, 0.945, 0.955, 0.96]
    TB = [0.835, 0.835, 0.84, 0.85, 0.88]
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
    hit, nn = surface_hit(tl, (0, 1.2, 0.9), (0, -1, 0))
    if hit is not None:
        obs.append(box_obj("Light_Tail_Housing", hit + nn * 0.004, (0.085, 0.012, 0.038), col, M['plastic_black'], bevel=0.005, segs=3))
        obs.append(box_obj("Light_Tail_Lens", hit + nn * 0.011, (0.078, 0.006, 0.031), col, M['lens_red'], bevel=0.004, segs=2))
    for sx in (-1, 1):
        hit, nn = surface_hit(tl, (sx * 0.4, 0.66, 0.895), (-sx, 0, 0))
        if hit is not None:
            R = frame_from_z(nn)
            obs.append(revolve_obj("Body_TailReflector_Bezel" + ("_L" if sx > 0 else "_R"),
                                   [(0, 0.0), (0, 0.021), (0.003, 0.022), (0.006, 0.019), (0.006, 0.0)], col, M['chrome'], mat4(hit, R),
                                   segs=32, axis='Z'))
            obs.append(revolve_obj("Body_TailReflector" + ("_L" if sx > 0 else "_R"),
                                   [(0.0, 0.0), (0.0068, 0.0), (0.0066, 0.016), (0.004, 0.0185)], col, M['lens_red'], mat4(hit, R),
                                   segs=32, axis='Z'))
        # rear indicators
        ip = Vector((sx * 0.07, 0.86, 0.85))
        obs.append(cyl_obj("Indicator_Rear_Stalk", ip, ip + Vector((sx * 0.05, 0.0, 0.0)), 0.006, col, M['rubber'], segs=12))
        obs.append(box_obj("Indicator_Rear_Housing", ip + Vector((sx * 0.06, 0.006, 0)), (0.026, 0.03, 0.022), col, M['plastic_black'], bevel=0.004))
        obs.append(box_obj("Indicator_Rear_Lens", ip + Vector((sx * 0.06, 0.022, 0)), (0.022, 0.005, 0.018), col, M['lens_orange'], bevel=0.002))

    # ---- side panels (under the seat nose) with round badge ----
    def side_panel(u, v, w):
        y = lerp(0.10, 0.37, (u + 1) / 2) + (w + 1) / 2 * 0.03
        z = lerp(0.635, 0.845, (w + 1) / 2) - 0.03 * ((u + 1) / 2) * (1 - (w + 1) / 2)
        x = 0.123 + (v + 1) / 2 * 0.018 - 0.01 * (w + 1) / 2
        return Vector((x, y, z))
    sp = cage_obj("Body_SidePanels", col, M['paint_yellow'], side_panel, cuts=2, view=2, render=3)
    mod_mirror(sp, mirror_obj=root)
    obs.append(sp)
    for sx in (-1, 1):
        c = Vector((sx * 0.142, 0.235, 0.745))
        obs.append(revolve_obj("Body_SideBadge" + ("_L" if sx > 0 else "_R"),
                               [(0, 0.0), (0, 0.032), (0.004, 0.033), (0.007, 0.028), (0.008, 0.012), (0.009, 0.0)],
                               col, M['chrome'], mat4(c, frame_from_z(Vector((sx, 0, 0)))), segs=40, axis='Z'))
        for (y, z) in ((0.13, 0.68), (0.33, 0.68), (0.14, 0.82), (0.34, 0.82)):
            obs.append(bolt(Vector((sx * 0.141, y, z)), (sx, 0, 0), col, 'cap', 0.8, "Body_SidePanelScrew"))

    # ---- front shroud boxes (the yellow blocks flanking the down tube) ----
    def shroud(u, v, w):
        y = lerp(-0.425, -0.25, (u + 1) / 2) + 0.06 * (1 - (w + 1) / 2) * (1 - (u + 1) / 2) * 0.0
        z = lerp(0.50, 0.70, (w + 1) / 2)
        y += -0.07 * (w + 1) / 2 * (1 - (u + 1) / 2)      # leading edge leans forward with the fork
        x = 0.105 + (v + 1) / 2 * 0.05
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
        for (y, z) in ((-0.40, 0.53), (-0.29, 0.53), (-0.43, 0.66), (-0.33, 0.67)):
            obs.append(bolt(Vector((sx * 0.158, y, z)), (sx, 0, 0), col, 'bolt', 1.0, "Body_ShroudBolt"))
        # louvre slots
        for k in range(4):
            obs.append(box_obj("Body_ShroudLouvre", Vector((sx * 0.157, -0.36 - 0.0 * k, 0.565 + k * 0.025)), (0.004, 0.07, 0.008), col,
                               M['dark_metal'], rot=(0, 0, 0), bevel=0.0015))

    # ---- skid plate ----
    path = [(-0.47, 0.33), (-0.44, 0.24), (-0.38, 0.185), (-0.2, 0.172), (0.0, 0.17), (0.13, 0.175)]
    secx = [(-0.115, 0.04), (-0.105, 0.0), (0.105, 0.0), (0.115, 0.04)]
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
    for y in (-0.3, -0.05, 0.08):
        for sx in (-1, 1):
            obs.append(bolt(Vector((sx * 0.07, y, 0.172)), (0, 0, -1), col, 'bolt', 1.0, "Body_SkidBolt"))
    # battery box
    obs.append(box_obj("Elec_BatteryBox", Vector((0, 0.45, 0.74)), (0.13, 0.09, 0.11), col, M['paint_black'], bevel=0.006))
    obs.append(box_obj("Elec_BatteryStrap", Vector((0, 0.45, 0.74)), (0.135, 0.02, 0.115), col, M['rubber'], bevel=0.003))
    obs.append(box_obj("Elec_Regulator", Vector((0.05, 0.36, 0.80)), (0.03, 0.06, 0.05), col, M['alu_fins'], bevel=0.002))
    return obs, tk, st, tl


# ---------------------------------------------------------------------------------------------
# ENGINE  (air-cooled 50 deg V-twin with exposed plumbing)
# ---------------------------------------------------------------------------------------------

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


def build_cylinder(prefix, base, direction, M, col, exhaust_side):
    obs = []
    dvec = Vector(direction).normalized()
    R = frame_from_z(dvec, Vector((1, 0, 0)))
    BM = mat4(base, R)
    obs += finned_block(prefix + "_Barrel", BM, 0.06, 0.0, 0.2, 0.205, 0.19, 0.0042, 0.0125, col, M)
    # head
    hb = bmesh.new()
    bm_box(hb, (0.165, 0.165, 0.095), Matrix.Translation((0, 0, 0.255)))
    head = obj_from_bm(prefix + "_Head", hb, col, M['alu_cast'], BM)
    mod_bevel(head, 0.012, 3, 40)
    obs.append(head)
    obs += finned_block(prefix + "_HeadFin", BM, 0.07, 0.212, 0.30, 0.235, 0.215, 0.004, 0.0135, col, M, corner=0.04)
    # rocker box
    rb = bmesh.new()
    bm_box(rb, (0.15, 0.12, 0.045), Matrix.Translation((0, 0, 0.325)))
    rock = obj_from_bm(prefix + "_RockerCover", rb, col, M['alu_cast'], BM)
    mod_bevel(rock, 0.016, 4, 40)
    obs.append(rock)
    for k in range(3):
        obs.append(box_obj(prefix + "_RockerRib", (BM @ Vector((0, -0.04 + 0.04 * k, 0.349))), (0.12, 0.006, 0.006), col,
                           M['alu_cast'], bevel=0.002))
        obs[-1].matrix_world = BM @ Matrix.Translation((0, -0.04 + 0.04 * k, 0.349))
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = BM @ Vector((sx * 0.06, sy * 0.045, 0.348))
            obs.append(bolt(p, dvec, col, 'cap', 0.9, prefix + "_RockerBolt"))
            p = BM @ Vector((sx * 0.085, sy * 0.075, 0.0))
            obs.append(bolt(p, dvec, col, 'nut', 0.55, prefix + "_BaseNut"))
    # spark plug + cap on the right side of the head
    sp0 = BM @ Vector((-0.06, 0.0, 0.29))
    sp_dir = (R @ Vector((-1, 0, 0.6))).normalized()
    obs.append(cyl_obj(prefix + "_SparkPlug_Hex", sp0, sp0 + sp_dir * 0.014, 0.0105, col, M['steel_bright'], segs=6))
    obs.append(cyl_obj(prefix + "_SparkPlug_Ceramic", sp0 + sp_dir * 0.014, sp0 + sp_dir * 0.04, 0.0055, col,
                       mat_cache('ceramic', lambda: mat_simple("M_Ceramic", (0.85, 0.82, 0.75), rough=0.25)), segs=16))
    obs.append(cyl_obj(prefix + "_PlugCap", sp0 + sp_dir * 0.03, sp0 + sp_dir * 0.075, 0.011, col, M['rubber'], r2=0.008, segs=20, bevel=0.002))
    # pushrod tubes (chrome) on the right
    for k in (-1, 1):
        a = BM @ Vector((-0.05, k * 0.03, -0.09))
        b = BM @ Vector((-0.07, k * 0.03, 0.32))
        obs.append(cyl_obj(prefix + "_PushrodTube", a, b, 0.0085, col, M['chrome'], segs=16))
    # exhaust port flange
    ep_dir = (R @ Vector((0, exhaust_side, -0.15))).normalized()
    ep = BM @ Vector((-0.03, exhaust_side * 0.07, 0.25))
    obs.append(cyl_obj(prefix + "_ExhaustFlange", ep, ep + ep_dir * 0.02, 0.03, col, M['steel_heavy_rust'], segs=24, bevel=0.002))
    # intake stub
    ip_dir = (R @ Vector((0, -exhaust_side, 0.0))).normalized()
    ip = BM @ Vector((0.0, -exhaust_side * 0.07, 0.25))
    obs.append(cyl_obj(prefix + "_IntakeStub", ip, ip + ip_dir * 0.03, 0.022, col, M['alu_cast'], segs=24))
    head_c = BM @ Vector((0, 0, 0.255))
    return obs, ep + ep_dir * 0.018, ep_dir, ip + ip_dir * 0.03, head_c


MAT_CACHE = {}


def mat_cache(key, fn):
    if key not in MAT_CACHE:
        MAT_CACHE[key] = fn()
    return MAT_CACHE[key]


def build_engine(M, root):
    col = coll('Engine', COLL['Bike'])
    obs = []
    # crankcase
    obs.append(box_obj("Engine_Crankcase", Vector((0, -0.035, 0.40)), (0.196, 0.43, 0.24), col, M['alu_cast'], bevel=0.045, segs=5))
    obs.append(cyl_obj("Engine_Crankcase_Nose", Vector((-0.096, -0.17, 0.39)), Vector((0.096, -0.17, 0.39)), 0.10, col, M['alu_cast'], segs=48, bevel=0.012))
    obs.append(box_obj("Engine_Gearbox", Vector((0, 0.11, 0.47)), (0.18, 0.17, 0.12), col, M['alu_cast'], bevel=0.035, segs=4))
    obs.append(box_obj("Engine_CaseSeam", Vector((0, -0.035, 0.40)), (0.006, 0.44, 0.25), col, M['dark_metal'], bevel=0.003))
    # sump fins
    bm = bmesh.new()
    bm_box(bm, (0.17, 0.005, 0.026))
    sf = obj_from_bm("Engine_SumpFins", bm, col, M['alu_fins'], mat4(Vector((0, -0.21, 0.29))))
    a = sf.modifiers.new("FinArray", 'ARRAY')
    a.count = 15
    a.use_relative_offset = False
    a.use_constant_offset = True
    a.constant_offset_displace = (0, 0.024, 0)
    mod_bevel(sf, 0.001, 1)
    obs.append(sf)
    obs.append(cyl_obj("Engine_DrainPlug", Vector((0, 0.0, 0.282)), Vector((0, 0.0, 0.265)), 0.01, col, M['steel_bright'], segs=6))
    # right-side covers (chrome clutch cover, points cover)
    cc = Vector((-0.1, 0.03, 0.405))
    obs.append(revolve_obj("Engine_ClutchCover", [(0.0, 0.0), (0.0, 0.096), (-0.006, 0.099), (-0.02, 0.095), (-0.036, 0.078), (-0.046, 0.045),
                                                  (-0.048, 0.0)], col, M['chrome'], mat4(cc), segs=64, axis='X', angle=50))
    obs.append(revolve_obj("Engine_ClutchCover_Cap", [(-0.046, 0.0), (-0.046, 0.04), (-0.052, 0.04), (-0.056, 0.033), (-0.056, 0.0)], col,
                           M['steel_bright'], mat4(cc), segs=48, axis='X'))
    bolt_ring(cc + Vector((-0.011, 0, 0)), (-1, 0, 0), 0.087, 10, col, 'cap', 0.9, name="Engine_ClutchCoverScrew")
    bolt_ring(cc + Vector((-0.056, 0, 0)), (-1, 0, 0), 0.026, 3, col, 'cap', 0.7, name="Engine_ClutchCapScrew")
    pc = Vector((-0.1, -0.175, 0.37))
    obs.append(revolve_obj("Engine_PointsCover", [(0.0, 0.0), (0.0, 0.06), (-0.006, 0.062), (-0.018, 0.055), (-0.024, 0.03), (-0.025, 0.0)],
                           col, M['chrome'], mat4(pc), segs=48, axis='X'))
    bolt_ring(pc + Vector((-0.008, 0, 0)), (-1, 0, 0), 0.05, 5, col, 'cap', 0.7, name="Engine_PointsCoverScrew")
    # oil filter canister (front right)
    obs.append(cyl_obj("Engine_OilFilter", Vector((-0.05, -0.29, 0.31)), Vector((-0.05, -0.36, 0.28)), 0.036, col, M['paint_black'], segs=32, bevel=0.004))
    # left-side covers (alternator, sprocket cover)
    ac = Vector((0.1, -0.13, 0.385))
    obs.append(revolve_obj("Engine_AlternatorCover", [(0.0, 0.0), (0.0, 0.1), (0.006, 0.103), (0.02, 0.098), (0.035, 0.07), (0.042, 0.03),
                                                       (0.043, 0.0)], col, M['alu_cast'], mat4(ac), segs=64, axis='X'))
    bolt_ring(ac + Vector((0.01, 0, 0)), (1, 0, 0), 0.09, 9, col, 'cap', 0.9, name="Engine_AltCoverScrew")
    obs.append(box_obj("Engine_SprocketCover", Vector((0.118, 0.07, 0.445)), (0.03, 0.15, 0.13), col, M['alu_cast'], bevel=0.02, segs=3))
    # front sprocket (behind the cover, chain line x=0.115)
    obs.append(build_sprocket("Engine_FrontSprocket", Vector((0.115, 0.11, 0.42)), 15, 0.0383, 0.007, col, M))
    # gear shaft & clutch actuator
    obs.append(cyl_obj("Engine_ClutchActuator", Vector((0.12, -0.02, 0.50)), Vector((0.145, -0.02, 0.52)), 0.016, col, M['alu_cast'], segs=24, bevel=0.002))

    # cylinders (V-twin)
    a1, a2 = rad(30), rad(20)
    fobs, f_port, f_dir, f_int, f_head = build_cylinder("Engine_FrontCyl", Vector((0, -0.085, 0.53)), (0, -sin(a1), cos(a1)), M, col, -1)
    robs, r_port, r_dir, r_int, r_head = build_cylinder("Engine_RearCyl", Vector((0, 0.06, 0.53)), (0, sin(a2), cos(a2)), M, col, 1)
    obs += fobs + robs
    # carburettor in the V + pod filter on the right
    cb = Vector((-0.02, -0.02, 0.715))
    obs.append(curve_obj("Engine_IntakeManifold_F", [f_int, cb + Vector((0, -0.03, 0.0))], 0.02, col, M['alu_cast']))
    obs.append(curve_obj("Engine_IntakeManifold_R", [r_int, cb + Vector((0, 0.03, 0.0))], 0.02, col, M['alu_cast']))
    obs.append(box_obj("Engine_Carb_Body", cb, (0.06, 0.07, 0.065), col, M['alu_cast'], bevel=0.012, segs=3))
    obs.append(cyl_obj("Engine_Carb_FloatBowl", cb + Vector((0, 0, -0.03)), cb + Vector((0, 0, -0.065)), 0.026, col, M['alu_cast'], segs=32, bevel=0.003))
    obs.append(cyl_obj("Engine_Carb_Top", cb + Vector((0, 0, 0.03)), cb + Vector((0, 0, 0.055)), 0.022, col, M['chrome'], segs=32, bevel=0.003))
    obs.append(cyl_obj("Engine_Carb_Throat", cb + Vector((-0.03, 0, 0)), cb + Vector((-0.08, 0, 0)), 0.028, col, M['alu_cast'], segs=32))
    filt_c = cb + Vector((-0.08, 0, 0))
    prof = [(0.0, 0.0), (0.0, 0.05), (-0.006, 0.058)]
    for i in range(14):
        a = -0.008 - i * 0.0045
        prof += [(a, 0.056), (a - 0.00225, 0.06)]
    prof += [(-0.075, 0.058), (-0.08, 0.05), (-0.082, 0.0)]
    obs.append(revolve_obj("Engine_AirFilter", prof, col, M['steel_rusty'], mat4(filt_c), segs=64, axis='X', angle=50))
    obs.append(revolve_obj("Engine_AirFilter_EndCap", [(-0.079, 0.0), (-0.079, 0.059), (-0.086, 0.058), (-0.09, 0.03), (-0.091, 0.0)],
                           col, M['chrome'], mat4(filt_c), segs=48, axis='X'))
    obs.append(cyl_obj("Engine_AirFilter_Nut", filt_c + Vector((-0.09, 0, 0)), filt_c + Vector((-0.1, 0, 0)), 0.01, col, M['steel_bright'], segs=6))
    # ignition coil + HT leads
    coil = Vector((-0.05, -0.29, 0.83))
    obs.append(cyl_obj("Elec_IgnitionCoil", coil, coil + Vector((0, 0.12, 0.0)), 0.025, col, M['paint_black'], segs=24, bevel=0.003))
    obs.append(box_obj("Elec_CoilBracket", coil + Vector((0.0, 0.06, 0.03)), (0.03, 0.10, 0.008), col, M['steel_rusty'], bevel=0.002))
    for k, (hc, end) in enumerate(((f_head, "F"), (r_head, "R"))):
        tip = hc + Vector((-0.14, 0, 0.02))
        obs.append(curve_obj("Elec_HTLead_" + end, [coil + Vector((-0.02, 0.02 + 0.08 * k, -0.012)), coil + Vector((-0.07, 0.05 + 0.06 * k, -0.06)),
                                                    tip], 0.0045, col, M['wire_red'], res=24))
    # rusty plumbing: oil lines, breathers (copper / braided / rubber)
    lines = [
        ("Engine_OilLine_Feed_F", [Vector((-0.11, -0.12, 0.47)), Vector((-0.15, -0.2, 0.55)), Vector((-0.13, -0.28, 0.66)), f_head + Vector((-0.07, -0.02, 0.06))], 0.0065, 'copper_rust'),
        ("Engine_OilLine_Feed_R", [Vector((-0.11, 0.0, 0.50)), Vector((-0.15, 0.05, 0.60)), Vector((-0.13, 0.12, 0.70)), r_head + Vector((-0.07, 0.02, 0.06))], 0.0065, 'copper_rust'),
        ("Engine_OilLine_Cooler_A", [Vector((-0.06, -0.29, 0.33)), Vector((-0.10, -0.40, 0.40)), Vector((-0.085, -0.47, 0.50))], 0.008, 'braided'),
        ("Engine_OilLine_Cooler_B", [Vector((0.06, -0.29, 0.33)), Vector((0.10, -0.42, 0.42)), Vector((0.085, -0.47, 0.50))], 0.008, 'braided'),
        ("Engine_Breather", [Vector((0.0, 0.14, 0.53)), Vector((0.02, 0.20, 0.62)), Vector((0.05, 0.27, 0.62)), Vector((0.06, 0.30, 0.40)), Vector((0.05, 0.31, 0.22))], 0.0075, 'hose'),
        ("Engine_FuelLine", [Vector((0.03, -0.05, 0.82)), Vector((0.02, -0.03, 0.78)), cb + Vector((0.02, 0.0, 0.03))], 0.0045, 'hose'),
        ("Engine_CopperTube_A", [Vector((-0.11, -0.05, 0.29)), Vector((-0.16, -0.02, 0.36)), Vector((-0.16, 0.12, 0.38)), Vector((-0.11, 0.16, 0.48))], 0.0055, 'copper_rust'),
        ("Engine_CopperTube_B", [Vector((-0.10, -0.24, 0.47)), Vector((-0.16, -0.26, 0.57)), Vector((-0.16, -0.14, 0.62)), Vector((-0.11, -0.11, 0.63))], 0.005, 'copper_rust'),
        ("Engine_CopperTube_C", [Vector((-0.12, 0.08, 0.53)), Vector((-0.17, 0.10, 0.60)), Vector((-0.16, 0.02, 0.66)), Vector((-0.08, -0.02, 0.69))], 0.0045, 'copper_rust'),
        ("Wire_Loom_Main", [Vector((0.0, -0.40, 0.86)), Vector((0.06, -0.30, 0.84)), Vector((0.08, -0.05, 0.82)), Vector((0.07, 0.25, 0.83)), Vector((0.05, 0.42, 0.80))], 0.008, 'wire_black'),
        ("Wire_Alternator", [ac + Vector((0.03, 0.03, 0.08)), Vector((0.13, -0.05, 0.56)), Vector((0.10, 0.02, 0.70)), Vector((0.08, 0.1, 0.80))], 0.005, 'wire_yellow'),
    ]
    for nm, pts, r, mk in lines:
        obs.append(curve_obj(nm, pts, r, col, M[mk], res=20))
    # hose clamps on the breather
    for t, p in ((0.0, Vector((0.0, 0.14, 0.535))), (1, Vector((0.05, 0.31, 0.24)))):
        obs.append(cyl_obj("Engine_HoseClamp", p + Vector((0, 0, -0.005)), p + Vector((0, 0, 0.005)), 0.0105, col, M['zinc'], segs=16))
    return obs, (f_port, f_dir), (r_port, r_dir)


# ---------------------------------------------------------------------------------------------
# EXHAUST
# ---------------------------------------------------------------------------------------------

def build_exhaust(M, root, fport, rport):
    col = coll('Exhaust', COLL['Bike'])
    obs = []
    fp, fd = fport
    rp, rd = rport
    lower = dict(c=Vector((-0.178, 0.0, 0.705)), r=0.039, y0=0.50, y1=0.95)
    upper = dict(c=Vector((-0.178, 0.0, 0.79)), r=0.043, y0=0.50, y1=0.975)
    hdr_f = [fp, fp + fd * 0.05, Vector((-0.07, -0.40, 0.62)), Vector((-0.112, -0.43, 0.46)), Vector((-0.118, -0.38, 0.27)),
             Vector((-0.112, -0.27, 0.198)), Vector((-0.112, 0.10, 0.195)), Vector((-0.16, 0.25, 0.235)),
             Vector((-0.198, 0.33, 0.38)), Vector((-0.198, 0.41, 0.62)), Vector((-0.18, 0.47, 0.70)), Vector((-0.178, 0.505, 0.705))]
    obs.append(curve_obj("Exhaust_Header_Front", hdr_f, 0.0215, col, M['steel_heavy_rust'], res=24, bevel_res=6))
    hdr_r = [rp, rp + rd * 0.04, Vector((-0.065, 0.285, 0.73)), Vector((-0.12, 0.36, 0.77)), Vector((-0.172, 0.44, 0.79)),
             Vector((-0.178, 0.505, 0.79))]
    obs.append(curve_obj("Exhaust_Header_Rear", hdr_r, 0.0215, col, M['steel_heavy_rust'], res=24, bevel_res=6))
    # heat-wrap section on the front header (spiral bands)
    obs.append(curve_obj("Exhaust_HeaderWrap", hdr_f[2:6], 0.0235, col,
                         mat_cache('wrap', lambda: mat_metal("M_HeatWrap", (0.22, 0.2, 0.17), 0.85, rust=0.5, grime=0.7, scale=30,
                                                             rust_cols=((0.06, 0.05, 0.04), (0.15, 0.10, 0.06), (0.25, 0.18, 0.10)))),
                         res=24, bevel_res=6))
    for nm, sp_ in (("Lower", lower), ("Upper", upper)):
        L = sp_['y1'] - sp_['y0']
        r = sp_['r']
        prof = [(0.0, 0.0), (0.0, 0.0225), (0.03, 0.026), (0.07, r), (0.075, r + 0.002), (0.08, r)]
        prof += [(L - 0.045, r), (L - 0.04, r + 0.002), (L - 0.035, r), (L - 0.01, r * 0.97), (L, r * 0.86), (L, r * 0.62),
                 (L - 0.006, r * 0.6), (L - 0.006, 0.0)]
        Mm = mat4(sp_['c'] + Vector((0, sp_['y0'], 0)), Matrix.Rotation(rad(-6), 3, 'X'))
        obs.append(revolve_obj("Exhaust_Muffler_" + nm, prof, col, M['chrome'], Mm, segs=48, axis='Y', angle=40))
        obs.append(revolve_obj("Exhaust_Muffler_" + nm + "_Outlet", [(L - 0.005, r * 0.45), (L + 0.018, r * 0.45), (L + 0.018, r * 0.38), (L - 0.03, r * 0.38)],
                               col, M['steel_heavy_rust'], Mm, segs=32, axis='Y', closed=True))
        for t in (0.13, L - 0.12):
            obs.append(revolve_obj("Exhaust_Strap_" + nm, [(t - 0.009, r + 0.0005), (t - 0.009, r + 0.003), (t + 0.009, r + 0.003), (t + 0.009, r + 0.0005)],
                                   col, M['steel_rusty'], Mm, segs=48, axis='Y', closed=True))
    # muffler hangers to the subframe
    for z in (0.705, 0.79):
        obs.append(box_obj("Exhaust_Hanger", Vector((-0.13, 0.62, z + 0.04)), (0.012, 0.03, 0.09), col, M['steel_rusty'], rot=(0, -25, 0), bevel=0.002))
    obs.append(box_obj("Exhaust_HangerBar", Vector((-0.115, 0.63, 0.855)), (0.06, 0.03, 0.01), col, M['steel_rusty'], bevel=0.002))
    obs.append(box_obj("Exhaust_HeatShield", Vector((-0.218, 0.75, 0.76)), (0.008, 0.22, 0.09), col, M['steel_heavy_rust'], rot=(0, -10, 0), bevel=0.003))
    for y in (0.67, 0.83):
        obs.append(bolt(Vector((-0.223, y, 0.765)), (-1, 0, 0), col, 'bolt', 0.8, "Exhaust_HeatShieldBolt"))
    return obs


# ---------------------------------------------------------------------------------------------
# REAR END (swingarm, monoshock, chain, rear wheel, brake)
# ---------------------------------------------------------------------------------------------

def build_rear(M, root):
    col = coll('RearEnd', COLL['Bike'])
    wcol = coll('Wheels', COLL['Bike'])
    sp = P['swing_pivot']
    RA = P['rear_axle']
    piv = make_empty("RearEnd_SwingarmPivot", mat4(sp), col, 'SINGLE_ARROW', 0.2)
    parent_later(piv, root)
    obs = []
    Yh = M['paint_yellow_heavy']
    # arms (mirrored)
    arm = beam_obj("Swingarm_Arms", Vector((0.145, sp.y + 0.02, sp.z)), Vector((0.155, RA.y + 0.05, RA.z)), 0.036, 0.082, 0.032, 0.056, col, Yh,
                   bevel=0.008, segs=3)
    mod_mirror(arm, mirror_obj=root)
    obs.append(arm)
    boss = cyl_obj("Swingarm_PivotBoss", Vector((0.128, sp.y, sp.z)), Vector((0.166, sp.y, sp.z)), 0.03, col, Yh, segs=32, bevel=0.003)
    mod_mirror(boss, mirror_obj=root)
    obs.append(boss)
    obs.append(beam_obj("Swingarm_CrossBrace", Vector((-0.135, sp.y + 0.11, sp.z + 0.005)), Vector((0.135, sp.y + 0.11, sp.z + 0.005)),
                        0.05, 0.06, 0.05, 0.06, col, Yh, bevel=0.008))
    for sx in (-1, 1):
        obs.append(box_obj("Swingarm_ShockClevis", Vector((sx * 0.02, sp.y + 0.12, sp.z + 0.05)), (0.007, 0.045, 0.05), col, Yh, bevel=0.002))
    # long box brace over the right arm (as on the concept)
    obs.append(beam_obj("Swingarm_BoxBrace_R", Vector((-0.158, sp.y + 0.09, sp.z + 0.045)), Vector((-0.162, RA.y - 0.09, RA.z + 0.04)),
                        0.04, 0.05, 0.036, 0.034, col, Yh, bevel=0.007))
    for t in (0.25, 0.5, 0.75):
        p = Vector((-0.18, lerp(sp.y + 0.09, RA.y - 0.09, t), lerp(sp.z + 0.045, RA.z + 0.04, t)))
        obs.append(bolt(p, (-1, 0, 0), col, 'bolt', 1.0, "Swingarm_BraceBolt"))
    # axle blocks / chain adjusters
    for sx in (-1, 1):
        obs.append(box_obj("Swingarm_AxleBlock", Vector((sx * 0.155, RA.y, RA.z)), (0.04, 0.05, 0.04), col, M['alu_cast'], bevel=0.004))
        obs.append(cyl_obj("Swingarm_Adjuster", Vector((sx * 0.155, RA.y + 0.025, RA.z + 0.008)), Vector((sx * 0.155, RA.y + 0.075, RA.z + 0.008)),
                           0.004, col, M['steel_bright'], segs=12))
        obs.append(bolt(Vector((sx * 0.155, RA.y + 0.072, RA.z + 0.008)), (0, 1, 0), col, 'nut', 0.55, "Swingarm_AdjusterNut"))
    # chain slider
    obs.append(box_obj("Swingarm_ChainSlider", Vector((0.145, sp.y + 0.12, sp.z + 0.045)), (0.04, 0.16, 0.014), col, M['plastic_black'], bevel=0.004))
    # rear hugger
    Rh = P['rear_tire']['R'] + 0.03
    secx = [(-0.115, -0.025), (-0.11, 0.0), (0.11, 0.0), (0.115, -0.025)]
    bm = bmesh.new()
    rows = []
    for i in range(17):
        b = rad(100 + 55 * i / 16)
        rows.append([bm.verts.new(RA + Vector((x, cos(b) * (Rh + dz) * -1 * -1, sin(b) * (Rh + dz))) - RA) for x, dz in secx])
    for i in range(16):
        for j in range(3):
            bm.faces.new((rows[i][j], rows[i][j + 1], rows[i + 1][j + 1], rows[i + 1][j]))
    hug = obj_from_bm("Fender_Rear_Hugger", bm, col, M['paint_yellow'], mat4(RA), angle=50)
    mod_solidify(hug, 0.003, 1)
    mod_subsurf(hug, 1, 2)
    obs.append(hug)

    # rear brake: caliper below the arm + torque arm
    ra = -0.088
    g = rad(205)
    cc = RA + Vector((ra, 0.09 * cos(g), 0.09 * sin(g)))
    obs.append(box_obj("Brake_Rear_Caliper", cc, (0.042, 0.07, 0.042), col, M['alu_cast'], rot=(205 + 90, 0, 0), bevel=0.008, segs=3))
    obs.append(cyl_obj("Brake_Rear_CaliperPiston", cc + Vector((-0.021, 0, 0)), cc + Vector((-0.029, 0, 0)), 0.015, col, M['alu_cast'], segs=24, bevel=0.002))
    obs.append(box_obj("Brake_Rear_Bracket", cc + Vector((-0.04, 0.04, 0.01)), (0.008, 0.12, 0.04), col, M['steel_rusty'], rot=(-8, 0, 0), bevel=0.003))
    obs.append(curve_obj("Brake_Rear_TorqueArm", [cc + Vector((-0.045, -0.01, -0.012)), Vector((-0.165, 0.42, 0.37))], 0.0065, col, M['steel_bright']))
    for p in (cc + Vector((-0.045, -0.01, -0.012)), Vector((-0.168, 0.42, 0.37))):
        obs.append(bolt(p, (-1, 0, 0), col, 'nut', 0.6, "Brake_Rear_TorqueArmNut"))
    obs.append(curve_obj("Hose_Rear_Brake", [cc + Vector((-0.01, 0.0, 0.02)), Vector((-0.12, 0.55, 0.42)), Vector((-0.13, 0.35, 0.47)), Vector((-0.13, 0.26, 0.40))],
                         0.0045, col, M['braided'], res=24))
    # rear wheel
    rw = build_wheel("RearWheel", RA, P['rear_tire'], wcol, M, rotor=dict(a=ra, r_out=0.112, r_in=0.05, holes=16, windows=6),
                     sprocket=dict(a=0.115, N=42, r_p=0.1062), axle_half=0.185)
    parent_later(rw, piv)

    # chain: one 2-pitch link module arrayed along a closed path with a curve modifier
    p = 0.015875
    c1, r1 = Vector((0.11, 0.42)), 0.0383
    c2, r2 = Vector((RA.y, RA.z)), 0.1062
    dv = c2 - c1
    L = dv.length
    alpha = math.atan2(dv.y, dv.x)
    phi = math.acos((r1 - r2) / L)
    path = []

    def arc(c, r, a0, a1, n):
        for i in range(n + 1):
            a = a0 + (a1 - a0) * i / n
            path.append(Vector((c.x + r * cos(a), c.y + r * sin(a))))
    up = alpha + phi
    lo = alpha - phi
    arc(c2, r2, up, lo, 40)                 # around the back of the rear sprocket
    arc(c1, r1, lo, up - 2 * pi, 24)        # around the front sprocket
    path3 = [(q.x, q.y, 0.0) for q in path]
    cur = poly_curve_obj("Chain_Path", path3, col, closed=True)
    cur.data.dimensions = '2D'
    cur.data.fill_mode = 'NONE'
    Rc = Matrix(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
    Mc = mat4(Vector((0.115, 0, 0)), Rc)
    cur.matrix_world = Mc
    cur.hide_render = True
    tot = sum((path[(i + 1) % len(path)] - path[i]).length for i in range(len(path)))
    n_mod = int(round(tot / (2 * p)))
    pe = tot / (2 * n_mod)
    bm = bmesh.new()

    def plate(z, w, mat_idx):
        pts = []
        for i in range(10):
            a = pi / 2 + pi * i / 9
            pts.append((0.0 + w * cos(a), w * sin(a)))
        for i in range(10):
            a = -pi / 2 + pi * i / 9
            pts.append((pe + w * cos(a), w * sin(a)))
        pts.insert(10, (pe / 2, -w * 0.78))
        pts.append((pe / 2, w * 0.78))
        extrude_bm(bm, pts, 0.0014, Matrix.Translation((0, 0, z)))
    for z in (-0.0062, 0.0062):
        plate(z, 0.0074, 0)                  # outer plates (pins at 0 and pe)
    for z in (-0.0045, 0.0045):
        bm2_off = Matrix.Translation((pe, 0, z))
        pts = []
        for i in range(10):
            a = pi / 2 + pi * i / 9
            pts.append((0.0 + 0.0072 * cos(a), 0.0072 * sin(a)))
        for i in range(10):
            a = -pi / 2 + pi * i / 9
            pts.append((pe + 0.0072 * cos(a), 0.0072 * sin(a)))
        extrude_bm(bm, pts, 0.0013, bm2_off)
    for x in (0.0, pe):
        bm_cyl(bm, 0.0049, 0.0049, 0.0078, 12, Matrix.Translation((x, 0, -0.0039)))   # rollers
        bm_cyl(bm, 0.0025, 0.0025, 0.0158, 8, Matrix.Translation((x, 0, -0.0079)))    # pins
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

    # monoshock (rigged with constraints so it follows the swingarm)
    top = Vector((0.0, 0.30, 0.805))
    bot = Vector((0.0, sp.y + 0.12, sp.z + 0.06))
    ax_ = (bot - top).normalized()
    Lsh = (bot - top).length
    Rt = frame_from_z(ax_)
    upper = bpy.data.objects.new("Shock_Upper", None)
    upper.empty_display_type = 'PLAIN_AXES'
    upper.empty_display_size = 0.03
    col.objects.link(upper)
    upper.matrix_world = mat4(top, Rt)
    lower = bpy.data.objects.new("Shock_Lower", None)
    lower.empty_display_type = 'PLAIN_AXES'
    lower.empty_display_size = 0.03
    col.objects.link(lower)
    lower.matrix_world = mat4(bot, frame_from_z(-ax_))
    parent_later(upper, root)
    parent_later(lower, piv)
    # fixed mount points (frame / swingarm); each shock half aims at the opposite mount
    top_m = make_empty("Shock_TopMount", mat4(top), col, 'SPHERE', 0.015)
    bot_m = make_empty("Shock_BottomMount", mat4(bot), col, 'SPHERE', 0.015)
    parent_later(top_m, root)
    parent_later(bot_m, piv)
    c = upper.constraints.new('DAMPED_TRACK')
    c.target = bot_m
    c.track_axis = 'TRACK_Z'
    c = lower.constraints.new('DAMPED_TRACK')
    c.target = top_m
    c.track_axis = 'TRACK_Z'
    sh_up, sh_lo = [], []
    # body (from top eye down)
    body_prof = [(0.0, 0.0), (0.0, 0.016), (0.012, 0.016), (0.016, 0.024), (0.17, 0.024), (0.175, 0.021), (0.18, 0.012), (0.18, 0.0)]
    sh_up.append(revolve_obj("Shock_Body", body_prof, col, M['alu_cast'], mat4(top, Rt), segs=32, axis='Z'))
    sh_up.append(cyl_obj("Shock_TopEye", top + Vector((-0.012, 0, 0)), top + Vector((0.012, 0, 0)), 0.014, col, M['alu_cast'], segs=24))
    sh_up.append(cyl_obj("Shock_Reservoir", top + ax_ * 0.03 + Vector((0, 0.035, 0.0)), top + ax_ * 0.15 + Vector((0, 0.035, 0.0)), 0.019, col,
                         M['paint_red'], segs=32, bevel=0.003))
    sh_up.append(curve_obj("Shock_ReservoirHose", [top + ax_ * 0.025, top + ax_ * 0.01 + Vector((0, 0.02, 0)), top + ax_ * 0.03 + Vector((0, 0.035, 0))],
                           0.004, col, M['braided']))
    sh_up.append(revolve_obj("Shock_PreloadCollar", [(0.04, 0.025), (0.04, 0.033), (0.052, 0.033), (0.052, 0.025)], col, M['paint_red'],
                             mat4(top, Rt), segs=32, axis='Z', closed=True))
    # lower: shaft + spring seat + eye (from bottom eye upward)
    Rb = frame_from_z(-ax_)
    sh_lo.append(cyl_obj("Shock_Shaft", bot, bot - ax_ * (Lsh - 0.16), 0.0075, col, M['chrome'], segs=20))
    sh_lo.append(cyl_obj("Shock_BottomEye", bot + Vector((-0.012, 0, 0)), bot + Vector((0.012, 0, 0)), 0.013, col, M['alu_cast'], segs=24))
    sh_lo.append(revolve_obj("Shock_LowerSeat", [(0.018, 0.0), (0.018, 0.03), (0.026, 0.034), (0.03, 0.034), (0.03, 0.0)], col, M['alu_cast'],
                             mat4(bot, Rb), segs=32, axis='Z'))
    # spring: helix generated by a Screw modifier (edit turns/pitch on the modifier); stretches between the seats
    s_len = Lsh - 0.03 - 0.052
    turns = 7
    bm = bmesh.new()
    circ = [bm.verts.new((0.031 + 0.0055 * cos(a), 0.0, 0.0055 * sin(a))) for a in [2 * pi * i / 10 for i in range(10)]]
    for i in range(10):
        bm.edges.new((circ[i], circ[(i + 1) % 10]))
    up_dir = -ax_
    xa = Vector((1, 0, 0))
    za = xa.cross(up_dir).normalized()
    Rs = Matrix((xa, up_dir, za)).transposed()          # local Y runs up the spring (Stretch To axis)
    bm = bmesh.new()
    circ = [bm.verts.new((0.031 + 0.0055 * cos(a), 0.0055 * sin(a), 0.0)) for a in [2 * pi * i / 10 for i in range(10)]]
    for i in range(10):
        bm.edges.new((circ[i], circ[(i + 1) % 10]))
    spring = obj_from_bm("Shock_Spring", bm, col, M['paint_red'], mat4(bot - ax_ * 0.03, Rs), uv=False)
    scr = spring.modifiers.new("Coil", 'SCREW')
    scr.axis = 'Y'
    scr.angle = 2 * pi * turns
    scr.screw_offset = s_len
    scr.steps = 16 * turns
    scr.render_steps = 24 * turns
    scr.use_merge_vertices = False
    scr.use_normal_calculate = True
    scr.use_smooth_shade = True
    spring_top = bpy.data.objects.new("Shock_SpringTopTarget", None)
    spring_top.empty_display_size = 0.02
    col.objects.link(spring_top)
    spring_top.matrix_world = mat4(top + ax_ * 0.052, Rt)
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
        if o.parent is None:
            parent_later(o, piv)
    return piv


# ---------------------------------------------------------------------------------------------
# CONTROLS (pegs, pedals, stand)
# ---------------------------------------------------------------------------------------------

def build_controls(M, root):
    col = coll('Controls', COLL['Bike'])
    obs = []
    for sx in (-1, 1):
        base = Vector((sx * 0.13, 0.15, 0.315))
        obs.append(box_obj("Peg_Bracket", base + Vector((sx * 0.0, 0.0, 0.02)), (0.03, 0.05, 0.07), col, M['steel_rusty'], bevel=0.004))
        pc = Vector((sx * 0.20, 0.15, 0.31))
        obs.append(box_obj("Peg_Body", pc, (0.11, 0.034, 0.016), col, M['steel_rusty'], bevel=0.003))
        bm = bmesh.new()
        for i in range(7):
            for sy in (-1, 1):
                x = -0.045 + i * 0.015
                bmesh.ops.create_cone(bm, cap_ends=True, segments=4, radius1=0.0045, radius2=0.0005, depth=0.008,
                                      matrix=Matrix.Translation((x, sy * 0.014, 0.012)) @ Matrix.Rotation(pi / 4, 4, 'Z'))
        obs.append(obj_from_bm("Peg_Teeth", bm, col, M['steel_rusty'], mat4(pc), smooth=False))
        obs.append(cyl_obj("Peg_PivotPin", base + Vector((sx * 0.02, -0.02, 0.0)), base + Vector((sx * 0.02, 0.02, 0.0)), 0.004, col, M['steel_bright'], segs=12))
    # brake pedal (right, red) + pushrod to master cylinder
    pv = Vector((-0.15, 0.185, 0.29))
    obs.append(cyl_obj("Pedal_Brake_Pivot", pv + Vector((0.02, 0, 0)), pv + Vector((-0.012, 0, 0)), 0.013, col, M['steel_bright'], segs=24))
    obs.append(curve_obj("Pedal_Brake_Arm", [pv, Vector((-0.162, 0.08, 0.255)), Vector((-0.175, -0.04, 0.255)), Vector((-0.19, -0.075, 0.275))],
                         0.0085, col, M['paint_red'], mode='bend', bend=0.04))
    obs.append(box_obj("Pedal_Brake_Pad", Vector((-0.205, -0.075, 0.28)), (0.045, 0.03, 0.012), col, M['steel_rusty'], rot=(10, 0, 0), bevel=0.003))
    obs.append(curve_obj("Pedal_Brake_Pushrod", [pv + Vector((0.0, 0.012, 0.02)), Vector((-0.14, 0.24, 0.36)), Vector((-0.135, 0.27, 0.40))],
                         0.0045, col, M['paint_red']))
    mc = Vector((-0.135, 0.27, 0.42))
    obs.append(cyl_obj("Brake_Rear_MasterCyl", mc + Vector((0, 0, -0.03)), mc + Vector((0, 0, 0.03)), 0.011, col, M['alu_cast'], segs=24, bevel=0.002))
    obs.append(cyl_obj("Brake_Rear_Reservoir", Vector((-0.12, 0.33, 0.56)), Vector((-0.12, 0.33, 0.61)), 0.015, col, M['plastic_black'], segs=24, bevel=0.003))
    obs.append(curve_obj("Brake_Rear_ResHose", [Vector((-0.12, 0.33, 0.56)), Vector((-0.13, 0.31, 0.50)), mc + Vector((0, 0, 0.03))], 0.003, col, M['hose']))
    # gear shifter (left)
    sv = Vector((0.13, -0.02, 0.36))
    obs.append(cyl_obj("Shifter_Shaft", Vector((0.11, -0.02, 0.36)), sv + Vector((0.03, 0, 0)), 0.006, col, M['steel_bright'], segs=12))
    obs.append(curve_obj("Shifter_Lever", [sv + Vector((0.028, 0, 0)), Vector((0.165, -0.08, 0.32)), Vector((0.17, -0.12, 0.305))], 0.0065, col,
                         M['steel_rusty'], mode='bend', bend=0.03))
    obs.append(cyl_obj("Shifter_Toe", Vector((0.165, -0.12, 0.305)), Vector((0.21, -0.12, 0.305)), 0.008, col, M['rubber'], segs=16, bevel=0.002))
    # kickstand (retracted)
    ks = Vector((0.095, 0.10, 0.215))
    obs.append(cyl_obj("Kickstand_Pivot", ks + Vector((-0.012, 0, 0)), ks + Vector((0.02, 0, 0)), 0.012, col, M['steel_bright'], segs=20))
    obs.append(curve_obj("Kickstand_Leg", [ks + Vector((0.015, 0, 0)), Vector((0.13, 0.30, 0.235)), Vector((0.135, 0.36, 0.25))], 0.011, col,
                         M['paint_black'], mode='bend', bend=0.05))
    obs.append(box_obj("Kickstand_Foot", Vector((0.135, 0.37, 0.25)), (0.03, 0.04, 0.01), col, M['paint_black'], rot=(20, 0, 0), bevel=0.003))
    obs.append(curve_obj("Kickstand_Spring", [ks + Vector((0.02, 0.02, -0.005)), Vector((0.10, 0.18, 0.21))], 0.003, col, M['steel_rusty']))
    return obs


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
    add_prop(root, "steer", 0.0, -38.0, 38.0, "Handlebar steering angle (degrees)")
    add_prop(root, "fork_travel", 0.0, 0.0, 0.14, "Front fork compression (m)")
    add_prop(root, "swingarm_angle", 0.0, -8.0, 12.0, "Swingarm rotation (degrees, + = compress)")
    add_prop(root, "distance", 0.0, -1000.0, 1000.0, "Distance rolled forward (m) - spins both wheels")
    ob = bpy.data.objects
    drive(ob["FrontEnd_Steer"], 'rotation_euler', 2, root, "steer", "v*pi/180")
    tr = ob["FrontEnd_ForkTravel"]
    drive(tr, 'location', 2, root, "fork_travel", "%.6f+v" % tr.location.z)
    drive(ob["RearEnd_SwingarmPivot"], 'rotation_euler', 0, root, "swingarm_angle", "v*pi/180")
    drive(ob["FrontWheel_Pivot"], 'rotation_euler', 0, root, "distance", "v/%.4f" % P['front_tire']['R'])
    drive(ob["RearWheel_Pivot"], 'rotation_euler', 0, root, "distance", "v/%.4f" % P['rear_tire']['R'])


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
    F, steer, travel = build_front(M, root)
    frame = build_frame(M, F, root)
    body, tank, seat, tail = build_body(M, root)
    eng, fport, rport = build_engine(M, root)
    exh = build_exhaust(M, root, fport, rport)
    rear = build_rear(M, root)
    ctrl = build_controls(M, root)
    # everything without a parent hangs off the root
    for o in bike.all_objects:
        if o is root or o.parent is not None:
            continue
        if any(o is c for c, _ in PARENT_JOBS):
            continue
        parent_later(o, root)
    resolve_parents()
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
