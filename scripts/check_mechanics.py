"""
Mechanical sanity checker: sweeps the rig through its range of motion and reports every place
where a moving assembly intersects something it must not touch.

    blender -b bike/wasteland_scrambler.blend --python scripts/check_mechanics.py -- [--static] [--json out.json]

Assemblies (from the rig hierarchy):
    FRONT_UNSPRUNG  everything under FrontEnd_ForkTravel (wheel, sliders, fender, caliper)
    STEER           everything under FrontEnd_Steer that is not unsprung (yokes, bars, headlight)
    REAR            everything under RearEnd_SwingarmPivot (swingarm, wheel, chain, caliper)
    STATIC          the rest (frame, engine, bodywork, exhaust, electrics)

Poses: steering lock +/-35 deg, full fork compression, swingarm full bump / droop and combinations.
With --static, also lists interferences between static parts (pipes through frame tubes, cables
through solid parts ...). Intended contacts (bolts, bearings, a stanchion inside its slider ...) are
whitelisted below.
"""
import bpy
import sys
import json
import fnmatch
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
STATIC_CHECK = "--static" in argv
LINE_SWEEP = "--lines" in argv
OUT = argv[argv.index("--json") + 1] if "--json" in argv else None

L_ = 35.0
POSES = [dict(steer=st, fork_travel=tr, swingarm_angle=sw)
         for st, tr, sw in ((0, 0, 0), (L_, 0, 0), (-L_, 0, 0), (0, 0.12, 0), (L_, 0.12, 0), (-L_, 0.12, 0), (L_ / 2, 0.06, 0),
                            (0, 0, 8), (0, 0, -6), (0, 0, 4))]

# intended contacts: (pattern A, pattern B) - fnmatch on object names, order independent
ALLOW = [
    ("Fork_Stanchions*", "Fork_Sliders*"), ("Fork_Stanchions*", "Fork_DustSeals*"), ("Fork_Stanchions*", "Fork_*Yoke*"),
    ("Fork_Stanchions*", "Headlight_ForkClamp*"), ("Fork_Stanchions*", "Fork_TopCaps*"),
    ("Fork_SteeringStem*", "Frame_Head*"), ("Fork_*Yoke*", "Frame_HeadBearing*"), ("Fork_SteeringStem*", "Fork_*Yoke*"),
    ("FrontWheel_Axle*", "Fork_*"), ("RearWheel_Axle*", "Swingarm_*"), ("RearWheel_AxleNut*", "Swingarm_*"), ("RearWheel_AxleCap*", "Swingarm_*"),
    ("FrontWheel_AxleNut*", "Fork_*"), ("FrontWheel_AxleCap*", "Fork_*"),
    ("Frame_SwingarmPivotBolt*", "Swingarm_*"), ("Frame_PivotNut*", "Swingarm_*"),
    ("Shock_*", "Shock_*"), ("Shock_*", "Swingarm_Shock*"), ("Shock_*", "Frame_ShockTab*"), ("Shock_*", "Swingarm_CrossBrace*"),
    ("Shock_*", "Frame_RailCross_Shock*"),
    ("Chain_*", "*Sprocket*"), ("Chain_*", "RearWheel_SprocketCarrier*"),
    ("Brake_Rear_Caliper*", "Swingarm_*"), ("Brake_Rear_Pad*", "Swingarm_*"), ("Brake_Rear_Bracket*", "Swingarm_*"), ("Brake_Rear_Torque*", "Swingarm_*"),
    ("Brake_Rear_BleedNipple*", "Swingarm_*"), ("Brake_Front_Caliper*", "Fork_*"), ("Brake_Front_HoseGuide*", "Fork_*"), ("Elec_SpeedSensor*", "Fork_*"), ("Fender_Front*", "Fork_Brace*"),
    ("Fender_Front*", "Fender_MountBolt*"), ("Fender_Rear_Hugger*", "Swingarm_*"),
    ("Swingarm_ChainGuard*", "Swingarm_*"), ("Brake_Rear_Bracket*", "RearWheel_Axle*"), ("Brake_Rear_*", "RearWheel_AxleNut*"),
    ("HW_*", "*"), ("*Bolt*", "*"), ("*Nut*", "*"), ("*Screw*", "*"),
    ("Cable_*", "*"), ("Wire_*", "*"), ("Hose_*", "*"), ("Loom_*", "*"),   # flexible lines are checked in --static mode
]
STATIC_ALLOW = [
    ("Frame_*", "Frame_*"), ("Body_*", "Body_*"), ("Engine_*", "Engine_*"), ("Exhaust_*", "Exhaust_*"), ("Elec_*", "Elec_*"),
    ("Engine_*", "Frame_EngineMount*"), ("Engine_*", "Frame_HeadSteady*"), ("Engine_*_ExhaustFlange*", "Exhaust_Header*"),
    ("Body_Seat*", "Frame_*"), ("Body_TailCowl*", "Frame_*"), ("Body_SideCover*", "Frame_*"), ("Body_FuelTank*", "Frame_TankMount*"),
    ("Body_SkidPlate*", "Frame_*"), ("Body_Shroud*", "Frame_*"), ("Body_FrontShrouds*", "Frame_*"),
    ("Engine_*", "Frame_PivotPlates*"), ("Engine_Countershaft*", "*"),
    ("HW_*", "*"), ("*Bolt*", "*"), ("*Nut*", "*"), ("*Screw*", "*"), ("Elec_*", "Frame_*"), ("Light_*", "Body_*"),
    ("Indicator_*", "*"), ("Peg_*", "Peg_*"), ("Peg_*", "Frame_*"), ("Pedal_*", "Pedal_*"), ("Brake_*", "Brake_*"), ("Pedal_*", "Brake_*"),
    ("Kickstand_*", "Kickstand_*"), ("Kickstand_*", "Frame_*"), ("Shifter_*", "Shifter_*"), ("Shifter_*", "Engine_*"), ("Exhaust_Hanger*", "*"),
    ("Kickstart_*", "Kickstart_*"), ("Kickstart_*", "Engine_*"), ("Fuel_*", "Fuel_*"), ("Fuel_*", "Body_FuelTank*"), ("Brake_Rear_*", "Frame_*"),
    ("Elec_*", "Engine_*"), ("Elec_*", "Body_*"), ("Exhaust_*", "Engine_*ExhaustStud*"),
    ("HW_CableTie*", "*"), ("Shock_*", "Shock_*"), ("Shock_*", "Frame_Shock*"), ("Shock_*", "Frame_RailCross_Shock*"), ("Light_*", "Light_*"),
    ("Loom_*", "Wire_*"), ("Loom_*", "Loom_*"), ("Body_TailCowl*", "Loom_Rear*"), ("Body_TailCowl*", "Wire_RearIndicator*"),
    ("Exhaust_HeaderHanger*", "*"), ("Pedal_*", "Pedal_*"), ("Brake_Rear_*", "Pedal_*"), ("Elec_RearBrakeSwitch*", "*"),
    ("Brake_Rear_*", "Hose_Rear_Reservoir*"), ("Body_SkidSpacer*", "Frame_*"), ("Engine_HoseClamp*", "Hose_*"),
    ("Brake_Rear_Master*", "Frame_PivotPlates*"), ("Pedal_Brake_PivotBoss*", "Frame_*"),
]
STATIC_ALLOW_LINES = [("Exhaust_*", "Exhaust_*"), ("Loom_Rear*", "Body_TailCowl*"), ("Wire_RearIndicator*", "Body_TailCowl*"),
                      ("Hose_Front_Brake*", "Brake_Front_HoseGuide*"), ("Hose_OilFeed*", "Engine_*"), ("Hose_Breather*", "Engine_*"), ("Shock_*", "Shock_*"), ("Hose_Fuel*", "Fuel_*"),
                      ("Hose_Rear_Reservoir*", "Brake_Rear_*"), ("Loom_Switch*", "Bar_*"), ("Hose_Front_Brake*", "Brake_Front_*")]
# endpoints of flexible lines legitimately enter their fittings; only interior penetrations are reported
LINE_END_TOL = 0.025

def allowed(a, b, table):
    for pa, pb in table:
        if (fnmatch.fnmatch(a, pa) and fnmatch.fnmatch(b, pb)) or (fnmatch.fnmatch(a, pb) and fnmatch.fnmatch(b, pa)):
            return True
    return False


bike = bpy.data.collections["Bike"]
cutters = set(bpy.data.collections["Cutters"].all_objects)
objs = [o for o in bike.all_objects if o.type in {'MESH', 'CURVE', 'FONT'} and o not in cutters and not o.hide_render]
names = bpy.data.objects


def under(o, root_name):
    p = o.parent
    while p:
        if p.name == root_name:
            return True
        p = p.parent
    return False


def group(o):
    if under(o, "FrontWheel_Pivot"):
        return "FRONT_WHEEL"
    if under(o, "RearWheel_Pivot"):
        return "REAR_WHEEL"
    if under(o, "FrontEnd_ForkTravel"):
        return "FRONT_UNSPRUNG"
    if under(o, "FrontEnd_Steer"):
        return "STEER"
    if under(o, "RearEnd_SwingarmPivot"):
        return "REAR"
    return "STATIC"


G = {o.name: group(o) for o in objs}
root = names["Bike_Root"]


LINES = ("Cable_", "Wire_", "Hose_", "Loom_", "Exhaust_Header_")
ends = {}


def build_bvhs():
    dg = bpy.context.evaluated_depsgraph_get()
    out = {}
    for o in objs:
        ev = o.evaluated_get(dg)
        try:
            me = ev.to_mesh()
        except RuntimeError:
            continue
        if me is None or not me.polygons:
            ev.to_mesh_clear()
            continue
        mw = ev.matrix_world
        vs = [mw @ v.co for v in me.vertices]
        polys = [tuple(p.vertices) for p in me.polygons]
        lo = Vector((min(v.x for v in vs), min(v.y for v in vs), min(v.z for v in vs)))
        hi = Vector((max(v.x for v in vs), max(v.y for v in vs), max(v.z for v in vs)))
        cent = [sum((vs[i] for i in p), Vector()) / len(p) for p in polys]
        out[o.name] = (BVHTree.FromPolygons(vs, polys, all_triangles=False, epsilon=0.0), lo, hi, cent)
        if o.type == 'CURVE' and o.data.splines and o.data.splines[0].bezier_points:
            bps = o.data.splines[0].bezier_points
            ends[o.name] = (mw @ bps[0].co, mw @ bps[-1].co)
        ev.to_mesh_clear()
    return out


def aabb_hit(a, b, pad=0.0):
    return all(a[1][i] - pad <= b[2][i] and b[1][i] - pad <= a[2][i] for i in range(3))


report = {}
PAIRS = [("FRONT_WHEEL", "FRONT_UNSPRUNG"), ("FRONT_WHEEL", "STEER"), ("FRONT_WHEEL", "STATIC"), ("FRONT_WHEEL", "REAR"),
         ("FRONT_WHEEL", "REAR_WHEEL"), ("FRONT_UNSPRUNG", "STATIC"), ("FRONT_UNSPRUNG", "STEER"), ("STEER", "STATIC"),
         ("REAR_WHEEL", "REAR"), ("REAR_WHEEL", "STATIC"), ("REAR", "STATIC"), ("FRONT_UNSPRUNG", "REAR")]
for pose in POSES:
    for k, v in pose.items():
        root[k] = v
    bpy.context.view_layer.update()
    bv = build_bvhs()
    by_g = {}
    for n, b in bv.items():
        by_g.setdefault(G[n], []).append(n)
    tag = "steer=%+d travel=%.2f swing=%+d" % (pose['steer'], pose['fork_travel'], pose['swingarm_angle'])
    for ga, gb in PAIRS:
        for a in by_g.get(ga, []):
            for b in by_g.get(gb, []):
                if allowed(a, b, ALLOW) or not aabb_hit(bv[a], bv[b]):
                    continue
                hits = bv[a][0].overlap(bv[b][0])
                if hits:
                    key = tuple(sorted((a, b)))
                    r = report.setdefault(key, dict(tris=0, poses=[]))
                    r['tris'] = max(r['tris'], len(hits))
                    r['poses'].append(tag)
    if LINE_SWEEP and pose != POSES[0]:
        lines = [n for n in bv if n.startswith(LINES)]
        for ln in lines:
            e0, e1 = ends.get(ln, (None, None))
            for b in bv:
                if b == ln or b.startswith(LINES) or b.startswith(("HW_", "Exhaust_Header")) or allowed(ln, b, STATIC_ALLOW_LINES) or not aabb_hit(bv[ln], bv[b]):
                    continue
                hits = bv[ln][0].overlap(bv[b][0])
                keep = [h for h in hits if e0 is None or ((bv[ln][3][h[0]] - e0).length > LINE_END_TOL and (bv[ln][3][h[0]] - e1).length > LINE_END_TOL)]
                if keep:
                    key = ("LINE:" + ln, b)
                    r = report.setdefault(key, dict(tris=0, poses=[]))
                    r['tris'] = max(r['tris'], len(keep))
                    r['poses'].append(tag)
for k in ("steer", "fork_travel", "swingarm_angle"):
    root[k] = 0.0
bpy.context.view_layer.update()

static = {}
if STATIC_CHECK:
    bv = build_bvhs()
    st = [n for n in bv if G[n] == "STATIC" or n.startswith(LINES)]
    for i, a in enumerate(st):
        for b in st[i + 1:]:
            if allowed(a, b, STATIC_ALLOW) or not aabb_hit(bv[a], bv[b]):
                continue
            hits = bv[a][0].overlap(bv[b][0])
            if not hits:
                continue
            line = a if a.startswith(LINES) else (b if b.startswith(LINES) else None)
            if line is not None and line in ends:
                e0, e1 = ends[line]
                other = b if line == a else a
                tri = bv[line][3]
                ia = 0 if line == a else 1
                keep = 0
                for h in hits:
                    c = tri[h[ia]]
                    if (c - e0).length > LINE_END_TOL and (c - e1).length > LINE_END_TOL:
                        keep += 1
                if keep == 0:
                    continue
                static[(a, b)] = keep
            else:
                static[(a, b)] = len(hits)

print("=" * 78)
print("MOTION INTERFERENCES: %d" % len(report))
for (a, b), r in sorted(report.items(), key=lambda kv: -kv[1]['tris']):
    print("  %-40s x %-34s tris=%4d  %s" % (a, b, r['tris'], r['poses'][0] + (" (+%d poses)" % (len(r['poses']) - 1) if len(r['poses']) > 1 else "")))
if STATIC_CHECK:
    print("STATIC INTERFERENCES: %d" % len(static))
    for (a, b), n in sorted(static.items(), key=lambda kv: -kv[1]):
        print("  %-34s x %-34s tris=%4d" % (a, b, n))
print("=" * 78)
if OUT:
    json.dump(dict(motion={"%s|%s" % k: v for k, v in report.items()},
                   static={"%s|%s" % k: v for k, v in static.items()}), open(OUT, "w"), indent=1)
