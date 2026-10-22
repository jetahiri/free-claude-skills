# SPDX-License-Identifier: GPL-3.0-or-later
# This script runs inside Blender and imports bpy, so it is distributed under the GPL (version 3 or later), as the
# Blender Foundation treats scripts that use bpy as derived works. The rest of this repository is MIT.
"""Image to 3D: rebuild the forms in a reference image as a lit, procedural 3D scene in Blender (Cycles, NVIDIA card
when Blender finds one, else the processor).

Reusable pipeline. Everything specific to one image lives in a settings file (a named preset, JSON);
the next image needs only a new settings file plus its depth map.

Run:
  blender -b --factory-startup --python-exit-code 1 --python image_to_3d.py -- --settings <preset.json>
          --stage clay|final|anim|projection --out <png, or a folder for anim> [--frame N] [--start A --end B]
          [--scale S] [--samples N] [--device auto|cpu] [--set key.sub=value ...]

Stages: clay (grey, no warm light), final (procedural material and lights), anim (frames start..end of the loop),
projection (the reference projected onto the surface, unlit: the measuring target, never the product).
--set overrides one setting for this run (used by inverse_fit.py to search material and light values).
--device cpu skips the card; auto uses OptiX or CUDA when Blender finds an NVIDIA card and says so when it finds none.

Inputs, made once per image: depth.npy (relative inverse depth, larger = nearer) and mask.npy (True on the forms),
from depth_from_ref.py in this folder. Settings and what they change are listed in SETTINGS.md beside this file.
"""
import bpy, sys, math, json, argparse, os
from pathlib import Path
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--settings", required=True)
ap.add_argument("--stage", default="clay", choices=["clay", "final", "anim", "projection"])
ap.add_argument("--out", required=True)
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--samples", type=int, default=0)
ap.add_argument("--scale", type=float, default=0)
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
args = ap.parse_args(argv)

SP = Path(args.settings).resolve()
S = json.loads(SP.read_text(encoding="utf-8"))
for kv in args.set:
    key, val = kv.split("=", 1)
    node_ = S
    parts = key.split(".")
    for p in parts[:-1]:
        node_ = node_[int(p)] if isinstance(node_, list) else node_[p]
    last = int(parts[-1]) if isinstance(node_, list) else parts[-1]
    node_[last] = json.loads(val)
rel = lambda p: (SP.parent / p).resolve()
CLAY, PROJ = args.stage == "clay", args.stage == "projection"
G, GEO, MAT, CAM, MOT, REN = S["image"], S["geometry"], S["material"], S["camera"], S["motion"], S["render"]
FRAMES = MOT["loop_frames"]
W, H = G["width"], G["height"]
for key in ("depth", "mask"):
    if not rel(G[key]).is_file():
        sys.exit(f"missing {key} file {rel(G[key])}: make it with depth_from_ref.py, then point image.{key} at it in the preset")

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def node(nt, kind, loc, **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n

def lin(h):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)

# --- camera at the origin looking down +Y; the surface is built through this same view, so it lands on the image
cd = bpy.data.cameras.new("Cam")
cd.sensor_fit = "VERTICAL"
cd.sensor_height = 24
cd.lens = CAM["lens"]
cam = bpy.data.objects.new("Cam", cd)
scene.collection.objects.link(cam)
scene.camera = cam
cam.rotation_euler = (math.radians(90), 0, 0)
TAN_V = (cd.sensor_height / 2) / cd.lens
TAN_H = TAN_V * W / H

# --- geometry from the depth map
GW, GH = GEO["grid"]

def down(a, w, h):
    # Blender's Python has no Pillow: box-average down to the grid
    a = a.astype(np.float32)
    ys = np.linspace(0, a.shape[0], h + 1).astype(int)
    xs = np.linspace(0, a.shape[1], w + 1).astype(int)
    return np.array([[a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean() for j in range(w)] for i in range(h)], dtype=np.float32)

def blur(a, sigma):
    r = max(1, int(round(sigma * 2.4)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for ax in (0, 1):
        a = np.apply_along_axis(lambda row: np.convolve(np.pad(row, r, mode="edge"), k, "valid"), ax, a)
    return a

D = down(np.load(rel(G["depth"])), GW, GH)
MS = down(np.load(rel(G["mask"])).astype(np.float32), GW, GH)
if G.get("image") and GEO.get("gap_depth", 0) > 0:
    # sharpen the separations a soft depth map loses: where the image is dark inside the forms (the gaps and
    # creases between folds), push the surface back so each fold reads as its own rolled layer
    ref = bpy.data.images.load(str(rel(G["image"])))
    px = np.array(ref.pixels[:], dtype=np.float32).reshape(ref.size[1], ref.size[0], 4)[::-1, :, :3]
    L = down(px @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), GW, GH)
    Lf = L[MS > 0.5]
    dark = np.clip((np.percentile(Lf, GEO["gap_percentile"]) - L) / max(1e-4, np.percentile(Lf, GEO["gap_percentile"]) - Lf.min()), 0, 1)
    D = D - GEO["gap_depth"] * blur(dark, GEO["gap_soften"]) * (MS > 0.5)
D = blur(D, GEO["smooth"])
MS = blur(MS, GEO["smooth"])
M = MS > 0.5
if PROJ:
    M[:] = True
# near every outline the surface curls away from the camera, so each form rolls off like a rounded body
ROLL = np.clip((MS - 0.5) / GEO["curl_width"], 0, 1)
ROLL = 1 - ROLL * ROLL * (3 - 2 * ROLL)
OV = GEO["overscan"]
UU, VV = np.meshgrid(((np.arange(GW) + 0.5) / GW * 2 - 1) * OV, (1 - (np.arange(GH) + 0.5) / GH * 2) * OV)
NEAR, FAR = GEO["near"], GEO["near"] + GEO["thickness"]
idx = -np.ones((GH, GW), int)
idx[M] = np.arange(int(M.sum()))
n = int(M.sum())
a, b, c_, d_ = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
ok = (a >= 0) & (b >= 0) & (c_ >= 0) & (d_ >= 0)
faces = np.stack([a[ok], b[ok], c_[ok], d_[ok]], -1).tolist()
me = bpy.data.meshes.new("Forms")
me.from_pydata([(0, 0, 0)] * n, [], faces)
forms = bpy.data.objects.new("Forms", me)
scene.collection.objects.link(forms)
sub = forms.modifiers.new("Soft", "SUBSURF")
sub.levels = 0
sub.render_levels = GEO["subdivision"]
for poly in me.polygons:
    poly.use_smooth = True
uvl = me.uv_layers.new(name="screen")
lv = np.zeros(len(me.loops), int)
me.loops.foreach_get("vertex_index", lv)
uvl.data.foreach_set("uv", np.stack([(UU + 1) / 2, (VV + 1) / 2], -1)[M][lv].astype(np.float32).ravel())

def pose(f):
    t = 2 * math.pi * (f % FRAMES) / FRAMES
    # a slow drift of the folds in depth and a small sideways sway; every term is whole cycles over the loop
    drift = MOT["drift"] * np.sin(2 * math.pi * (0.8 * VV + 0.3 * UU) + t) + MOT["drift2"] * np.sin(2 * math.pi * 1.7 * VV - 2 * t)
    dist = NEAR + (1 - D) * (FAR - NEAR) + drift * (0.4 + D) + GEO["curl"] * ROLL + (S.get("tubes") or {}).get("body_offset", 0)
    sway = MOT["sway"] * np.sin(2 * math.pi * 0.6 * VV + t)
    pts = np.stack([(UU + sway) * TAN_H * dist, dist, VV * TAN_V * dist], -1)[M]
    me.vertices.foreach_set("co", pts.astype(np.float32).ravel())
    me.update()

TB = S.get("tubes")
tube_objs = []
if TB and not PROJ:
    # each traced fold becomes a thick round tube of near-constant radius swept along the image's own centre-line,
    # with a rolled loop at its left end where the image's folds curl back; the depth surface stays behind as body
    def catmull(P, n):
        # closed Catmull-Rom through the hand-traced points: a smooth loop with no corners
        P = np.asarray(P, dtype=np.float64)
        m = len(P)
        out = []
        for i in range(m):
            p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
            for s_ in np.linspace(0, 1, max(2, n // m), endpoint=False):
                s2, s3 = s_ * s_, s_ * s_ * s_
                out.append(0.5 * (2 * p1 + (-p0 + p2) * s_ + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3))
        return np.array(out)

    for k, lp in enumerate(TB.get("loops", [])):
        # a hand-traced fold: a closed tube of constant radius along the loop the image shows
        P = catmull(lp["points"], TB["samples_per_loop"])
        cv = bpy.data.curves.new(f"Fold{k}", "CURVE")
        cv.dimensions = "3D"
        cv.bevel_depth = 1.0
        cv.bevel_resolution = 8
        sp = cv.splines.new("POLY")
        sp.use_cyclic_u = True
        sp.points.add(len(P) - 1)
        for p_, (u, v, d) in zip(sp.points, P):
            dist = NEAR + (1 - d) * (FAR - NEAR)
            p_.co = ((u * 2 - 1) * TAN_H * dist, dist, (1 - v * 2) * TAN_V * dist, 1)
            p_.radius = lp["radius"] * 2 * TAN_V * dist
        ob = bpy.data.objects.new(f"Fold{k}", cv)
        scene.collection.objects.link(ob)
        tube_objs.append(ob)
        if TB.get("webs"):
            # the soft membrane inside each loop: rings from the tube's path toward a point set back behind it,
            # sagging away from the camera, so the loop reads as one rolled sheet, not a hoop
            Q = P[:: max(1, len(P) // 96)]
            ctr = Q.mean(0)
            ctr[2] -= TB["web_back"]
            R_ = TB["web_rings"]
            wv, wf = [], []
            for ri in range(R_ + 1):
                t_ = ri / R_ * TB["web_reach"]
                for (u, v, d) in Q:
                    uu, vv = u + (ctr[0] - u) * t_, v + (ctr[1] - v) * t_
                    dd = d + (ctr[2] - d) * t_ - TB["web_sag"] * math.sin(math.pi * t_)
                    dist = NEAR + (1 - dd) * (FAR - NEAR)
                    wv.append(((uu * 2 - 1) * TAN_H * dist, dist, (1 - vv * 2) * TAN_V * dist))
            m_ = len(Q)
            for ri in range(R_):
                for j in range(m_):
                    wf.append((ri * m_ + j, ri * m_ + (j + 1) % m_, (ri + 1) * m_ + (j + 1) % m_, (ri + 1) * m_ + j))
            wm = bpy.data.meshes.new(f"Web{k}")
            wm.from_pydata(wv, [], wf)
            for poly in wm.polygons:
                poly.use_smooth = True
            wo = bpy.data.objects.new(f"Web{k}", wm)
            scene.collection.objects.link(wo)
            sm = wo.modifiers.new("Thick", "SOLIDIFY")
            sm.thickness = lp["radius"] * 2 * TAN_V * NEAR * TB["web_thickness"]
            wo.modifiers.new("Soft", "SUBSURF").render_levels = 2
            tube_objs.append(wo)
    traced = json.loads(rel(TB["source"]).read_text(encoding="utf-8"))["folds"] if TB.get("source") else []
    for k, fd in enumerate(traced):
        P = np.array(fd["points"])
        P = P[np.argsort(P[:, 0])]
        cv = bpy.data.curves.new(f"Fold{k}", "CURVE")
        cv.dimensions = "3D"
        cv.bevel_depth = 1.0
        cv.bevel_resolution = 6
        cv.use_fill_caps = True
        sp = cv.splines.new("POLY")
        r_img = max(TB["min_radius"], float(np.median(P[:, 3])) * TB["radius_scale"])
        pts = []
        for u, v, d, _ in P:
            dist = NEAR + (1 - d) * (FAR - NEAR) + TB["inset"]
            pts.append(((u * 2 - 1) * TAN_H * dist, dist, (1 - v * 2) * TAN_V * dist, r_img * 2 * TAN_V * dist))
        x0, y0, z0, R = pts[0]
        loop = []
        for i in range(1, TB["curl_steps"] + 1):
            th = math.pi * i / TB["curl_steps"]
            # the curl: round the left end back into the depth and up, one tube width across
            loop.append((x0 - R * TB["curl_reach"] * math.sin(th), y0 + R * TB["curl_depth"] * (1 - math.cos(th)),
                         z0 + R * TB["curl_rise"] * (1 - math.cos(th)), R))
        allp = loop[::-1] + pts
        sp.points.add(len(allp) - 1)
        for p_, (x, y, z, rr) in zip(sp.points, allp):
            p_.co = (x, y, z, 1)
            p_.radius = rr
        ob = bpy.data.objects.new(f"Fold{k}", cv)
        scene.collection.objects.link(ob)
        ob.modifiers.new("Soft", "SUBSURF").render_levels = 1
        tube_objs.append(ob)

pose(args.frame)
bpy.app.handlers.frame_change_pre.append(lambda sc, *a: pose(sc.frame_current))

# --- material: procedural, so it stays right when the form changes
fm = bpy.data.materials.new("M_Form")
nt = fm.node_tree
bsdf = nt.nodes["Principled BSDF"]
if PROJ:
    nt.nodes.remove(bsdf)
    img = nt.nodes.new("ShaderNodeTexImage")
    img.image = bpy.data.images.load(str(rel(G["image"])))
    img.extension = "EXTEND"
    uvn = node(nt, "ShaderNodeUVMap", (-900, 0), uv_map="screen")
    nt.links.new(uvn.outputs["UV"], img.inputs["Vector"])
    em = node(nt, "ShaderNodeEmission", (0, 0))
    nt.links.new(img.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
elif CLAY:
    bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
else:
    bsdf.inputs["Base Color"].default_value = (*lin(MAT["base"]), 1)
    bsdf.inputs["Roughness"].default_value = MAT["roughness"]
    bsdf.inputs["Specular IOR Level"].default_value = MAT["specular"]
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Radius"].default_value = tuple(MAT["subsurface_radius"])
    bsdf.inputs["Subsurface Scale"].default_value = MAT["subsurface_scale"]
    bsdf.inputs["Sheen Weight"].default_value = MAT["sheen"]
    if MAT.get("grain_strength", 0) > 0:
        nz = node(nt, "ShaderNodeTexNoise", (-700, -300))
        nz.inputs["Scale"].default_value = MAT["grain_scale"]
        nz.inputs["Detail"].default_value = 2.0
        geo = node(nt, "ShaderNodeNewGeometry", (-900, -300))
        nt.links.new(geo.outputs["Position"], nz.inputs["Vector"])
        bump = node(nt, "ShaderNodeBump", (-450, -300))
        bump.inputs["Strength"].default_value = MAT["grain_strength"]
        bump.inputs["Distance"].default_value = MAT.get("grain_depth", 0.005)
        nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
me.materials.append(fm)
if TB and TB.get("hide_body"):
    # traced tubes alone: the depth surface between them showed as flat cut-off panels
    forms.hide_render = True
for ob in tube_objs:
    ob.data.materials.append(fm)

MB = S.get("mesh")
if MB and not PROJ:
    # a generated closed mesh (image_to_mesh.py) replaces the depth surface and tubes. The generator saw the image
    # cropped to the silhouette's box, so the mesh's box is fitted onto that box on screen at the set depth.
    forms.hide_render = True
    for ob in tube_objs:
        ob.hide_render = True
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(rel(MB["path"])))
    gen = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.context.view_layer.update()
    V = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in gen])
    lo, hi = V.min(0), V.max(0)
    al = np.load(rel(G["mask"]))
    ys, xs = np.where(al > 0.5)
    u0, u1 = xs.min() / al.shape[1] * 2 - 1, xs.max() / al.shape[1] * 2 - 1
    v1, v0 = 1 - ys.min() / al.shape[0] * 2, 1 - ys.max() / al.shape[0] * 2
    dist = MB["depth"]
    # screen box in world units at that depth; uniform scale from the height, centre from the box
    sx = (u1 - u0) * TAN_H * dist / max(1e-6, hi[0] - lo[0])
    sz = (v1 - v0) * TAN_V * dist / max(1e-6, hi[2] - lo[2])
    # scale and offset fine-tune the box fit (offset in frame fractions, x right and y up)
    s_ = (sz if MB.get("fit") == "height" else (sx + sz) / 2) * MB.get("scale", 1.0)
    ox, oy = MB.get("offset", [0.0, 0.0])
    piv = bpy.data.objects.new("GenPivot", None)
    scene.collection.objects.link(piv)
    for o in gen:
        o.parent = piv
        o.data.materials.clear()
        o.data.materials.append(fm)
        for poly in o.data.polygons:
            poly.use_smooth = True
    ctr = (lo + hi) / 2
    piv.scale = (s_, s_, s_)
    piv.location = (((u0 + u1) / 2 + 2 * ox) * TAN_H * dist - ctr[0] * s_, dist - ctr[1] * s_,
                    ((v0 + v1) / 2 + 2 * oy) * TAN_V * dist - ctr[2] * s_)
    piv.rotation_euler = (0, 0, math.radians(MB.get("yaw", 0)))

# --- lights from the settings: positions are in the camera's space (x right, y depth, z up)
for L_ in S["lights"]:
    if CLAY and L_.get("warm"):
        continue
    ld = bpy.data.lights.new(L_["name"], "AREA")
    ld.shape = "DISK"
    ld.size = L_["size"]
    ld.energy = L_["power"]
    ld.color = lin(L_["color"])
    ob = bpy.data.objects.new(L_["name"], ld)
    scene.collection.objects.link(ob)
    ob.location = L_["at"]
    ob.rotation_euler = (Vector(L_["aim"]) - Vector(L_["at"])).to_track_quat("-Z", "Y").to_euler()

world = bpy.data.worlds.new("World")
scene.world = world
world.node_tree.nodes["Background"].inputs["Color"].default_value = (*lin(S["world"]["color"]), 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = S["world"]["strength"]

# --- backdrop seen only by the camera: a vertical gradient with a warm bloom
BD = S["backdrop"]
bpy.ops.mesh.primitive_plane_add(size=1, location=(0, BD["distance"], 0), rotation=(math.radians(90), 0, 0))
back = bpy.context.object
back.scale = (2.2 * TAN_H * BD["distance"], 2.2 * TAN_V * BD["distance"], 1)
back.visible_diffuse = back.visible_glossy = back.visible_transmission = back.visible_volume_scatter = back.visible_shadow = False
bm = bpy.data.materials.new("M_Backdrop")
bnt = bm.node_tree
bnt.nodes.remove(bnt.nodes["Principled BSDF"])
em = node(bnt, "ShaderNodeEmission", (0, 0))
btc = node(bnt, "ShaderNodeTexCoord", (-1000, 0))
sep = node(bnt, "ShaderNodeSeparateXYZ", (-800, 0))
bnt.links.new(btc.outputs["Generated"], sep.inputs[0])
vr = node(bnt, "ShaderNodeValToRGB", (-550, 150))
bnt.links.new(sep.outputs["Y"], vr.inputs[0])
els = vr.color_ramp.elements
els[0].position, els[0].color = 0.25, (*lin(BD["low"]), 1)
els[1].position, els[1].color = 0.8, (*lin(BD["high"]), 1)
gl = node(bnt, "ShaderNodeVectorMath", (-550, -150), operation="DISTANCE")
bnt.links.new(btc.outputs["Generated"], gl.inputs[0])
gl.inputs[1].default_value = (*BD["bloom_at"], 0.5)
gr = node(bnt, "ShaderNodeMapRange", (-350, -150))
gr.inputs["From Min"].default_value = BD["bloom_radius"]
gr.inputs["From Max"].default_value = 0.0
bnt.links.new(gl.outputs["Value"], gr.inputs["Value"])
mx = node(bnt, "ShaderNodeMix", (-150, 0), data_type="RGBA", blend_type="MIX")
bnt.links.new(gr.outputs[0], mx.inputs[0])
bnt.links.new(vr.outputs["Color"], mx.inputs[6])
mx.inputs[7].default_value = (*lin(BD["bloom"]), 1)
bnt.links.new(mx.outputs[2], em.inputs["Color"])
bnt.links.new(em.outputs[0], bnt.nodes["Material Output"].inputs["Surface"])
back.data.materials.append(bm)

# --- atmosphere: a thin haze between the camera and the backdrop, so the far forms and the glow soften into the air
AT = S.get("atmosphere")
if AT and AT.get("density", 0) > 0 and not (CLAY or PROJ):
    d0, d1 = AT.get("from", 1.0), AT.get("to", BD["distance"] - 1)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, (d0 + d1) / 2, 0))
    haze = bpy.context.object
    haze.scale = (1.2 * TAN_H * d1, (d1 - d0) / 2, 1.2 * TAN_V * d1)
    hm = bpy.data.materials.new("M_Haze")
    hnt = hm.node_tree
    hnt.nodes.remove(hnt.nodes["Principled BSDF"])
    vol = node(hnt, "ShaderNodeVolumePrincipled", (0, 0))
    vol.inputs["Color"].default_value = (*lin(AT.get("color", "#ffffff")), 1)
    vol.inputs["Density"].default_value = AT["density"]
    vol.inputs["Anisotropy"].default_value = AT.get("anisotropy", 0.3)
    hnt.links.new(vol.outputs[0], hnt.nodes["Material Output"].inputs["Volume"])
    haze.data.materials.append(hm)

# --- focus on the nearest form around the settings' focus point; the far forms melt
fx, fy = CAM["focus_at"]
c0, r0 = int(GW * fx), int(GH * fy)
near_d = float(NEAR + (1 - D[r0 - 20:r0 + 20, c0 - 20:c0 + 20].max()) * (FAR - NEAR))
focus_hit = None
if MB and not PROJ:
    # the photo's depth map does not describe the generated mesh: focus where the camera ray through the point meets it
    # only the mesh may stop the ray: the hidden depth surface, the haze box and the backdrop are taken out of the scene view
    for ob in [forms, back, *tube_objs] + ([haze] if "haze" in globals() else []):
        ob.hide_viewport = True
    bpy.context.view_layer.update()
    ray = Vector(((2 * fx - 1) * TAN_H, 1.0, (1 - 2 * fy) * TAN_V)).normalized()
    hit, loc_, *_ = scene.ray_cast(bpy.context.view_layer.depsgraph, Vector((0, 0, 0)), ray)
    focus_hit = bool(hit)
    if hit:
        near_d = loc_.y
    else:
        print(f"WARNING focus ray at {fx},{fy} missed the mesh; focusing by the photo depth map ({near_d:.2f})",
              file=sys.stderr, flush=True)
cd.dof.use_dof = not PROJ
near_d = CAM.get("focus_distance", near_d)
cd.dof.focus_distance = near_d
cd.dof.aperture_fstop = CAM["fstop"]

# --- render
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
device = "NONE"
for kind in () if args.device == "cpu" else ("OPTIX", "CUDA"):
    try:
        prefs.compute_device_type = kind
        prefs.get_devices()
        if [d for d in prefs.devices if d.type == kind]:
            for d in prefs.devices:
                d.use = d.type == kind
            device = kind
            break
    except TypeError:
        continue
if device == "NONE" and args.device != "cpu":
    print("NOTICE no NVIDIA card found by Blender (OptiX or CUDA); rendering on the processor, which is much slower",
          file=sys.stderr, flush=True)
scene.cycles.device = "GPU" if device != "NONE" else "CPU"
scene.cycles.samples = args.samples or (48 if CLAY else 24 if PROJ else REN["samples"])
scene.cycles.use_denoising = not PROJ
# NVIDIA denoiser with colour, albedo and normal guides: clean before any grade can stretch noise
scene.cycles.denoiser = "OPTIX" if device == "OPTIX" else "OPENIMAGEDENOISE"
scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
scene.cycles.max_bounces = 8
scale = args.scale or REN["scale"]
scene.render.resolution_x = int(W * scale)
scene.render.resolution_y = int(H * scale)
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard" if PROJ else "AgX"
try:
    scene.view_settings.look = "None" if PROJ else REN["look"]
except TypeError:
    pass
scene.render.image_settings.file_format = "PNG"
# 16-bit so a later grade stretches smooth gradients, not 8-bit steps
scene.render.image_settings.color_depth = "16"
scene.render.image_settings.color_mode = "RGB"

# --- glow on the highlights (compositor glare), the soft bloom of the warm light through the forms
GL = S.get("glare")
if GL and GL.get("strength", 0) > 0 and not (CLAY or PROJ):
    # Blender 5: the compositor is a node group with a group output, and the glare settings are input sockets
    cng = bpy.data.node_groups.new("Comp", "CompositorNodeTree")
    cng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = cng.nodes.new("CompositorNodeRLayers")
    gn = cng.nodes.new("CompositorNodeGlare")
    gn.inputs["Type"].default_value = GL.get("type", "Bloom")
    gn.inputs["Quality"].default_value = "High"
    gn.inputs["Threshold"].default_value = GL.get("threshold", 0.8)
    gn.inputs["Strength"].default_value = GL["strength"]
    gn.inputs["Size"].default_value = GL.get("size", 0.6)
    out_ = cng.nodes.new("NodeGroupOutput")
    cng.links.new(rl.outputs["Image"], gn.inputs["Image"])
    cng.links.new(gn.outputs["Image"], out_.inputs[0])
    scene.compositing_node_group = cng
    scene.render.use_compositing = True

if args.stage == "anim":
    os.makedirs(args.out, exist_ok=True)
    scene.frame_start, scene.frame_end = args.start, (args.end if args.end >= 0 else FRAMES - 1)
    scene.render.filepath = os.path.join(args.out, "f_")
    bpy.ops.render.render(animation=True)
    res = {"stage": "anim", "device": device, "frames": [scene.frame_start, scene.frame_end]}
else:
    scene.frame_set(args.frame)
    pose(args.frame)
    scene.render.filepath = args.out
    bpy.ops.render.render(write_still=True)
    res = {"stage": args.stage, "device": device, "focus": round(near_d, 2), "focus_hit": focus_hit, "verts": n}
print("RESULT " + json.dumps(res))
