"""Decimate the Apollo CAD meshes (visual fidelity kept, memory/render cost reduced ~10x)."""
import os
import re
import shutil
import numpy as np
import trimesh
import fast_simplification

SRC = os.environ.get("APOLLO_SRC", os.path.join("mujoco_menagerie", "apptronik_apollo"))   # Menagerie checkout
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "apollo")


def decimate_all(target_faces=6000, min_keep=0.06):
    os.makedirs(os.path.join(DST, "assets", "ability_hand"), exist_ok=True)
    stats = []
    for root, _, files in os.walk(os.path.join(SRC, "assets")):
        for f in files:
            if not f.lower().endswith(".stl"):
                continue
            src = os.path.join(root, f)
            rel = os.path.relpath(src, os.path.join(SRC, "assets"))
            dst = os.path.join(DST, "assets", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            mesh = trimesh.load(src, force="mesh", process=True)
            nf = len(mesh.faces)
            if nf > target_faces:
                red = max(0.0, min(0.97, 1.0 - max(target_faces / nf, min_keep)))
                v, fc = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32),
                                                     np.asarray(mesh.faces, np.int32), target_reduction=red, agg=3)
                out = trimesh.Trimesh(v, fc, process=True)
                out.fix_normals()
            else:
                out = mesh
            out.export(dst)
            stats.append((rel, nf, len(out.faces)))
    xml = open(os.path.join(SRC, "apptronik_apollo.xml")).read()
    open(os.path.join(DST, "apptronik_apollo.xml"), "w").write(xml)
    for f in os.listdir(SRC):
        if f.endswith(".png") or f == "LICENSE":
            shutil.copy(os.path.join(SRC, f), DST)
    return stats


if __name__ == "__main__":
    st = decimate_all()
    print(sum(s[1] for s in st), "->", sum(s[2] for s in st), "faces")
    for s in sorted(st, key=lambda x: -x[1])[:10]:
        print(s)
