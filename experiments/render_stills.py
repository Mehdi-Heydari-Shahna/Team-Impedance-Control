"""Render the snapshot strip of the N=4 mission for the paper (PNG, cropped)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
import numpy as np
from PIL import Image

from teamimp import mission as MS, render

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results", "mission")
OUT = os.path.join(ROOT, "paper", "figures")


def stills(tag, times, cam_fn, size=(900, 760), prefix="snap", crop=None):
    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(RES, f"{tag}.npz"), allow_pickle=True)
    M = MS.Mission(MS.Cfg(n=int(d["n"][0]), verbose=False, background=True))
    S = render.Snap(M.m, *size)
    paths = []
    for k, t in enumerate(times):
        j = int(np.clip(np.searchsorted(d["t"], t), 0, len(d["t"]) - 1))
        img = S(d["qpos"][j], cam_fn(t, d["beam"][j]))
        im = Image.fromarray(img)
        if crop is not None:
            im = im.crop(crop)
        p = os.path.join(OUT, f"{prefix}_{k}.png")
        im.save(p)
        paths.append(p)
        print("saved", p, "t =", d["t"][j], d["phase"][j], flush=True)
    return paths


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "al_n4_kz0"
    times = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [9.6, 25.2, 31.4, 33.8, 76.0, 85.2]

    az = float(sys.argv[3]) if len(sys.argv) > 3 else 145.0
    el = float(sys.argv[4]) if len(sys.argv) > 4 else -16.0
    dist = float(sys.argv[5]) if len(sys.argv) > 5 else 4.9

    def cam(t, beam):
        look = np.array([0.0, 0.5, 0.74])
        return render.free_cam(look, dist, az, el)
    stills(tag, times, cam, size=(760, 640), crop=(60, 160, 700, 545))
