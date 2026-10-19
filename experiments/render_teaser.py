"""First-page figure: the four humanoids carrying the girder, rendered without background (RGBA PNG).

The scene geoms of the world body (ground, trestles) are hidden, the skybox is off, and a segmentation pass
gives the alpha mask; the image is cropped to the robots and the girder.
Usage: python experiments/render_teaser.py [tag] [time_s] [azimuth] [elevation] [distance]
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
import numpy as np
import mujoco
from PIL import Image

from teamimp import mission as MS, render

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results", "mission")
OUT = os.path.join(ROOT, "paper", "figures")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "al_n4_kz0"
    t_show = float(sys.argv[2]) if len(sys.argv) > 2 else 43.8
    az = float(sys.argv[3]) if len(sys.argv) > 3 else 62.0
    el = float(sys.argv[4]) if len(sys.argv) > 4 else -22.0
    dist = float(sys.argv[5]) if len(sys.argv) > 5 else 4.9
    W, H = 3600, 2250
    d = np.load(os.path.join(RES, f"{tag}.npz"), allow_pickle=True)
    M = MS.Mission(MS.Cfg(n=int(d["n"][0]), verbose=False, background=False))
    m = M.m
    # hide everything attached to the world body (ground plane, trestles, pads)
    hide = 5
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            m.geom_group[g] = hide
    m.vis.quality.offsamples = 8
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    j = int(np.clip(np.searchsorted(d["t"], t_show), 0, len(d["t"]) - 1))
    q = d["qpos"][j]
    data = mujoco.MjData(m)
    data.qpos[:] = q
    mujoco.mj_forward(m, data)
    look = data.xpos[m.body("beam").id].copy()
    look[2] -= 0.25
    cam = render.free_cam(look, dist, az, el)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    opt.sitegroup[:] = [0] * 6
    r = mujoco.Renderer(m, H, W)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    r.update_scene(data, cam, opt)
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(data, cam, opt)
    seg = r.render().copy()
    r.disable_segmentation_rendering()
    r.close()
    alpha = (seg[:, :, 0] >= 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(alpha)
    pad = 20
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    rgba = np.dstack([rgb, alpha])[y0:y1, x0:x1]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_teaser.png")
    Image.fromarray(rgba, "RGBA").save(path)
    print("saved", path, rgba.shape, "t =", float(d["t"][j]), d["phase"][j], flush=True)


if __name__ == "__main__":
    main()
