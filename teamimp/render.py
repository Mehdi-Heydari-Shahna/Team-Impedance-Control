"""Off-screen rendering helpers (GLFW on Windows, EGL elsewhere)."""
import os
os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
import numpy as np
import mujoco


def free_cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c


class Snap:
    def __init__(self, m, width=960, height=540, shadows=True):
        self.m = m
        self.d = mujoco.MjData(m)
        self.r = mujoco.Renderer(m, height, width)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
        self.opt.sitegroup[:] = [0, 0, 0, 0, 0, 0]
        if not shadows:
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0

    def __call__(self, q, cam, forward=True, d=None):
        d = d if d is not None else self.d
        if q is not None:
            d.qpos[:] = q
            if forward:
                mujoco.mj_forward(self.m, d)
        self.r.update_scene(d, cam, self.opt)
        return self.r.render().copy()

    def close(self):
        self.r.close()


def grid(imgs, cols):
    rows = [np.concatenate(imgs[k:k + cols], axis=1) for k in range(0, len(imgs), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    return np.concatenate(rows, axis=0)
