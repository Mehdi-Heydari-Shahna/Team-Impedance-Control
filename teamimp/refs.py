"""Reference generators."""
import numpy as np

from . import scene
from .so3 import exp, log, minjerk, hat
from .controller import Refs


K_RATE = dict(
    torso_yaw=60.0, torso_roll=60.0, torso_pitch=60.0, neck_yaw=1.0, neck_roll=1.0, neck_pitch=1.0,
    l_shoulder_aa=12.0, l_shoulder_ie=8.0, l_shoulder_fe=12.0, l_elbow_fe=10.0, l_wrist_roll=1.5,
    l_wrist_yaw=1.5, l_wrist_pitch=1.5,
    r_shoulder_aa=12.0, r_shoulder_ie=8.0, r_shoulder_fe=12.0, r_elbow_fe=10.0, r_wrist_roll=1.5,
    r_wrist_yaw=1.5, r_wrist_pitch=1.5,
    l_hip_ie=40.0, l_hip_aa=60.0, l_hip_fe=60.0, l_knee_fe=50.0, l_ankle_ie=20.0, l_ankle_pd=25.0,
    r_hip_ie=40.0, r_hip_aa=60.0, r_hip_fe=60.0, r_knee_fe=50.0, r_ankle_ie=20.0, r_ankle_pd=25.0)


class PoseTraj:
    """Piecewise min-jerk pose trajectory through waypoints (t_k, p_k, R_k)."""

    def __init__(self, waypoints):
        self.w = waypoints

    def __call__(self, t):
        w = self.w
        if t <= w[0][0]:
            return w[0][1].copy(), w[0][2].copy(), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for k in range(len(w) - 1):
            t0, p0, R0 = w[k]
            t1, p1, R1 = w[k + 1]
            if t <= t1:
                T = t1 - t0
                s, ds, dds = minjerk((t - t0) / T)
                dp = p1 - p0
                phi = log(R1 @ R0.T)
                p = p0 + s * dp
                R = exp(s * phi) @ R0
                return p, R, ds / T * dp, ds / T * phi, dds / T ** 2 * dp, dds / T ** 2 * phi
        return w[-1][1].copy(), w[-1][2].copy(), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)


class VecTraj:
    def __init__(self, waypoints):
        self.w = waypoints      # (t_k, x_k)

    def __call__(self, t):
        w = self.w
        if t <= w[0][0]:
            z = np.zeros_like(w[0][1])
            return w[0][1].copy(), z, z
        for k in range(len(w) - 1):
            t0, x0 = w[k]
            t1, x1 = w[k + 1]
            if t <= t1:
                T = t1 - t0
                s, ds, dds = minjerk((t - t0) / T)
                return x0 + s * (x1 - x0), ds / T * (x1 - x0), dds / T ** 2 * (x1 - x0)
        z = np.zeros_like(w[-1][1])
        return w[-1][1].copy(), z, z


class HoldRefs(Refs):
    """References that hold an initial configuration, with an optional payload trajectory and internal wrench."""

    def __init__(self, tm, q0, payload_traj=None, lam_d=None):
        import mujoco
        d = tm.d
        d.qpos[:] = q0
        mujoco.mj_kinematics(tm.m, d)
        self.pel = {r.i: (d.xpos[r.pelvis].copy(), d.xmat[r.pelvis].reshape(3, 3).copy()) for r in tm.robots}
        self.jq = {}
        for r in tm.robots:
            names = scene.TORSO + scene.NECK + [scene.ARM_RED["l"], scene.ARM_RED["r"]]
            self.jq[r.i] = np.array([q0[r.qpos_adr[j]] for j in names])
        self.ees = {}
        for r in tm.robots:
            for (kind, s), sid in r.site.items():
                self.ees[(r.i, kind, s)] = (d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy())
        p0, R0 = d.xpos[tm.beam].copy(), d.xmat[tm.beam].reshape(3, 3).copy()
        self.ptraj = payload_traj or PoseTraj([(0.0, p0, R0)])
        self._lam = lam_d
        self.pelvis_offset = {}
        self.joint_traj = {}

    def payload(self, t):
        return self.ptraj(t)

    def pelvis(self, i, t):
        p, R = self.pel[i]
        z = np.zeros(3)
        if i in self.pelvis_offset:
            dp, dv, da = self.pelvis_offset[i](t)
            return p + dp, R, dv, z, da, z
        return p, R, z, z, z, z

    def joints(self, i, t):
        if i in self.joint_traj:
            return self.joint_traj[i](t)
        z = np.zeros(8)
        return self.jq[i], z, z

    def ee(self, i, kind, s, t):
        p, R = self.ees[(i, kind, s)]
        z = np.zeros(3)
        return p, R, z, z, z, z

    def lam_d(self, t):
        if self._lam is None:
            return None
        return self._lam(t) if callable(self._lam) else self._lam


def squeeze_pattern(tm, F):
    """Normal push F (N) at every closed grasp (grasp frame z points into the payload), projected on N(G)."""
    lam = np.zeros(6 * tm.ng)
    for j in range(tm.ng):
        lam[6 * j + 2] = F
    return tm.PI @ lam
