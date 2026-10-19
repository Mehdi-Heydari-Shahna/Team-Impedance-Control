"""Common set-up: team configuration, modes, gains, environment models."""
import numpy as np
import mujoco

from . import scene, team, ik, refs, controller

GAIN_SCALE = dict(leg=4.0, arm=2.0, torso=2.0)


def rate_gains(scale=GAIN_SCALE):
    kr = dict(refs.K_RATE)
    for j in kr:
        if "hip" in j or "knee" in j or "ankle" in j:
            kr[j] *= scale["leg"]
        elif "shoulder" in j or "elbow" in j:
            kr[j] *= scale["arm"]
        elif "torso" in j:
            kr[j] *= scale["torso"]
    return kr


def ctrl_model(n, mass=135.0, frictionless=True, inertia_scale=None, payload_scale=1.0, seed=0):
    """Controller/exact-plant model. inertia_scale: relative std of random link mass/inertia errors."""
    s, m = scene.build(n, "ctrl", mass=mass)
    if frictionless:
        m.dof_frictionloss[:] = 0.0
    if inertia_scale is not None or payload_scale != 1.0:
        rng = np.random.default_rng(seed)
        beam = m.body("beam").id
        for b in range(1, m.nbody):
            if b == beam:
                f = payload_scale
            elif inertia_scale is not None:
                f = float(np.clip(1.0 + inertia_scale * rng.standard_normal(), 0.5, 1.5))
            else:
                f = 1.0
            m.body_mass[b] *= f
            m.body_inertia[b] *= f
    return m


def grasp_mode(tm, q, feet=True, grasps=True):
    d = mujoco.MjData(tm.m)
    d.qpos[:] = q
    mujoco.mj_kinematics(tm.m, d)
    pb, Rb = d.xpos[tm.beam].copy(), d.xmat[tm.beam].reshape(3, 3).copy()
    F, G = {}, {}
    for r in tm.robots:
        for s in "lr":
            if feet:
                sid = r.site[("foot", s)]
                F[(r.i, s)] = (d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy())
            if grasps:
                sid = r.site[("hand", s)]
                ph, Rh = d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()
                G[(r.i, s)] = (Rb.T @ (ph - pb), Rb.T @ Rh)
    return team.Mode(F, G)


def team_hold_config(m, tm, n, z=0.86, y_off=0.44, hand_dx=0.30, beam_xy=(0.0, 0.0)):
    lay = scene.Layout(n, y_off=y_off)
    q, info = ik.team_config(m, tm, np.array([beam_xy[0], beam_xy[1], z]), lay, iters=600, hand_dx=hand_dx)
    return q, lay


def default_params(tm, **kw):
    p = dict(k_rate=rate_gains(), gp=4.0, go=4.0, gb_p=4.0, gb_o=4.0, gj=4.0,
             Mc=[150.0] * 3 + [20.0] * 3, Dc=[1900.0] * 3 + [270.0] * 3, Kp=[6000.0] * 3, Ko=[900.0] * 3,
             mz=0.03, dz=0.35, kz=0.0)
    p.update(kw)
    return controller.Params(tm, **p)


class SpringSupport:
    """Remark 4 environment: h = (-Ke (p - pe) - De pdot, 0), engaged from t_on (bilateral model)."""

    def __init__(self, Ke, De, t_on=np.inf):
        self.Ke, self.De = np.diag(Ke), np.diag(De)
        self.t_on = t_on
        self.pe = None
        self.t = 0.0

    def __call__(self, tm):
        if self.t < self.t_on or not tm.mode.payload:
            return np.zeros(6)
        p, R, v, w = tm.payload_state()
        if self.pe is None:
            self.pe = p.copy()
        return np.r_[-self.Ke @ (p - self.pe) - self.De @ v, np.zeros(3)]
