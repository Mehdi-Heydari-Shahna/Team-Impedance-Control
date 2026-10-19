"""MuJoCo contact-physics plant driven by the two-loop controller (compiled support/grasp modes).

The plant has soft contacts (soles-ground, palms-girder, girder-trestles), joint friction, damping,
armature and actuator torque limits. The controller uses the compiled rigid-connection model of the current
mode. Actuators: force = ctrl - k_i qdot_i (the rate-gain term of (5) acts continuously), clipped to the
torque limits.
"""
import time
import numpy as np
import mujoco

from . import scene, team, controller
from .so3 import hat


class MjRunner:
    def __init__(self, n, k_rate, spec_kw=None, ctrl_kw=None, dt_ctrl=0.002, h_filter=60.0, payload_scale=1.0,
                 ctrl_model_scale=None):
        spec_kw = dict(spec_kw or {})
        self.n = n
        sp = scene.build_spec(n, "plant", rate_gains=k_rate, **spec_kw)
        self.m = sp.compile()
        self.d = mujoco.MjData(self.m)
        ck = dict(spec_kw)
        ck.pop("supports", None)
        ck.pop("background", None)
        if payload_scale != 1.0:
            ck["mass"] = (ck.get("mass") or scene.PROFILES[ck.get("profile", "AL 280")][4] * ck.get("beam_len", 2.4)) \
                * payload_scale
        sc = scene.build_spec(n, "ctrl", **ck)
        self.mc = sc.compile()
        self.mc.dof_frictionloss[:] = 0.0
        if ctrl_model_scale is not None:
            ctrl_model_scale(self.mc)
        self.tm = team.TeamModel(self.mc, n)
        self.k_rate = k_rate
        self.K = np.array([k_rate[j] for _ in range(n) for j in scene.JOINTS])
        self.dt = self.m.opt.timestep
        self.dt_ctrl = dt_ctrl
        self.nsub = int(round(dt_ctrl / self.dt))
        self.h_filter = h_filter
        self.h_f = np.zeros(6)
        # geom bookkeeping
        m = self.m
        self.beam = m.body("beam").id
        self.beam_geoms = {m.geom(nm).id for nm in ("beam_bot", "beam_top", "beam_web")}
        self.palm_geom = {}
        self.sole_geom = {}
        for i in range(n):
            for s in "lr":
                self.palm_geom[m.geom(f"r{i}_collision_{s}_hand_plate").id] = (i, s)
                self.sole_geom[m.geom(f"r{i}_collision_{s}_sole").id] = (i, s)
        self.env_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        self.act_dofs = np.array([self.tm.robots[i].dof[j] for i in range(n) for j in scene.JOINTS])
        self.frc_lim = m.actuator_forcerange[:, 1].copy()
        self.ctrl = None
        self.noise = None                 # optional sensor noise (sigma_force N, sigma_moment N m) on palm, sole and h
        self.rng = np.random.default_rng(0)
        self.blend_tau = 0.0              # bumpless transfer at mode switches (0: off)
        self.blend_pending = False
        self.blend_delta = None
        self.f_applied = None
        self.log = []
        self.events = []
        self.t = 0.0

    # ------------------------------------------------------------------ contact sensing
    def contact_wrenches(self):
        """Per-palm and per-sole contact wrenches (world force on the payload/foot... see below) and the
        environment wrench on the girder (world force, moment about the girder origin)."""
        m, d = self.m, self.d
        palm = {}
        sole = {}
        h = np.zeros(6)
        pb = d.xpos[self.beam]
        c6 = np.zeros(6)
        for k in range(d.ncon):
            con = d.contact[k]
            g1, g2 = con.geom1, con.geom2
            mujoco.mj_contactForce(m, d, k, c6)
            fr = con.frame.reshape(3, 3)            # rows: normal, tangent1, tangent2 (world)
            f_w = fr.T @ c6[:3]                     # force on geom2 from geom1 (world)
            t_w = fr.T @ c6[3:]
            p = con.pos
            if g1 in self.beam_geoms or g2 in self.beam_geoms:
                # force on the girder
                fb = f_w if g2 in self.beam_geoms else -f_w
                tb = t_w if g2 in self.beam_geoms else -t_w
                other = g1 if g2 in self.beam_geoms else g2
                if other in self.palm_geom:
                    key = self.palm_geom[other]
                    w = palm.setdefault(key, [np.zeros(3), np.zeros(3), []])
                    w[0] += fb
                    w[1] += tb
                    w[2].append((p.copy(), fb.copy(), tb.copy()))
                else:
                    h[:3] += fb
                    h[3:] += np.cross(p - pb, fb) + tb
            elif g1 in self.sole_geom or g2 in self.sole_geom:
                ff = f_w if g2 in self.sole_geom else -f_w      # force on the foot
                tf = t_w if g2 in self.sole_geom else -t_w
                key = self.sole_geom[g2] if g2 in self.sole_geom else self.sole_geom[g1]
                w = sole.setdefault(key, [np.zeros(3), np.zeros(3), []])
                w[0] += ff
                w[1] += tf
                w[2].append((p.copy(), ff.copy(), tf.copy()))
        if getattr(self, "noise", None) is not None:
            # white measurement noise of each force/torque sensor (palms and soles in contact, and h); the
            # noise-free wrenches are kept for evaluation
            self._true = dict(palm={k: [w[0].copy(), w[1].copy(), list(w[2])] for k, w in palm.items()},
                              sole={k: [w[0].copy(), w[1].copy(), list(w[2])] for k, w in sole.items()},
                              h=h.copy())
            sf, sm = self.noise
            for dct in (palm, sole):
                for w in dct.values():
                    p0, f0, t0 = w[2][0]
                    nf, nm = sf * self.rng.standard_normal(3), sm * self.rng.standard_normal(3)
                    w[2][0] = (p0, f0 + nf, t0 + nm)
                    w[0] += nf
                    w[1] += nm
            h = h + np.r_[sf * self.rng.standard_normal(3), sm * self.rng.standard_normal(3)]
        return palm, sole, h

    def grasp_wrench(self, palm, i, s):
        """Wrench applied by hand (i, s) on the girder in the grasp frame, moment about the grasp point."""
        tm = self.tm
        if (i, s) not in palm:
            return np.zeros(6)
        po, Ro = tm.mode.grasps[(i, s)]
        pb = self.d.xpos[self.beam]
        Rb = self.d.xmat[self.beam].reshape(3, 3)
        pk, Rk = pb + Rb @ po, Rb @ Ro
        F = np.zeros(3)
        T = np.zeros(3)
        for (p, f, tq) in palm[(i, s)][2]:
            F += f
            T += np.cross(p - pk, f) + tq
        return np.r_[Rk.T @ F, Rk.T @ T]

    def foot_wrench(self, sole, i, s):
        """Ground wrench on the foot in the sole frame (moment about the sole site)."""
        if (i, s) not in sole:
            return np.zeros(6)
        sid = self.m.site(f"r{i}_{s}_sole").id
        ps = self.d.site_xpos[sid]
        Rs = self.d.site_xmat[sid].reshape(3, 3)
        F = np.zeros(3)
        T = np.zeros(3)
        for (p, f, tq) in sole[(i, s)][2]:
            F += f
            T += np.cross(p - ps, f) + tq
        return np.r_[Rs.T @ F, Rs.T @ T]

    # ------------------------------------------------------------------ control
    def control_step(self):
        tm, ctrl, d = self.tm, self.ctrl, self.d
        q, v = d.qpos.copy(), d.qvel.copy()
        tm.update(q, v)
        palm, sole, h = self.contact_wrenches()
        a = 1.0 - np.exp(-self.h_filter * self.dt_ctrl)
        self.h_f = self.h_f + a * (h - self.h_f)
        h_use = self.h_f if tm.mode.payload else np.zeros(6)
        f = ctrl.effort(self.t, h_use)
        # internal impedance and external impedance states (explicit Euler at the control rate)
        if tm.ng:
            lam_g = np.concatenate([self.grasp_wrench(palm, i, s) for (kind, i, s) in tm.rows if kind == "grasp"])
        else:
            lam_g = np.zeros(0)
        # anti-windup of the internal impedance: while an actuator is at its torque limit (applied torques of the
        # last plant step), no new internal-wrench error is integrated
        ctrl.hold_integration = bool(np.max(self.saturation()) >= 0.999)
        xd = ctrl.state_derivs(lam_g)
        ctrl.unpack(ctrl.pack() + self.dt_ctrl * xd)
        ctrl.clamp()
        # bumpless transfer at mode switches (outside the per-mode theory): the jump of the effort at a switch
        # decays with time constant blend_tau
        f_cmd = f
        if getattr(self, "blend_pending", False):
            self.blend_delta = self.f_applied - f if getattr(self, "f_applied", None) is not None else np.zeros_like(f)
            self.blend_t0 = self.t
            self.blend_pending = False
        if getattr(self, "blend_delta", None) is not None and self.blend_tau > 0:
            a_b = np.exp(-(self.t - self.blend_t0) / self.blend_tau)
            if a_b > 1e-3:
                f_cmd = f + a_b * self.blend_delta
            else:
                self.blend_delta = None
        self.f_applied = f_cmd.copy()
        u = f_cmd + self.K * v[self.act_dofs]
        d.ctrl[:] = u
        self._last = dict(f=f, lam_g=lam_g, h=h, h_f=self.h_f.copy(), palm=palm, sole=sole)
        return f

    def step(self, nctrl=1, callback=None):
        for _ in range(nctrl):
            self.control_step()
            for _ in range(self.nsub):
                mujoco.mj_step(self.m, self.d)
            self.t += self.dt_ctrl
            if callback is not None:
                callback(self)

    def applied_torque(self):
        return self.d.actuator_force.copy()

    def saturation(self):
        return np.abs(self.d.actuator_force) / self.frc_lim
