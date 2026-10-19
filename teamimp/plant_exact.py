"""Exact compiled-model plant: M vdot + C v + g + D v = B f + A^T lam + J_o^T h, A v = 0 (ideal connections).

Integrated with RK4 (the controller is evaluated at every stage, i.e., in continuous time) or with a
zero-order hold of period T_s. Constraint drift is removed by projection after each step.
"""
import numpy as np
import scipy.linalg as sla
import mujoco

from .so3 import log


class ExactPlant:
    def __init__(self, tm, alpha=30.0):
        self.tm = tm                 # TeamModel evaluated on the *true* model
        self.alpha = alpha

    def constraint_error(self, q):
        """phi with d(phi)/dt = A v (to first order), stacked like the rows of A."""
        tm = self.tm
        m, d = tm.m, tm.d
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        out = []
        for kind, i, s in tm.rows:
            r = tm.robots[i]
            if kind == "foot":
                sid = r.site[("foot", s)]
                p, R = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
                ps, Rs = tm.mode.feet[(i, s)]
                out.append(np.r_[R.T @ (p - ps), log(Rs.T @ R)])
            else:
                sid = r.site[("hand", s)]
                ph, Rh = d.site_xpos[sid], d.site_xmat[sid].reshape(3, 3)
                po, Ro = d.xpos[tm.beam], d.xmat[tm.beam].reshape(3, 3)
                pk_o, Rk_o = tm.mode.grasps[(i, s)]
                pk, Rk = po + Ro @ pk_o, Ro @ Rk_o
                out.append(np.r_[Rk.T @ (pk - ph), log(Rh.T @ Rk)])
        return np.concatenate(out) if out else np.zeros(0)

    def env_gen_force(self, h):
        tm = self.tm
        Q = np.zeros(tm.n)
        if tm.mode.payload:
            c = tm.col[tm.beam_dof]
            Q[c:c + 3] = h[:3]
            Q[c + 3:c + 6] = tm.Ro.T @ h[3:]
        return Q

    def accel(self, q, v, f, h, phi=None):
        """Forward dynamics at (q, v) for effort f. Assumes tm.update(q, v) has been called."""
        tm = self.tm
        rhs = tm.B @ f + self.env_gen_force(h) - tm.bias - tm.Dv @ tm.v
        cM = sla.cho_factor(tm.M)
        Minv_rhs = sla.cho_solve(cM, rhs)
        MinvAt = sla.cho_solve(cM, tm.A.T)
        S = tm.A @ MinvAt
        b = tm.Ad @ tm.v + 2 * self.alpha * (tm.A @ tm.v)
        if phi is not None:
            b = b + self.alpha ** 2 * phi
        lam = -np.linalg.solve(S, tm.A @ Minv_rhs + b)
        vdot = Minv_rhs + MinvAt @ lam
        return vdot, lam

    def project(self, q, v, iters=3):
        tm = self.tm
        for _ in range(iters):
            phi = self.constraint_error(q)
            if phi.size == 0 or np.abs(phi).max() < 1e-13:
                break
            tm.update(q, v, dyn=False, need_dot=False)
            dq = -np.linalg.lstsq(tm.A, phi, rcond=None)[0]
            dv = np.zeros(tm.m.nv)
            dv[tm.sys] = dq
            mujoco.mj_integratePos(tm.m, q, dv, 1.0)
        tm.update(q, v, dyn=True, need_dot=False)
        if tm.r:
            cM = sla.cho_factor(tm.M)
            MinvAt = sla.cho_solve(cM, tm.A.T)
            vs = tm.v - MinvAt @ np.linalg.solve(tm.A @ MinvAt, tm.A @ tm.v)
            v = v.copy()
            v[tm.sys] = vs
        return q, v
