"""Simulation driver for the exact compiled-model plant (validation of Theorems 1-2, Proposition 1)."""
import time
import numpy as np
import scipy.linalg as sla
import mujoco

from .plant_exact import ExactPlant


class ExactSim:
    def __init__(self, tm_plant, ctrl, env=None, tm_ctrl=None, Ts=None, log_every=1, diag_every=1):
        self.tp = tm_plant
        self.tc = tm_ctrl if tm_ctrl is not None else tm_plant
        self.ctrl = ctrl
        self.plant = ExactPlant(tm_plant)
        self.env = env if env is not None else (lambda tm: np.zeros(6))
        self.Ts = Ts                    # None: continuous-time controller; else zero-order hold period
        self.log_every = log_every
        self.diag_every = diag_every
        self.held = None
        self.log = {}

    def _deriv(self, t, q, v, x, f_hold=None, diag=False):
        tp, tc, ctrl = self.tp, self.tc, self.ctrl
        ctrl.unpack(x)
        tp.update(q, v)
        if tc is not tp:
            tc.update(q, v)
        h = self.env(tp)
        if f_hold is None:
            f = ctrl.effort(t, h)
        else:
            f = f_hold
            if tc.mode.payload:
                ctrl._ext_cache = ctrl.ext_derivs(t, h)
        vdot, lam = self.plant.accel(q, v, f, h)
        xdot = ctrl.state_derivs(lam[tp.gidx]) if tp.ng else ctrl.state_derivs(np.zeros(0))
        out = dict(vdot=vdot, xdot=xdot, f=f, lam=lam, h=h)
        if diag and f_hold is None:
            out["diag"] = self.diagnostics(vdot, lam)
        return out

    def diagnostics(self, vdot, lam):
        """Residuals of the identities of Theorem 1 (exact model)."""
        tm, c = self.tc, self.ctrl.last
        K = self.ctrl.K
        s = tm.v - c["vr"]
        sd = vdot - c["vr_dot"]
        BK = tm.B * K
        BKBs = BK @ (tm.B.T @ s)
        Cs = tm.coriolis(s)
        Ds = tm.Dv @ s
        Ms = tm.M @ s
        dV_id = s @ (tm.M @ sd) + s @ Cs           # d/dt (1/2 s^T M s)
        diss = s @ (BKBs + Ds)
        res_a = dV_id + diss
        cM = sla.cho_factor(tm.M)
        MinvAt = sla.cho_solve(cM, tm.A.T)
        S = tm.A @ MinvAt
        Xis = np.linalg.solve(S, tm.A @ sla.cho_solve(cM, Cs + Ds + BKBs) - tm.Ad @ s)
        mu = tm.Mmu @ (K * c["wperp"] + c["f_int"] + c["fF"])
        res_b = lam - c["lam_r"] + mu - Xis
        out = dict(V=0.5 * s @ Ms, sM=np.sqrt(max(s @ Ms, 0)), res_a=res_a, diss=diss, dV=dV_id,
                   res_b=np.abs(res_b).max(), lam_scale=np.abs(lam).max())
        if tm.ng:
            g = tm.gidx
            lamI = tm.PI @ lam[g]
            Psi_Kw = -(tm.Mmu @ (K * c["wperp"]))[g]
            Theta_s = tm.PI @ Xis[g]
            res_I = lamI - (c["psi"] + Psi_Kw + Theta_s)
            G = tm.grasp_matrix()
            res_c = G @ (lam[g] - c["lam_r"][g] - Xis[g])
            eI = tm.VIp @ (lamI - c["lam_d"])
            out.update(res_I=np.abs(res_I).max(), res_c=np.abs(res_c).max(), lamI=lamI, eI=eI,
                       eI_norm=np.linalg.norm(eI), Xis=Xis, Gl=G @ lam[g])
        return out

    def run(self, q0, v0, t0, T, dt, callback=None, verbose=True):
        tp, ctrl = self.tp, self.ctrl
        m = tp.m
        q, v = q0.copy(), v0.copy()
        t = t0
        x = ctrl.pack()
        n = int(round((T - t0) / dt))
        logs = {k: [] for k in ["t", "q", "v", "f", "lam", "x", "h"]}
        diags = []
        tic = time.time()
        f_hold = None
        next_sample = t0
        for k in range(n):
            if self.Ts is not None and t >= next_sample - 1e-12:
                tp.update(q, v)
                if self.tc is not tp:
                    self.tc.update(q, v)
                ctrl.unpack(x)
                f_hold = ctrl.effort(t, self.env(tp))
                next_sample += self.Ts
            do_diag = (k % self.diag_every == 0)
            r1 = self._deriv(t, q, v, x, f_hold if self.Ts is not None else None, diag=do_diag)
            if k % self.log_every == 0:
                logs["t"].append(t)
                logs["q"].append(q.copy())
                logs["v"].append(v.copy())
                logs["f"].append(r1["f"].copy())
                logs["lam"].append(r1["lam"].copy())
                logs["x"].append(x.copy())
                logs["h"].append(r1["h"].copy())
                d = r1.get("diag")
                if d is not None:
                    d = {kk: vv for kk, vv in d.items() if kk not in ("Xis",)}
                    d["t"] = t
                    d["rinfo"] = {str(kk): vv for kk, vv in ctrl.last["rinfo"].items()}
                    d["lam_d"] = ctrl.last["lam_d"].copy()
                    d["psi"] = ctrl.last["psi"].copy()
                    d["wperp"] = np.linalg.norm(ctrl.last["wperp"])
                    diags.append(d)
            S = tp.sys

            def adv(qq, vv, xx, rr, a):
                q2 = qq.copy()
                mujoco.mj_integratePos(m, q2, vv, a)
                v2 = v.copy()
                v2[S] = v[S] + a * rr["vdot"]
                return q2, v2, x + a * rr["xdot"]
            fh = f_hold if self.Ts is not None else None
            q2, v2, x2 = adv(q, v, x, r1, dt / 2)
            r2 = self._deriv(t + dt / 2, q2, v2, x2, fh)
            q3, v3, x3 = q.copy(), v.copy(), None
            mujoco.mj_integratePos(m, q3, v2, dt / 2)
            v3 = v.copy()
            v3[S] = v[S] + dt / 2 * r2["vdot"]
            x3 = x + dt / 2 * r2["xdot"]
            r3 = self._deriv(t + dt / 2, q3, v3, x3, fh)
            q4 = q.copy()
            mujoco.mj_integratePos(m, q4, v3, dt)
            v4 = v.copy()
            v4[S] = v[S] + dt * r3["vdot"]
            x4 = x + dt * r3["xdot"]
            r4 = self._deriv(t + dt, q4, v4, x4, fh)
            vavg = (v + 2 * v2 + 2 * v3 + v4) / 6
            qn = q.copy()
            mujoco.mj_integratePos(m, qn, vavg, dt)
            vn = v.copy()
            vn[S] = v[S] + dt * (r1["vdot"] + 2 * r2["vdot"] + 2 * r3["vdot"] + r4["vdot"]) / 6
            x = x + dt * (r1["xdot"] + 2 * r2["xdot"] + 2 * r3["xdot"] + r4["xdot"]) / 6
            q, v = self.plant.project(qn, vn)
            t = t0 + (k + 1) * dt
            if callback is not None:
                callback(t, q, v, x)
            if verbose and (k % max(1, n // 10) == 0):
                el = time.time() - tic
                dd = diags[-1] if diags else {}
                print(f"  t={t:7.3f}  wall={el:7.1f}s  |s|_M={dd.get('sM', np.nan):.3e}  res_b={dd.get('res_b', np.nan):.2e}"
                      f"  eI={dd.get('eI_norm', np.nan):.3e}", flush=True)
        for kk in logs:
            logs[kk] = np.array(logs[kk])
        self.logs = logs
        self.diags = diags
        return q, v, x
