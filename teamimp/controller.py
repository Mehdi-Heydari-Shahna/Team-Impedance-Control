"""Two-loop team impedance (Sections III-IV of the letter).

Outer loops: external impedance (6) of the payload, required rates (7), compiled lift (8);
internal impedance (9). Inner loop: fitted reference (3), required effort (4), effort (5).
"""
import numpy as np
import scipy.linalg as sla

from .so3 import hat, log, jl_inv, quat2mat, exp


class Params:
    def __init__(self, tm, **kw):
        self.k_rate = kw.get("k_rate", None)          # dict joint -> rate gain (N m s/rad)
        self.gp, self.go = kw.get("gp", 6.0), kw.get("go", 6.0)             # payload request gains
        self.gb_p, self.gb_o = kw.get("gb_p", 6.0), kw.get("gb_o", 6.0)     # pelvis/free end effector
        self.gj = kw.get("gj", 6.0)                                         # joint coordinates
        self.Mc = np.diag(kw.get("Mc", [80.0] * 3 + [12.0] * 3))
        self.Dc = np.diag(kw.get("Dc", [1200.0] * 3 + [240.0] * 3))
        self.Kp = np.diag(kw.get("Kp", [4000.0] * 3))
        self.Ko = np.diag(kw.get("Ko", [600.0] * 3))
        self.mz, self.dz, self.kz = kw.get("mz", 1e-3), kw.get("dz", 0.05), kw.get("kz", 0.0)
        self.foot_lmom = kw.get("foot_lmom", 0.05)


class Refs:
    """Reference provider. Subclass / replace the callables."""

    def payload(self, t):
        raise NotImplementedError       # -> p, R, pd, w, pdd, wd

    def pelvis(self, i, t):
        raise NotImplementedError       # -> p, R, pd, w, pdd, wd

    def joints(self, i, t):
        raise NotImplementedError       # -> q(8), qd(8), qdd(8)  (torso3, neck3, shoulder_ie l/r)

    def ee(self, i, kind, s, t):
        raise NotImplementedError       # -> p, R, pd, w, pdd, wd

    def lam_d(self, t):
        return None                     # internal wrench (grasp frames), in N(G)

    def foot_share(self, i, s, t):
        return 0.5

    def w_extra(self, t, tm):
        return None                     # optional inconsistency w_a (actuator rates)


def pose_request(p, R, v, w, pd, Rd, vd, wd, ad, wdd, gp, go):
    """World-twist request for a pose-type coordinate and its time derivative along the measured motion."""
    phi = log(Rd @ R.T)
    u = np.r_[vd + gp * (pd - p), wd + go * phi]
    Rt = Rd @ R.T
    du = np.r_[ad + gp * (vd - v), wdd + go * jl_inv(phi) @ (wd - Rt @ w)]
    return u, du


class TeamImpedance:
    def __init__(self, tm, params, refs):
        self.tm = tm
        self.P = params
        self.refs = refs
        from . import scene
        kr = params.k_rate
        self.K = np.array([kr[j] for _ in range(tm.N) for j in scene.JOINTS])
        self.ext = None      # (pc, Rc, nuc)
        self.z = None
        self.zd = None
        self.psi_scale = 1.0
        self.internal_mode = "impedance"     # "impedance" | "feedforward" | "off"
        self.share = None                    # optional LoadShare (internal wrench and foot distribution QP)
        self.internal_active = True          # integrate the internal impedance (False: z frozen)
        self.hold_integration = False        # anti-windup: no new error is integrated while an actuator saturates
        self.z_max = None                    # optional clamp of the internal-impedance state (anti-windup)
        self.last = {}

    # ---------------------------------------------------------------- controller state
    def init_states(self, t):
        tm = self.tm
        if tm.mode.payload:
            p, R, v, w = tm.payload_state()
            self.ext = [p.copy(), R.copy(), np.r_[v, w]]
        else:
            self.ext = None
        self.z = np.zeros(tm.nI)
        self.zd = np.zeros(tm.nI)

    def pack(self):
        parts = []
        if self.ext is not None:
            parts += [self.ext[0], self.ext[1].reshape(9), self.ext[2]]
        parts += [self.z, self.zd]
        return np.concatenate(parts) if parts else np.zeros(0)

    def clamp(self):
        if self.z_max is not None and self.z is not None and self.z.size:
            self.z = np.clip(self.z, -self.z_max, self.z_max)

    def unpack(self, x):
        k = 0
        if self.ext is not None:
            pc = x[0:3]
            Rc = x[3:12].reshape(3, 3)
            U, _, Vt = np.linalg.svd(Rc)
            Rc = U @ Vt
            nuc = x[12:18]
            self.ext = [pc.copy(), Rc, nuc.copy()]
            k = 18
        n = self.tm.nI
        self.z = x[k:k + n].copy()
        self.zd = x[k + n:k + 2 * n].copy()

    # ---------------------------------------------------------------- external impedance
    def elastic_wrench(self, pd, Rd, pc, Rc):
        P = self.P
        dp = pd - pc
        Kpd = Rd @ P.Kp @ Rd.T
        Kpc = Rc @ P.Kp @ Rc.T
        f = 0.5 * (Kpd + Kpc) @ dp
        Rcd = Rc.T @ Rd
        q = np.zeros(4)
        import mujoco
        mujoco.mju_mat2Quat(q, Rcd.reshape(9))
        if q[0] < 0:
            q = -q
        eta, eps = q[0], q[1:]
        Eq = eta * np.eye(3) - hat(eps)
        mo = Rc @ (2.0 * Eq.T @ P.Ko @ eps)
        mu = 0.5 * hat(dp) @ Kpc @ dp + mo
        return np.r_[f, mu]

    def ext_derivs(self, t, h):
        """Target impedance (6) for the compliant frame; returns d/dt of (pc, Rc, nuc) and nuc_dot."""
        P = self.P
        pc, Rc, nuc = self.ext
        pd, Rd, vd, wd, ad, wdd = self.refs.payload(t)
        nud = np.r_[vd + hat(wd) @ (pc - pd), wd]
        dnu = nud - nuc
        hD = self.elastic_wrench(pd, Rd, pc, Rc)
        dnu_dot = np.linalg.solve(P.Mc, -h - P.Dc @ dnu - hD)
        nud_dot = np.r_[ad + hat(wdd) @ (pc - pd) + hat(wd) @ (nuc[:3] - vd), wdd]
        nuc_dot = nud_dot - dnu_dot
        return nuc[:3], hat(nuc[3:]) @ Rc, nuc_dot

    # ---------------------------------------------------------------- requests
    def requests(self, t):
        tm, P, refs = self.tm, self.P, self.refs
        d = tm.d
        uq = np.zeros(tm.d_ind)
        duq = np.zeros(tm.d_ind)
        info = {}
        for kind, key, off, size in tm.blocks:
            if kind == "payload":
                p, R, v, w = tm.payload_state()
                pc, Rc, nuc = self.ext
                _, _, nuc_dot = self._ext_cache
                phi = log(Rc @ R.T)
                xi = np.r_[pc - p, phi]
                uq[off:off + 6] = nuc + np.r_[P.gp * xi[:3], P.go * xi[3:]]
                Rt = Rc @ R.T
                xid = np.r_[nuc[:3] - v, jl_inv(phi) @ (nuc[3:] - Rt @ w)]
                duq[off:off + 6] = nuc_dot + np.r_[P.gp * xid[:3], P.go * xid[3:]]
                info["xi_o"] = xi
            elif kind == "pelvis":
                p, R, v, w = tm.pelvis_state(key)
                pd, Rd, vd, wd, ad, wdd = refs.pelvis(key, t)
                u, du = pose_request(p, R, v, w, pd, Rd, vd, wd, ad, wdd, P.gb_p, P.gb_o)
                uq[off:off + 6], duq[off:off + 6] = u, du
                info[("xi_p", key)] = np.r_[pd - p, log(Rd @ R.T)]
            elif kind == "joint":
                i, dofs = key
                qa = [tm.m.jnt_qposadr[tm.m.dof_jntid[k]] for k in dofs]
                qq = d.qpos[qa]
                qv = d.qvel[list(dofs)]
                qd, qdd, qddd = refs.joints(i, t)
                uq[off:off + size] = qdd + P.gj * (qd - qq)
                duq[off:off + size] = qddd + P.gj * (qdd - qv)
                info[("xi_j", i)] = qd - qq
            elif kind == "ee":
                i, k2, s = key
                p, R, v, w = tm.site_state(i, k2, s)
                pd, Rd, vd, wd, ad, wdd = refs.ee(i, k2, s, t)
                u, du = pose_request(p, R, v, w, pd, Rd, vd, wd, ad, wdd, P.gb_p, P.gb_o)
                uq[off:off + 6], duq[off:off + 6] = u, du
                info[("xi_e", key)] = np.r_[pd - p, log(Rd @ R.T)]
        return uq, duq, info

    # ---------------------------------------------------------------- inner loop
    def effort(self, t, h=None, lam_meas=None, ext_dot_cache=None):
        """Compute f at the current model state (tm.update must have been called).
        h: environment wrench on the payload (world force, moment about the payload origin)."""
        tm, P = self.tm, self.P
        K = self.K
        if h is None:
            h = np.zeros(6)
        if tm.mode.payload:
            self._ext_cache = self.ext_derivs(t, h)
        uq, duq, rinfo = self.requests(t)
        L, E, Ed, B = tm.L, tm.E, tm.Ed, tm.B
        w = L @ uq
        wa = self.refs.w_extra(t, tm)
        if wa is not None:
            w = w + wa
        LtK = L.T * K
        LtKL = LtK @ L
        lu = sla.cho_factor(LtKL)
        ur = sla.cho_solve(lu, LtK @ w)
        wperp = w - L @ ur
        dur = duq                      # consistent part; d/dt of the inconsistent part is supplied by refs
        if wa is not None and hasattr(self.refs, "dur_extra"):
            dur = dur + self.refs.dur_extra(t, tm, lu, LtK)
        vr = E @ ur
        vr_dot = Ed @ ur + E @ dur
        Cvr = tm.coriolis(vr)
        tau = tm.M @ vr_dot + Cvr + tm.g + tm.Dv @ vr
        if tm.mode.payload:
            c = tm.col[tm.beam_dof]
            Ro = tm.Ro
            tau[c:c + 3] -= h[:3]
            tau[c + 3:c + 6] -= Ro.T @ h[3:]
        tm.effort_split(K)
        fbar = K * (L @ sla.cho_solve(lu, E.T @ tau))
        lbar = sla.cho_solve(tm.AAt_lu, tm.A @ (tau - B @ fbar))
        if tm.ng:
            PIlg = tm.PI @ lbar[tm.gidx]
            fr = fbar - tm.Psip @ PIlg
        else:
            fr = fbar
        lam_r = sla.cho_solve(tm.AAt_lu, tm.A @ (tau - B @ fr))
        # internal command
        if tm.ng:
            lam_d = self.refs.lam_d(t)
            if lam_d is None:
                lam_d = np.zeros(6 * tm.ng)
            if self.share is not None and self.share.lam_d is not None and self.share.lam_d.shape == lam_d.shape:
                lam_d = lam_d + self.share.lam_d
            if self.internal_mode == "off":
                psi = np.zeros(6 * tm.ng)
            elif self.internal_mode == "feedforward":
                psi = lam_d.copy()
            else:
                psi = lam_d + tm.VI @ self.z
            f_int = tm.Psip @ psi
        else:
            lam_d = np.zeros(0)
            psi = np.zeros(0)
            f_int = np.zeros(tm.mact)
        # distribution of each robot's support between its planted feet (effort in N(L^T) cap N(Psi))
        if self.share is not None and self.share.fF is not None and self.share.fF.shape[0] == tm.mact \
                and tm.NF.shape[1] > 0:
            fF = tm.NF @ (tm.NF.T @ self.share.fF)
        else:
            fF = self.foot_effort(t, lam_r - tm.mu(f_int))
        f = fr - K * (B.T @ tm.v - w) + f_int + fF
        self.last = dict(uq=uq, duq=duq, w=w, wperp=wperp, ur=ur, vr=vr, vr_dot=vr_dot, tau=tau, fr=fr,
                         lam_r=lam_r, psi=psi, lam_d=lam_d, f_int=f_int, fF=fF, f=f, h=h, rinfo=rinfo)
        return f

    def foot_effort(self, t, lam_pre):
        tm = self.tm
        if not tm.foot_rows or tm.NF.shape[1] == 0:
            return np.zeros(tm.mact)
        lm = self.P.foot_lmom
        target = lam_pre.copy()
        by_robot = {}
        for k, (kind, i, s) in enumerate(tm.rows):
            if kind == "foot":
                by_robot.setdefault(i, []).append((k, s))
        for i, lst in by_robot.items():
            if len(lst) < 2:
                continue
            o = np.mean([tm.site_p[(i, "foot", s)] for _, s in lst], axis=0)
            Xs, Ws = [], []
            Wtot = np.zeros(6)
            for k, s in lst:
                R = tm.site_R[(i, "foot", s)]
                p = tm.site_p[(i, "foot", s)]
                X = np.zeros((6, 6))
                X[:3, :3] = R
                X[3:, :3] = hat(p - o) @ R
                X[3:, 3:] = R
                lam = lam_pre[6 * k:6 * k + 6]
                Wtot += X @ lam
                a = max(self.refs.foot_share(i, s, t), 1e-6)
                Qi = np.diag(np.r_[np.ones(3), np.ones(3) / lm ** 2]) / a
                Xs.append(X)
                Ws.append(np.linalg.inv(Qi))
            S = sum(X @ Wi @ X.T for X, Wi in zip(Xs, Ws))
            mult = np.linalg.solve(S, Wtot)
            for (k, s), X, Wi in zip(lst, Xs, Ws):
                target[6 * k:6 * k + 6] = Wi @ X.T @ mult
        rows = tm.fidx
        Mf = tm.Mmu[rows] @ tm.NF
        rhs = (lam_pre - target)[rows]
        c, *_ = np.linalg.lstsq(Mf, rhs, rcond=None)
        return tm.NF @ c

    # ---------------------------------------------------------------- internal impedance dynamics
    def internal_derivs(self, lam_g_meas):
        tm, P = self.tm, self.P
        if not tm.ng:
            return np.zeros(0), np.zeros(0)
        if not self.internal_active:
            return np.zeros(tm.nI), -50.0 * self.zd
        lamI = tm.PI @ lam_g_meas
        lam_d = self.last["lam_d"]
        eI = tm.VIp @ (lamI - lam_d)
        if self.hold_integration:            # conditional integration (outside the theory: actuator at its limit)
            eI = np.zeros_like(eI)
        zdd = (-P.dz * self.zd - P.kz * self.z - eI) / P.mz
        return self.zd.copy(), zdd

    def state_derivs(self, lam_g_meas):
        parts = []
        if self.ext is not None:
            pdot, Rdot, nudot = self._ext_cache
            parts += [pdot, Rdot.reshape(9), nudot]
        zd, zdd = self.internal_derivs(lam_g_meas)
        parts += [zd, zdd]
        return np.concatenate(parts)
