"""Compiled team interfaces: connection Jacobian A, tangent lift E, actuator map B, L = B^T E,
dynamics terms M, C(q,v)w (factorization with Mdot - 2C skew), g, D, and the effort split of Lemma 1.

Conventions
-----------
* v is MuJoCo's qvel restricted to the "system" dofs (all robots, plus the payload while it is grasped).
* Free joints (pelvis, payload): qvel = (linear velocity of the body origin in world, angular velocity in body
  axes). Independent pose speeds are world-frame twists (pdot, omega_world); T = diag(I, R^T) maps them to qvel.
* Connection rows: a planted sole (6 rows, expressed in the sole frame; multiplier = ground wrench on the foot)
  and a closed grasp (6 rows, expressed in the grasp frame attached to the payload; multiplier = wrench applied
  by the hand to the payload, moment about the grasp point).
* Independent coordinates u: payload twist nu (if grasped), then per robot: pelvis twist (6), torso (3),
  neck (3), the two arm redundancy joints (shoulder_ie), and a world twist for each free end effector.
  Dependent: the six leg joints of each leg and six arm joints of each arm (all but shoulder_ie).
"""
import numpy as np
import mujoco
import scipy.linalg as sla

from . import scene
from .so3 import hat


class Robot:
    def __init__(self, m, i):
        self.i = i
        p = f"r{i}_"
        jid = m.joint(p + "floating_base").id
        self.base = int(m.jnt_dofadr[jid])
        self.qadr = int(m.jnt_qposadr[jid])
        self.dof = {jn: int(m.jnt_dofadr[m.joint(p + jn).id]) for jn in scene.JOINTS}
        self.qpos_adr = {jn: int(m.jnt_qposadr[m.joint(p + jn).id]) for jn in scene.JOINTS}
        self.dofs = list(range(self.base, self.base + 6)) + [self.dof[j] for j in scene.JOINTS]
        self.act = [self.dof[j] for j in scene.JOINTS]                      # actuated dofs (qvel index)
        self.pelvis = m.body(p + "base_link").id
        self.site = {("foot", s): m.site(p + f"{s}_sole").id for s in "lr"}
        self.site.update({("hand", s): m.site(p + f"{s}_palm").id for s in "lr"})
        self.chain = {("foot", s): [self.dof[j] for j in scene.LEG[s]] for s in "lr"}
        self.chain.update({("hand", s): [self.dof[j] for j in scene.ARM[s] if j != scene.ARM_RED[s]] for s in "lr"})
        self.torso = [self.dof[j] for j in scene.TORSO]
        self.neck = [self.dof[j] for j in scene.NECK]
        self.red = [self.dof[scene.ARM_RED[s]] for s in "lr"]


class Mode:
    """Support/grasp mode. feet[(i, s)] = (p*, R*) if planted; grasps[(i, s)] = (p_o, R_o) in payload frame."""

    def __init__(self, feet=None, grasps=None):
        self.feet = dict(feet or {})
        self.grasps = dict(grasps or {})

    @property
    def payload(self):
        return len(self.grasps) > 0

    def key(self):
        return (tuple(sorted(self.feet)), tuple(sorted(self.grasps)))

    def copy(self):
        return Mode(self.feet, self.grasps)


class TeamModel:
    def __init__(self, m, n):
        self.m = m
        self.d = mujoco.MjData(m)
        self.N = n
        self.robots = [Robot(m, i) for i in range(n)]
        self.beam = m.body("beam").id
        jb = m.joint("beam_free").id
        self.beam_dof = int(m.jnt_dofadr[jb])
        self.beam_qadr = int(m.jnt_qposadr[jb])
        self.nv = m.nv
        # ancestry for the Coriolis factorization
        self.anc = np.zeros((m.nbody, m.nv))
        for b in range(1, m.nbody):
            c = b
            while c > 0:
                for k in range(m.body_dofadr[c], m.body_dofadr[c] + m.body_dofnum[c]):
                    self.anc[b, k] = 1.0
                c = m.body_parentid[c]
        self.damping = m.dof_damping.copy()
        self.dof_body = m.dof_bodyid.copy()
        self.free_trans = np.array([m.jnt_dofadr[j] + k for j in range(m.njnt)
                                    if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for k in range(3)], int)
        self.act_all = [k for r in self.robots for k in r.act]
        self._jp = np.zeros((3, m.nv))
        self._jr = np.zeros((3, m.nv))
        self.mode = None
        self.l_char = 0.3          # characteristic length for moments (weights, norms)

    # ------------------------------------------------------------------ layout
    def set_mode(self, mode):
        self.mode = mode
        sysd = [k for r in self.robots for k in r.dofs]
        if mode.payload:
            sysd += list(range(self.beam_dof, self.beam_dof + 6))
        self.sys = np.array(sysd)
        self.col = -np.ones(self.nv, int)
        self.col[self.sys] = np.arange(len(self.sys))
        self.n = len(self.sys)
        # independent blocks
        blocks = []
        if mode.payload:
            blocks.append(("payload", None, 6))
        for r in self.robots:
            blocks.append(("pelvis", r.i, 6))
            blocks.append(("joint", (r.i, tuple(r.torso + r.neck + r.red)), len(r.torso + r.neck + r.red)))
            for kind in ("foot", "hand"):
                for s in "lr":
                    connected = ((r.i, s) in mode.feet) if kind == "foot" else ((r.i, s) in mode.grasps)
                    if not connected:
                        blocks.append(("ee", (r.i, kind, s), 6))
        self.blocks = []
        off = 0
        for b in blocks:
            self.blocks.append((b[0], b[1], off, b[2]))
            off += b[2]
        self.d_ind = off
        self.blk = {(b[0], b[1]): (b[2], b[3]) for b in self.blocks}
        # connection rows
        rows = []
        for (i, s) in sorted(mode.feet):
            rows.append(("foot", i, s))
        for (i, s) in sorted(mode.grasps):
            rows.append(("grasp", i, s))
        self.rows = rows
        self.r = 6 * len(rows)
        self.grasp_rows = [k for k, rw in enumerate(rows) if rw[0] == "grasp"]
        self.foot_rows = [k for k, rw in enumerate(rows) if rw[0] == "foot"]
        self.gidx = np.concatenate([np.arange(6 * k, 6 * k + 6) for k in self.grasp_rows]).astype(int) \
            if self.grasp_rows else np.zeros(0, int)
        self.fidx = np.concatenate([np.arange(6 * k, 6 * k + 6) for k in self.foot_rows]).astype(int) \
            if self.foot_rows else np.zeros(0, int)
        self.ng = len(self.grasp_rows)
        # actuator map
        self.act = np.array([self.col[k] for k in self.act_all])
        self.mact = len(self.act)
        self.B = np.zeros((self.n, self.mact))
        self.B[self.act, np.arange(self.mact)] = 1.0
        # constant grasp quantities (payload-frame grasp matrix, internal projector, basis of N(G))
        if self.ng:
            Go = np.zeros((6, 6 * self.ng))
            for j, k in enumerate(self.grasp_rows):
                _, i, s = rows[k]
                po, Ro = mode.grasps[(i, s)]
                Go[:3, 6 * j:6 * j + 3] = Ro
                Go[3:, 6 * j:6 * j + 3] = hat(po) @ Ro
                Go[3:, 6 * j + 3:6 * j + 6] = Ro
            self.Go = Go
            w = np.tile(np.r_[np.ones(3), np.ones(3) / self.l_char ** 2], self.ng)
            Wi = np.diag(1.0 / w)
            self.W = np.diag(w)
            self.DG_o = Wi @ Go.T @ np.linalg.inv(Go @ Wi @ Go.T)        # D_G in payload frame
            self.PI = np.eye(6 * self.ng) - self.DG_o @ Go
            U, S, Vt = np.linalg.svd(Go)
            self.VI = Vt[6:].T                                            # orthonormal basis of N(G)
            self.VIp = self.VI.T
        else:
            self.Go = np.zeros((6, 0))
            self.PI = np.zeros((0, 0))
            self.VI = np.zeros((0, 0))
            self.VIp = np.zeros((0, 0))
        self.nI = self.VI.shape[1]

    # ------------------------------------------------------------------ kinematics/dynamics evaluation
    def _site_jac(self, sid):
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, sid)
        return np.vstack([self._jp, self._jr])

    def _point_jac(self, point, body):
        mujoco.mj_jac(self.m, self.d, self._jp, self._jr, point, body)
        return np.vstack([self._jp, self._jr])

    def _point_jacdot(self, point, body):
        mujoco.mj_jacDot(self.m, self.d, self._jp, self._jr, point, body)
        return np.vstack([self._jp, self._jr])

    def update(self, q, v, dyn=True, need_dot=True):
        """Evaluate kinematics, A, E (and their time derivatives along v), and dynamics at (q, v)."""
        m, d = self.m, self.d
        d.qpos[:] = q
        d.qvel[:] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        if dyn:
            mujoco.mj_crb(m, d)
        mujoco.mj_comVel(m, d)
        if dyn:
            mujoco.mj_rne(m, d, 0, d.qfrc_bias)
        S = self.sys
        mode = self.mode
        n, dI = self.n, self.d_ind
        self.v = v[S].copy()
        # rotations
        Rp = {r.i: d.xmat[r.pelvis].reshape(3, 3).copy() for r in self.robots}
        self.Rp = Rp
        if mode.payload:
            self.Ro = d.xmat[self.beam].reshape(3, 3).copy()
            self.po = d.xpos[self.beam].copy()
            self.wo = self.Ro @ d.qvel[self.beam_dof + 3:self.beam_dof + 6]
        # --- E: independent rows
        E = np.zeros((n, dI))
        Ed = np.zeros((n, dI))
        for kind, key, off, size in self.blocks:
            if kind == "payload":
                c = self.col[self.beam_dof]
                E[c:c + 3, off:off + 3] = np.eye(3)
                E[c + 3:c + 6, off + 3:off + 6] = self.Ro.T
                Ed[c + 3:c + 6, off + 3:off + 6] = -self.Ro.T @ hat(self.wo)
            elif kind == "pelvis":
                r = self.robots[key]
                c = self.col[r.base]
                R = Rp[key]
                w = R @ d.qvel[r.base + 3:r.base + 6]
                E[c:c + 3, off:off + 3] = np.eye(3)
                E[c + 3:c + 6, off + 3:off + 6] = R.T
                Ed[c + 3:c + 6, off + 3:off + 6] = -R.T @ hat(w)
            elif kind == "joint":
                i, dofs = key
                cc = self.col[list(dofs)]
                E[cc, off + np.arange(size)] = 1.0
        # payload twist selector and payload-point Jacobian helpers
        if mode.payload:
            o, _ = self.blk[("payload", None)]
            cb = self.col[self.beam_dof]
        self.J = {}
        self.Jd = {}
        A = np.zeros((self.r, n))
        Ad = np.zeros((self.r, n))
        self.site_R = {}
        self.site_p = {}
        for r in self.robots:
            for (kind, s), sid in r.site.items():
                Jf = self._site_jac(sid)
                J = Jf[:, S]
                Rk = d.site_xmat[sid].reshape(3, 3).copy()
                pk = d.site_xpos[sid].copy()
                self.site_R[(r.i, kind, s)] = Rk
                self.site_p[(r.i, kind, s)] = pk
                if need_dot:
                    Jdf = self._point_jacdot(pk, int(m.site_bodyid[sid]))
                    Jd = Jdf[:, S]
                else:
                    Jd = np.zeros_like(J)
                self.J[(r.i, kind, s)] = J
                self.Jd[(r.i, kind, s)] = Jd
                D = self.col[r.chain[(kind, s)]]
                JD = J[:, D]
                # target twist map Xi (6 x dI) and its time derivative
                connected = ((r.i, s) in mode.feet) if kind == "foot" else ((r.i, s) in mode.grasps)
                Xi, Xid = None, None
                if not connected:
                    eo, _ = self.blk[("ee", (r.i, kind, s))]
                    Xi = np.zeros((6, dI))
                    Xi[:, eo:eo + 6] = np.eye(6)
                elif kind == "hand":
                    Jpo = self._point_jac(pk, self.beam)[:, S]
                    Xi = Jpo @ E
                    if need_dot:
                        Jpod = self._point_jacdot(pk, self.beam)[:, S]
                        Xid = Jpod @ E + Jpo @ Ed
                    self.J[(r.i, "po", s)] = Jpo
                    self.Jd[(r.i, "po", s)] = Jpod if need_dot else np.zeros_like(Jpo)
                # rows D of E are still zero here, so J @ E = J_R E_R (other chains have zero columns)
                JDi = np.linalg.inv(JD)
                rhs = -(J @ E) if Xi is None else Xi - J @ E
                E[D, :] = JDi @ rhs
                if need_dot:
                    rhs = -(Jd @ E) - J @ Ed
                    if Xid is not None:
                        rhs = rhs + Xid
                    Ed[D, :] = JDi @ rhs
        # connection rows
        for k, (kind, i, s) in enumerate(self.rows):
            sl = slice(6 * k, 6 * k + 6)
            if kind == "foot":
                J = self.J[(i, "foot", s)]
                Jd = self.Jd[(i, "foot", s)]
                R = self.site_R[(i, "foot", s)]
            else:
                J = self.J[(i, "po", s)] - self.J[(i, "hand", s)]
                Jd = self.Jd[(i, "po", s)] - self.Jd[(i, "hand", s)]
                R = self.Ro @ self.mode.grasps[(i, s)][1]
            RR = np.zeros((6, 6))
            RR[:3, :3] = R.T
            RR[3:, 3:] = R.T
            A[sl] = RR @ J
            if need_dot:
                if kind == "foot":
                    w = self.J[(i, "foot", s)][3:] @ self.v
                else:
                    w = self.wo
                Wx = np.zeros((6, 6))
                Wx[:3, :3] = hat(w)
                Wx[3:, 3:] = hat(w)
                Ad[sl] = RR @ (Jd - Wx @ J)
        self.A, self.Ad, self.E, self.Ed = A, Ad, E, Ed
        self.L = E[self.act, :]
        if dyn:
            M = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, d, M)
            self.M = M[np.ix_(S, S)]
            self.bias = d.qfrc_bias[S].copy()
            self.Dv = np.diag(self.damping[S])
            self.Cv = self.coriolis(self.v)
            self.g = self.bias - self.Cv
        return self

    # ------------------------------------------------------------------ Coriolis factorization
    def coriolis(self, w_sys):
        """C(q, v) w for the factorization C_i(V)W = 1/2[V x* I W - I (V x W) + W x* I V] (fixed frame),
        which satisfies Mdot = C + C^T. w_sys may be (n,) or (n, k)."""
        m, d = self.m, self.d
        w = np.zeros((m.nv,) + np.shape(w_sys)[1:])
        w[self.sys] = w_sys
        single = w.ndim == 1
        if single:
            w = w[:, None]
        cdof = d.cdof                      # (nv, 6) [ang; lin]
        V = d.cvel                         # (nbody, 6)
        # true time derivative of the motion subspaces (MuJoCo's cdof_dot uses the parent velocity for
        # ball/free rotations, which is enough for the bias but not for C(q,v)w with w != v)
        Vd = V[self.dof_body]
        wa, va = Vd[:, :3], Vd[:, 3:]
        cdd = np.concatenate([np.cross(wa, cdof[:, :3]), np.cross(wa, cdof[:, 3:]) + np.cross(va, cdof[:, :3])],
                             axis=1)
        cdd[self.free_trans] = 0.0
        k = w.shape[1]
        Wb = (self.anc @ (cdof[:, :, None] * w[:, None, :]).reshape(m.nv, 6 * k)).reshape(m.nbody, 6, k)
        Ab = (self.anc @ (cdd[:, :, None] * w[:, None, :]).reshape(m.nv, 6 * k)).reshape(m.nbody, 6, k)
        ci = d.cinert                      # (nbody, 10)
        I = np.zeros((m.nbody, 6, 6))
        I[:, 0, 0], I[:, 1, 1], I[:, 2, 2] = ci[:, 0], ci[:, 1], ci[:, 2]
        I[:, 0, 1] = I[:, 1, 0] = ci[:, 3]
        I[:, 0, 2] = I[:, 2, 0] = ci[:, 4]
        I[:, 1, 2] = I[:, 2, 1] = ci[:, 5]
        h = ci[:, 6:9]
        mass = ci[:, 9]
        Sh = np.zeros((m.nbody, 3, 3))
        Sh[:, 0, 1], Sh[:, 0, 2], Sh[:, 1, 0] = -h[:, 2], h[:, 1], h[:, 2]
        Sh[:, 1, 2], Sh[:, 2, 0], Sh[:, 2, 1] = -h[:, 0], -h[:, 1], h[:, 0]
        I[:, 0:3, 3:6] = Sh
        I[:, 3:6, 0:3] = np.transpose(Sh, (0, 2, 1))
        I[:, 3, 3] = I[:, 4, 4] = I[:, 5, 5] = mass

        def cross_m(a, b):        # motion x motion; a (nb,6), b (nb,6,k)
            wa, va = a[:, :3, None], a[:, 3:, None]
            wb, vb = b[:, :3], b[:, 3:]
            return np.concatenate([np.cross(wa, wb, axis=1), np.cross(wa, vb, axis=1) + np.cross(va, wb, axis=1)],
                                  axis=1)

        def cross_f(a, f):        # motion x* force; a (nb,6,k) or (nb,6), f (nb,6,k)
            if a.ndim == 2:
                a = a[:, :, None]
            wa, va = a[:, :3], a[:, 3:]
            n, fl = f[:, :3], f[:, 3:]
            return np.concatenate([np.cross(wa, n, axis=1) + np.cross(va, fl, axis=1), np.cross(wa, fl, axis=1)],
                                  axis=1)

        IW = np.einsum("bij,bjk->bik", I, Wb)
        IV = np.einsum("bij,bj->bi", I, V)[:, :, None]
        F = np.einsum("bij,bjk->bik", I, Ab) + 0.5 * (
            cross_f(V, IW) - np.einsum("bij,bjk->bik", I, cross_m(V, Wb)) + cross_f(Wb, np.broadcast_to(IV, Wb.shape)))
        Fsub = (self.anc.T @ F.reshape(m.nbody, 6 * k)).reshape(m.nv, 6, k)   # bodies supported by dof j
        tau = np.einsum("jc,jck->jk", cdof, Fsub)
        out = tau[self.sys]
        return out[:, 0] if single else out

    # ------------------------------------------------------------------ grasp matrix (current) and effort split
    def grasp_matrix(self):
        Rb = np.zeros((6, 6))
        Rb[:3, :3] = self.Ro
        Rb[3:, 3:] = self.Ro
        return Rb @ self.Go

    def effort_split(self, K):
        """Matrices of Lemma 1: Mmu = (AA^T)^{-1} A B (mu(f) = Mmu f), Psi~, Psi^+ = (Psi~ Pi_L)^+ (range in
        N(L^T)), an orthonormal basis Z (= NL) of N(L^T), and a basis NF of N(L^T) cap N(Psi~)."""
        A, L = self.A, self.L
        AAt = A @ A.T
        self.AAt_lu = sla.cho_factor(AAt)
        Mmu = sla.cho_solve(self.AAt_lu, A[:, self.act])          # A B (B selects the actuated columns)
        self.Mmu = Mmu
        Q, _ = np.linalg.qr(L, mode="complete")
        Z = Q[:, L.shape[1]:]
        self.NL = Z
        if self.ng:
            Psit = -Mmu[self.gidx]
            self.Psit = Psit
            U, Sv, Vt = np.linalg.svd(Psit @ Z, full_matrices=True)
            k = self.nI
            # (Psi~ Pi_L)^+ = Z (Psi~ Z)^+ since Pi_L = Z Z^T
            self.Psip = Z @ ((Vt[:k].T / Sv[:k]) @ U[:, :k].T)
            self.psi_sv = Sv
            self.NF = Z @ Vt[k:].T
        else:
            self.Psit = np.zeros((0, self.mact))
            self.Psip = np.zeros((self.mact, 0))
            self.NF = Z
        return self

    def mu(self, f):
        return self.Mmu @ f

    # ------------------------------------------------------------------ measured independent coordinates
    def payload_state(self):
        d = self.d
        p = d.xpos[self.beam].copy()
        R = d.xmat[self.beam].reshape(3, 3).copy()
        v = d.qvel[self.beam_dof:self.beam_dof + 3].copy()
        w = R @ d.qvel[self.beam_dof + 3:self.beam_dof + 6]
        return p, R, v, w

    def pelvis_state(self, i):
        d = self.d
        r = self.robots[i]
        p = d.xpos[r.pelvis].copy()
        R = d.xmat[r.pelvis].reshape(3, 3).copy()
        v = d.qvel[r.base:r.base + 3].copy()
        w = R @ d.qvel[r.base + 3:r.base + 6]
        return p, R, v, w

    def site_state(self, i, kind, s):
        J = self.J[(i, kind, s)]
        tw = J @ self.v
        return self.site_p[(i, kind, s)], self.site_R[(i, kind, s)], tw[:3], tw[3:]
