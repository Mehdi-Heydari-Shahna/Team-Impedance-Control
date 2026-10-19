"""Numerical validation of Theorems 1-2 and Proposition 1 on the exact compiled model of the humanoid team
(N = 4 Apollo humanoids, 8 feet and 8 palm grasps, n = 158, d = 62, r = 96, m = 128; ideal connections).

The plant integrates (1) with RK4 and constraint projection; the controller is evaluated at every RK4 stage
(continuous time) unless a zero-order hold period is given.

Usage: python experiments/exact_validation.py <experiment> [args]
  identities | decoupling | zoh <Ts_ms> | rates | refdisp | antagonism | corollary
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import scipy.linalg as sla
import mujoco

from teamimp import mission as MS, team, setup, controller, refs as RF, sim_exact, scene
from teamimp.so3 import rot_z, rot_x, exp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exact")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(7)


PROFILE = os.environ.get("GIRDER_PROFILE", "AL 280")     # "AL 280": 63 kg aluminium girder; "HEA 280": 183 kg steel


def base_team():
    """Compiled team at the carrying configuration of the mission (girder PROFILE, 2.4 m)."""
    cfg = MS.Cfg(n=4, background=False, verbose=False, D_app=0.0, profile=PROFILE)
    M = MS.Mission(cfg)
    mc = M.R.mc
    q0 = M.q_pick.copy()
    return M, mc, q0


def perturbed_model(M, rel=0.3, payload=1.2, seed=3):
    """Controller model with random link mass/inertia errors (uniform in +-rel) and a payload mass error."""
    spec = scene.build_spec(4, "ctrl", profile=PROFILE, beam_pos=tuple(M.beamA))
    mp = spec.compile()
    mp.dof_frictionloss[:] = 0.0
    rng = np.random.default_rng(seed)
    beam = mp.body("beam").id
    for b in range(1, mp.nbody):
        f = payload if b == beam else 1.0 + rel * rng.uniform(-1, 1)
        mp.body_mass[b] *= f
        mp.body_inertia[b] *= f
    return mp


class Refs(RF.HoldRefs):
    """Hold references + optional payload/pelvis motion, internal command and request inconsistency."""

    def __init__(self, tm, q0, motion=False, lam_d=None, wa=None, K=None):
        d = tm.d
        d.qpos[:] = q0
        mujoco.mj_kinematics(tm.m, d)
        p0, R0 = d.xpos[tm.beam].copy(), d.xmat[tm.beam].reshape(3, 3).copy()
        traj = None
        if motion:
            # the team moves the girder 6 cm sideways, 5 cm up and yaws it by 5 deg, then returns
            p1 = p0 + np.array([0.0, 0.06, 0.05])
            R1 = rot_z(np.radians(5.0)) @ R0
            traj = RF.PoseTraj([(0.0, p0, R0), (1.2, p1, R1), (2.6, p0, R0)])
        super().__init__(tm, q0, payload_traj=traj, lam_d=lam_d)
        if motion:
            for r in tm.robots:
                self.pelvis_offset[r.i] = RF.VecTraj([(0.0, np.zeros(3)), (1.2, np.array([0.0, 0.06, 0.03])),
                                                      (2.6, np.zeros(3))])
        self._wa = wa
        self.K = K

    def w_extra(self, t, tm):
        if self._wa is None:
            return None
        return self._wa(t, tm)

    def dur_extra(self, t, tm, lu, LtK):
        """Time derivative of du_a = (L^T K L)^{-1} L^T K w_a along the motion (w_a(t, q) given)."""
        wa = self._wa(t, tm)
        if getattr(self._wa, "zero_du", False):
            return np.zeros(tm.d_ind)
        wad = self._wa.dot(t, tm) if hasattr(self._wa, "dot") else np.zeros_like(wa)
        L = tm.L
        Ld = tm.Ed[tm.act, :]
        K = self.K
        du = sla.cho_solve(lu, LtK @ wa)
        LdtK = Ld.T * K
        rhs = LdtK @ wa - (LdtK @ L + LtK @ Ld) @ du + LtK @ wad
        return sla.cho_solve(lu, rhs)


def make(M, mc, q0, model=None, motion=False, lam_d=None, wa=None, env=None, Ts=None, internal="impedance",
         kz=0.0, diag_every=1, log_every=1):
    tm = team.TeamModel(mc, 4)
    mode = setup.grasp_mode(tm, q0)
    tm.set_mode(mode)
    sim = sim_exact.ExactSim(tm, None, env=env, Ts=Ts, diag_every=diag_every, log_every=log_every)
    q, v = sim.plant.project(q0.copy(), np.zeros(mc.nv))
    if model is not None:
        tc = team.TeamModel(model, 4)
        tc.set_mode(mode)
    else:
        tc = tm
    K = setup.rate_gains()
    P = setup.default_params(tc, kz=kz)
    Kvec = np.array([K[j] for _ in range(4) for j in scene.JOINTS])
    R = Refs(tc, q, motion=motion, lam_d=lam_d, wa=wa, K=Kvec)
    C = controller.TeamImpedance(tc, P, R)
    C.internal_mode = internal
    tc.update(q, v)
    if tc is not tm:
        tm.update(q, v)
    C.init_states(0.0)
    sim.ctrl = C
    sim.tc = tc
    return sim, q, v, tm, tc, C


def kappa_s(tm, K, qs):
    """min over the logged configurations of lambda_min(E^T (B K B^T + D) E, E^T M E)."""
    out = []
    for q in qs:
        tm.update(q, np.zeros(tm.m.nv))
        E, M_, B, D = tm.E, tm.M, tm.B, tm.Dv
        Ku = E.T @ ((B * K) @ B.T + D) @ E
        Mu = E.T @ M_ @ E
        out.append(sla.eigh(Ku, Mu, eigvals_only=True, subset_by_index=[0, 0])[0])
    return float(np.min(out)), np.array(out)


def squeeze_pattern(tm, F=80.0, M_=0.0):
    """Internal wrench pattern: palms push the girder horizontally towards its axis (y), plus a moment
    redistribution; projected on N(G)."""
    lam = np.zeros(6 * tm.ng)
    for j, k in enumerate(tm.grasp_rows):
        _, i, s = tm.rows[k]
        lam[6 * j + 1] = -F          # grasp-frame y points away from the girder axis
        lam[6 * j + 3] = M_ * (1 if s == "l" else -1)
    return tm.PI @ lam


def run(sim, q, v, T, dt, verbose=False):
    t0 = time.time()
    sim.run(q, v, 0.0, T, dt, verbose=verbose)
    return time.time() - t0


def initial_velocity(tm, q, scale, seed=1):
    rng = np.random.default_rng(seed)
    tm.update(q, np.zeros(tm.m.nv))
    u = rng.standard_normal(tm.d_ind) * scale
    v = np.zeros(tm.m.nv)
    v[tm.sys] = tm.E @ u
    return v


# ---------------------------------------------------------------------------------------------- experiments
def exp_identities(T=2.6, dt=1e-3):
    M, mc, q0 = base_team()
    env = setup.SpringSupport(Ke=[0.0, 2.0e4, 0.0], De=[0.0, 400.0, 0.0], t_on=0.0)
    tmp = team.TeamModel(mc, 4)
    tmp.set_mode(setup.grasp_mode(tmp, q0))
    lamA = squeeze_pattern(tmp, 120.0, 4.0)

    def lam_d(t):
        return np.sin(2 * np.pi * 0.7 * t) * lamA
    sim, q, v, tm, tc, C = make(M, mc, q0, motion=True, lam_d=lam_d, env=_env_wrap(env))
    v0 = initial_velocity(tm, q, 0.05)
    wall = run(sim, q, v0, T, dt)
    D = sim.diags
    t = np.array([d["t"] for d in D])
    sM = np.array([d["sM"] for d in D])
    res_a = np.array([abs(d["res_a"]) for d in D])
    diss = np.array([abs(d["diss"]) for d in D])
    res_b = np.array([d["res_b"] for d in D])
    lam_scale = np.array([d["lam_scale"] for d in D])
    res_I = np.array([d["res_I"] for d in D])
    res_c = np.array([d["res_c"] for d in D])
    K = C.K
    ks, ks_t = kappa_s(team.TeamModel(mc, 4) if False else tm, K, sim.logs["q"][::50])
    tm.update(sim.logs["q"][-1], sim.logs["v"][-1])
    out = dict(t=t, sM=sM, res_a=res_a, diss=diss, res_b=res_b, lam_scale=lam_scale, res_I=res_I, res_c=res_c,
               kappa_s=ks, wall=wall, dims=dict(n=tm.n, d=tm.d_ind, r=tm.r, m=tm.mact, nI=tm.nI,
                                                 nNLt=int(tm.NL.shape[1]), nNF=int(tm.NF.shape[1])))
    np.savez(os.path.join(OUT, "identities.npz"), **{k: v for k, v in out.items() if k != "dims"},
             dims=json.dumps(out["dims"]))
    print("identities: max rel res_a", (res_a / np.maximum(diss, 1e-9)).max(), "max res_a", res_a.max(),
          "max res_b", res_b.max(), "lam", lam_scale.max(), "res_I", res_I.max(), "res_c", res_c.max(),
          "kappa_s", ks, "dims", out["dims"], "wall", wall, flush=True)


class _env_wrap:
    """Passes time to the spring environment (engagement) and keeps it callable as env(tm)."""

    def __init__(self, env):
        self.env = env

    def __call__(self, tm):
        return self.env(tm)


def exp_decoupling(T=2.6, dt=1e-3):
    """Theorem 1(d): identical motion with psi = 0 and with a large varying internal command, for the exact
    model and for a controller model with +-30% link inertia errors and +20% payload mass."""
    M, mc, q0 = base_team()
    tmp = team.TeamModel(mc, 4)
    tmp.set_mode(setup.grasp_mode(tmp, q0))
    lamA = squeeze_pattern(tmp, 200.0, 6.0)

    def lam_d(t):
        return (0.6 + 0.4 * np.sin(2 * np.pi * 1.1 * t)) * lamA
    results = {}
    for model_tag, model in (("exact", None), ("inexact", perturbed_model(M))):
        runs = {}
        for itag, internal, ld in (("off", "off", None), ("on", "impedance", lam_d)):
            sim, q, v, tm, tc, C = make(M, mc, q0, model=model, motion=True, lam_d=ld, internal=itag and internal,
                                        diag_every=10)
            v0 = initial_velocity(tm, q, 0.05)
            wall = run(sim, q, v0, T, dt)
            lg = sim.logs
            eI = np.array([np.linalg.norm(d.get("eI", np.zeros(1))) for d in sim.diags])
            lamI = np.array([np.linalg.norm(d.get("lamI", np.zeros(1))) for d in sim.diags])
            runs[itag] = dict(t=lg["t"], q=lg["q"], v=lg["v"], eI=eI, lamI=lamI, td=np.array([d["t"] for d in sim.diags]))
            print(f"  decoupling {model_tag} internal={itag}: wall {wall:.0f}s |lamI|max {lamI.max():.1f}", flush=True)
        dq = np.abs(runs["on"]["q"] - runs["off"]["q"]).max(axis=1)
        dv = np.abs(runs["on"]["v"] - runs["off"]["v"]).max(axis=1)
        results[model_tag] = dict(t=runs["on"]["t"], dq=dq, dv=dv, lamI_on=runs["on"]["lamI"],
                                  lamI_off=runs["off"]["lamI"], td=runs["on"]["td"], q_off=runs["off"]["q"])
        print(f"decoupling {model_tag}: max |dq| {dq.max():.3e} max |dv| {dv.max():.3e}", flush=True)
    # deviation of the inexact-model motion from the exact-model motion (for scale)
    dev = np.abs(results["inexact"]["q_off"] - results["exact"]["q_off"]).max(axis=1)
    np.savez(os.path.join(OUT, "decoupling.npz"), t=results["exact"]["t"], dq_exact=results["exact"]["dq"],
             dv_exact=results["exact"]["dv"], dq_inexact=results["inexact"]["dq"],
             dv_inexact=results["inexact"]["dv"], lamI_on=results["exact"]["lamI_on"],
             lamI_off=results["exact"]["lamI_off"], td=results["exact"]["td"], dev_model=dev)


def exp_zoh(Ts, T=1.0, dt=2.5e-4):
    """Sampled-data decoupling: motion difference between psi = 0 and a large internal command under a ZOH."""
    M, mc, q0 = base_team()
    tmp = team.TeamModel(mc, 4)
    tmp.set_mode(setup.grasp_mode(tmp, q0))
    lamA = squeeze_pattern(tmp, 200.0, 6.0)

    def lam_d(t):
        return (0.6 + 0.4 * np.sin(2 * np.pi * 1.1 * t)) * lamA
    runs = {}
    for itag, internal, ld in (("off", "off", None), ("on", "impedance", lam_d)):
        sim, q, v, tm, tc, C = make(M, mc, q0, motion=True, lam_d=ld, internal=internal, Ts=Ts, diag_every=1000000,
                                    log_every=4)
        v0 = initial_velocity(tm, q, 0.05)
        wall = run(sim, q, v0, T, dt)
        runs[itag] = sim.logs
        print(f"  zoh Ts={Ts * 1e3:.1f}ms internal={itag}: wall {wall:.0f}s", flush=True)
    dq = np.abs(runs["on"]["q"] - runs["off"]["q"]).max(axis=1)
    pb = slice(tm.beam_qadr, tm.beam_qadr + 3)
    dp = np.linalg.norm(runs["on"]["q"][:, pb] - runs["off"]["q"][:, pb], axis=1)
    np.savez(os.path.join(OUT, f"zoh_{Ts * 1e3:.2f}.npz"), t=runs["on"]["t"], dq=dq, dp=dp, Ts=Ts)
    print(f"zoh Ts={Ts * 1e3:.2f} ms: max |dq| {dq.max():.3e}, max |dp_payload| {dp.max():.3e}", flush=True)


def exp_rates(T=3.0, dt=1e-3):
    """Theorem 2: regulation from initial errors against a compliant support; measured decay vs certified."""
    M, mc, q0 = base_team()
    Ke, De = np.array([2.0e4, 2.0e4, 2.0e4]), np.array([300.0, 300.0, 300.0])
    env = setup.SpringSupport(Ke=Ke, De=De, t_on=0.0)
    tmp = team.TeamModel(mc, 4)
    tmp.set_mode(setup.grasp_mode(tmp, q0))
    lamA = squeeze_pattern(tmp, 60.0, 0.0)
    sim, q, v, tm, tc, C = make(M, mc, q0, motion=False, lam_d=lambda t: lamA, env=_env_wrap(env))
    # initial errors: velocity error, payload/pelvis pose errors (references offset), compliant-frame and
    # internal-impedance states
    v0 = initial_velocity(tm, q, 0.05)
    R = C.refs
    p0, R0 = R.ptraj(0.0)[:2]
    pd = p0 + np.array([0.0, 0.02, 0.0])        # desired pose pressed 2 cm into the compliant support
    R.ptraj = RF.PoseTraj([(0.0, pd, R0)])
    for r in tm.robots:
        R.pelvis_offset[r.i] = RF.VecTraj([(0.0, np.array([0.0, 0.0, 0.01]))])
    C.ext[0] = C.ext[0] + np.array([0.0, -0.01, 0.01])  # compliant frame offset (chi(0) != 0)
    C.z = 20.0 * RNG.standard_normal(tm.nI) / np.sqrt(tm.nI)
    wall = run(sim, q, v0, T, dt)
    D = sim.diags
    t = np.array([d["t"] for d in D])
    sM = np.array([d["sM"] for d in D])
    eI = np.array([d["eI_norm"] for d in D])
    # xi: payload pose error w.r.t. the compliant frame, pelvis pose errors, joint errors
    xi = []
    for d in D:
        ri = d["rinfo"]
        parts = [np.asarray(vv) for kk, vv in ri.items()]
        xi.append(np.linalg.norm(np.concatenate(parts)))
    xi = np.array(xi)
    # target-interaction error chi: compliant frame vs its equilibrium p_c* = (Kp pd + Ke pe)/(Kp + Ke)
    Kp = np.diag(C.P.Kp)
    pe = env.pe if env.pe is not None else p0
    pcs = (Kp * pd + Ke * pe) / (Kp + Ke)
    xs = sim.logs["x"]
    chi = np.array([np.linalg.norm(np.r_[x[0:3] - pcs, x[12:18]]) for x in xs])
    tl = sim.logs["t"]
    ks, _ = kappa_s(tm, C.K, sim.logs["q"][::100])
    gam = min(C.P.gp, C.P.go, C.P.gb_p, C.P.gb_o, C.P.gj)
    mz, dz, kz = C.P.mz, C.P.dz, C.P.kz
    roots = np.roots([mz, dz, kz + 1.0])
    kz_rate = -np.max(roots.real)
    Mc, Dc = np.diag(C.P.Mc), np.diag(C.P.Dc)
    rt = []
    for ax in range(3):
        rr = np.roots([Mc[ax], Dc[ax] + De[ax], Kp[ax] + Ke[ax]])
        rt.append(-np.max(rr.real))
    Ko = np.diag(C.P.Ko)
    for ax in range(3):
        rr = np.roots([Mc[3 + ax], Dc[3 + ax], Ko[ax]])
        rt.append(-np.max(rr.real))
    kT = float(np.min(rt))
    np.savez(os.path.join(OUT, "rates.npz"), t=t, sM=sM, eI=eI, xi=xi, chi=chi, tl=tl, kappa_s=ks, kappa_xi=gam,
             kappa_z=kz_rate, kappa_T=kT)
    print(f"rates: kappa_s {ks:.3f} kappa_xi {gam:.2f} kappa_z {kz_rate:.2f} kappa_T {kT:.2f}  wall {wall:.0f}s",
          flush=True)


class WaConst:
    """Constant inconsistency in actuator space (not in R(L))."""

    def __init__(self, w):
        self.w = w

    def __call__(self, t, tm):
        return self.w


class WaDisp:
    """Request displacement from actuator references, w_a = K^{-1} K_a da (Prop. 1(iii)); 'frozen' keeps
    da = K_a^{-1} Psi^+(q0) psi' constant, 'projected' uses da(t) = K_a^{-1} Psi^+(q(t)) psi'."""

    def __init__(self, K, psi_p, mode):
        self.K = K
        self.psi_p = psi_p
        self.mode = mode
        self.w0 = None
        self.zero_du = (mode == "projected")     # du_a = 0 identically: w_a in K^{-1} N(L^T(q(t)))
        self.ramp = lambda t: min(1.0, t / 0.3) ** 2 * (3 - 2 * min(1.0, t / 0.3))
        self.dramp = lambda t: (6 * (t / 0.3) - 6 * (t / 0.3) ** 2) / 0.3 if t < 0.3 else 0.0

    def dot(self, t, tm):
        return self.dramp(t) * self.w0 if self.w0 is not None else np.zeros(tm.mact)

    def __call__(self, t, tm):
        tm.effort_split(self.K)
        if self.mode == "projected" or self.w0 is None:
            w = (tm.Psip @ self.psi_p) / self.K
            if self.mode == "frozen" and self.w0 is None:
                self.w0 = w.copy()
        if self.mode == "frozen":
            w = self.w0
        return self.ramp(t) * w


def exp_refdisp(T=2.6, dt=1e-3):
    """Proposition 1(iii) and Remark 5: internal command via psi (proposed) versus a reference displacement
    as in [6] (frozen at q0) versus the projected displacement; payload motion bias."""
    M, mc, q0 = base_team()
    tmp = team.TeamModel(mc, 4)
    tmp.set_mode(setup.grasp_mode(tmp, q0))
    psi_p = squeeze_pattern(tmp, 150.0, 0.0)
    K = np.array([setup.rate_gains()[j] for _ in range(4) for j in scene.JOINTS])
    ramp = lambda t: min(1.0, t / 0.3) ** 2 * (3 - 2 * min(1.0, t / 0.3))
    out = {}
    for tag, lam_d, wa, internal in (
            ("nominal", None, None, "off"),
            ("psi", lambda t: ramp(t) * psi_p, None, "feedforward"),
            ("frozen", None, WaDisp(K, psi_p, "frozen"), "off"),
            ("projected", None, WaDisp(K, psi_p, "projected"), "off")):
        sim, q, v, tm, tc, C = make(M, mc, q0, motion=True, lam_d=lam_d, wa=wa, internal=internal, diag_every=10)
        wall = run(sim, q, v.copy(), T, dt)
        lg = sim.logs
        pb = lg["q"][:, tm.beam_qadr:tm.beam_qadr + 3]
        lamI = np.array([d.get("lamI", np.zeros(tm.nI)) for d in sim.diags])
        out[tag] = dict(t=lg["t"], pb=pb, q=lg["q"], lamI=lamI, td=np.array([d["t"] for d in sim.diags]))
        print(f"  refdisp {tag}: wall {wall:.0f}s", flush=True)
    res = {}
    for tag in ("psi", "frozen", "projected"):
        dp = np.linalg.norm(out[tag]["pb"] - out["nominal"]["pb"], axis=1)
        dq = np.abs(out[tag]["q"] - out["nominal"]["q"]).max(axis=1)
        lamI_err = np.linalg.norm(out[tag]["lamI"] - out["nominal"]["lamI"] - psi_p[None, :] *
                                  np.array([ramp(tt) for tt in out[tag]["td"]])[:, None], axis=1)
        res[tag] = (dp, dq, lamI_err)
        print(f"refdisp {tag}: max payload bias {dp.max() * 1e3:.4f} mm, max |dq| {dq.max():.3e}, "
              f"max internal-wrench error {lamI_err.max():.3e} N", flush=True)
    np.savez(os.path.join(OUT, "refdisp.npz"), t=out["nominal"]["t"], td=out["nominal"]["td"],
             **{f"dp_{k}": v[0] for k, v in res.items()}, **{f"dq_{k}": v[1] for k, v in res.items()},
             **{f"eI_{k}": v[2] for k, v in res.items()})


def exp_antagonism(T=4.0, dt=1e-3):
    """Proposition 1(iv): steady internal-wrench error under a constant inconsistency w_a for psi = lambda_d,
    a stiffness-type internal impedance (K_z = I) and integral action (K_z = 0); predicted vs measured.
    Also Remark 3: inertial model error with psi = lambda_d versus K_z = 0."""
    M, mc, q0 = base_team()
    tmp = team.TeamModel(mc, 4)
    tmp.set_mode(setup.grasp_mode(tmp, q0))
    tmp.update(q0, np.zeros(mc.nv))
    K = np.array([setup.rate_gains()[j] for _ in range(4) for j in scene.JOINTS])
    tmp.effort_split(K)
    rng = np.random.default_rng(11)
    w = rng.standard_normal(tmp.mact)
    # keep only the part outside R(L) and scale it so that the antagonism is ~100 N
    LtK = tmp.L.T * K
    w = w - tmp.L @ np.linalg.solve(LtK @ tmp.L, LtK @ w)
    alpha = -(tmp.Mmu @ (K * w))[tmp.gidx]
    w *= 100.0 / np.linalg.norm(tmp.VIp @ tmp.PI @ alpha)
    lam_const = squeeze_pattern(tmp, 40.0, 0.0)
    out = {}
    cases = [("ff", "feedforward", 0.0, None), ("kz1", "impedance", 1.0, None), ("int", "impedance", 0.0, None),
             ("ff_model", "feedforward", 0.0, "model"), ("int_model", "impedance", 0.0, "model")]
    for tag, internal, kz, err in cases:
        wa = WaConst(w) if err is None else None
        model = perturbed_model(M, rel=0.0, payload=1.2) if err == "model" else None
        sim, q, v, tm, tc, C = make(M, mc, q0, model=model, motion=False, lam_d=lambda t: lam_const, wa=wa,
                                    internal=internal, kz=kz, diag_every=5)
        wall = run(sim, q, v.copy(), T, dt)
        D = sim.diags
        t = np.array([d["t"] for d in D])
        eI = np.array([d["eI_norm"] for d in D])
        out[tag] = dict(t=t, eI=eI)
        # predicted steady value: |V_I^+ alpha_inf| scaled by K_z (K_z + I)^{-1}
        c = sim.ctrl
        tcm = sim.tc
        a_inf = None
        if err is None:
            tcm.update(sim.logs["q"][-1], sim.logs["v"][-1])
            tcm.effort_split(c.K)
            wperp = c.last["wperp"]
            a_inf = np.linalg.norm(tcm.VIp @ (-(tcm.Mmu @ (c.K * wperp))[tcm.gidx]))
        out[tag]["a_inf"] = a_inf
        print(f"antagonism {tag}: wall {wall:.0f}s final |eI| {eI[-1]:.3f} (|V_I^+ alpha_inf| = {a_inf})", flush=True)
    np.savez(os.path.join(OUT, "antagonism.npz"), **{f"t_{k}": v["t"] for k, v in out.items()},
             **{f"eI_{k}": v["eI"] for k, v in out.items()},
             a_inf=np.array([out["ff"]["a_inf"]]))


def exp_corollary(T=3.0, dt=1e-3, every=10):
    """Corollary 1: run the setting of exp_rates (squeeze and z(0) scaled with the girder mass, so that lambda* keeps
    its relative margins in the palm cones) and evaluate, along the solution, the margins rho_k of
    lambda* in the contact-wrench cones of all soles and palms and the left side of the condition,
    |[mu(Psi^+ V_I z)]_k| + xibar_k |s|_M (norms in (f, m / 0.1 m)); xibar_k is the maximum of
    |Xi_k M^{-1/2}| over the visited states. Also checks lambda = lambda* - mu(Psi^+ V_I z) + Xi s."""
    from teamimp import share
    M, mc, q0 = base_team()
    Ke, De = np.array([2.0e4, 2.0e4, 2.0e4]), np.array([300.0, 300.0, 300.0])
    env = setup.SpringSupport(Ke=Ke, De=De, t_on=0.0)
    tmp = team.TeamModel(mc, 4)
    tmp.set_mode(setup.grasp_mode(tmp, q0))
    # squeeze and initial internal-impedance state scale with the girder mass (60 N and 20 N rms for the 183 kg
    # steel girder), so that lambda* keeps the same relative margins in the palm cones
    ls = float(M.mass) / 183.36
    lamA = squeeze_pattern(tmp, 60.0 * ls, 0.0)
    sim, q, v, tm, tc, C = make(M, mc, q0, motion=False, lam_d=lambda t: lamA, env=_env_wrap(env), diag_every=every)
    v0 = initial_velocity(tm, q, 0.05)
    R = C.refs
    p0, R0 = R.ptraj(0.0)[:2]
    pd = p0 + np.array([0.0, 0.02, 0.0])
    R.ptraj = RF.PoseTraj([(0.0, pd, R0)])
    for r in tm.robots:
        R.pelvis_offset[r.i] = RF.VecTraj([(0.0, np.array([0.0, 0.0, 0.01]))])
    C.ext[0] = C.ext[0] + np.array([0.0, -0.01, 0.01])
    C.z = 20.0 * ls * np.random.default_rng(7).standard_normal(tm.nI) / np.sqrt(tm.nI)
    wall = run(sim, q, v0, T, dt)
    lg = sim.logs
    K = C.K
    Ls = share.L_SCALE
    Dsc = np.r_[np.ones(3), np.ones(3) / Ls]
    rows = tm.rows
    idx = list(range(0, len(lg["t"]), every))
    recs = []
    for k in idx:
        t = lg["t"][k]
        tm.update(lg["q"][k], lg["v"][k])
        C.unpack(lg["x"][k])
        h = env(tm)
        C.effort(t, h)
        c = C.last
        lam_star = c["lam_r"] - tm.Mmu @ (tm.Psip @ c["lam_d"] + c["fF"])
        mu_z = tm.Mmu @ (tm.Psip @ (tm.VI @ C.z))
        s = tm.v - c["vr"]
        sM = float(np.sqrt(max(s @ tm.M @ s, 0.0)))
        # Xi (full matrix) and Xi s
        n = tm.n
        Cm = tm.coriolis(np.eye(n))
        cM = sla.cho_factor(tm.M)
        MinvAt = sla.cho_solve(cM, tm.A.T)
        S = tm.A @ MinvAt
        BKBt = (tm.B * K) @ tm.B.T
        Xi = np.linalg.solve(S, tm.A @ sla.cho_solve(cM, Cm + tm.Dv + BKBt) - tm.Ad)
        ev, U = np.linalg.eigh(tm.M)
        Mmh = (U / np.sqrt(ev)) @ U.T
        XiM = Xi @ Mmh
        lam = lg["lam"][k]
        res = np.abs(lam - (lam_star - mu_z + Xi @ s)).max()
        per = []
        for j, (kind, i, sd) in enumerate(rows):
            sl = slice(6 * j, 6 * j + 6)
            spec = share.SOLE if kind == "foot" else share.PALM
            rho = share.margin(lam_star[sl], spec)[0]
            rho_act = share.margin(lam[sl], spec)[0]
            xin = np.linalg.norm(Dsc[:, None] * XiM[sl], 2)
            per.append((rho, np.linalg.norm(Dsc * mu_z[sl]), xin, rho_act))
        recs.append(dict(t=t, sM=sM, res=res, per=np.array(per)))
    P = np.array([r["per"] for r in recs])            # (nt, nconn, 4)
    t = np.array([r["t"] for r in recs])
    sM = np.array([r["sM"] for r in recs])
    xibar = P[:, :, 2].max(axis=0)                    # max over visited states
    lhs = P[:, :, 1] + xibar[None, :] * sM[:, None]
    gap = P[:, :, 0] - lhs                            # > 0: condition of Corollary 1 holds
    ok_all = (gap > 0).all(axis=1)
    t_cert = None
    for kk in range(len(t)):
        if ok_all[kk:].all():
            t_cert = float(t[kk])
            break
    out = dict(t_cert=t_cert, min_rho_star=float(P[:, :, 0].min()), min_rho_actual=float(P[:, :, 3].min()),
               max_identity_residual=float(max(r["res"] for r in recs)), xibar_max=float(xibar.max()),
               min_gap_after=float(gap[t >= (t_cert if t_cert is not None else np.inf)].min()) if t_cert is not None else None,
               nconn=int(P.shape[1]), wall=wall)
    np.savez(os.path.join(OUT, "corollary.npz"), t=t, sM=sM, rho=P[:, :, 0], muz=P[:, :, 1], xin=P[:, :, 2],
             rho_act=P[:, :, 3], xibar=xibar, gap=gap)
    print("corollary:", json.dumps(out), flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "identities":
        exp_identities()
    elif what == "decoupling":
        exp_decoupling()
    elif what == "zoh":
        exp_zoh(float(sys.argv[2]) * 1e-3)
    elif what == "rates":
        exp_rates()
    elif what == "refdisp":
        exp_refdisp()
    elif what == "antagonism":
        exp_antagonism()
    elif what == "corollary":
        exp_corollary()
