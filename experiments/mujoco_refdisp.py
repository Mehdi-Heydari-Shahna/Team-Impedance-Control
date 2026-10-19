"""Internal command versus reference displacement with contact physics (Proposition 1(iii), Remark 5).

From the post-lift state of the N = 4 mission (double support), the desired girder pose moves by (2, 2, 3) cm
and back (min-jerk, 3 s, then a 1 s hold), as in experiments/mujoco_decoupling.py. The internal-impedance
state is frozen in all runs, so each squeeze below acts as a feedforward and is not corrected. The same squeeze
(palm normal-force redistribution, amplitude amp N, ramped in over 0.5 s) is issued in four ways:
  psi        internal command psi' added to lambda_d (proposed);
  hand       displacement of the hand references along the palm normals, mapped to the arm joints by the
             pseudoinverse of the hand Jacobian at the current configuration (as in [6]); requests from actuator
             references with K_a = K add w_a = da;
  frozen     actuator-reference displacement that issues psi' at the initial configuration, held constant;
  projected  displacement da(t) = K^{-1} Psi^+(q(t)) psi', kept in K_a^{-1} N(L^T(q(t))).
The squeeze psi' is the antagonism that the hand displacement creates at the initial configuration, so all
variants start with the same internal wrench. Reports the girder pose deviation from the nominal run (no
squeeze), the realized palm normal-force change and the actuator utilization.

Usage: python experiments/mujoco_refdisp.py <state.pkl> [amplitude_N] [nofriction]
("nofriction" removes the joint friction of the plant in all runs, as in mujoco_push.py)
"""
import os
import sys
import json
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import scipy.linalg as sla
import mujoco

from teamimp import mission as MS, scene

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mission")
ARM = {s: [f"{s}_shoulder_aa", f"{s}_shoulder_ie", f"{s}_shoulder_fe", f"{s}_elbow_fe", f"{s}_wrist_roll",
           f"{s}_wrist_yaw", f"{s}_wrist_pitch"] for s in "lr"}
T_RAMP = 0.5


def ramp(tau):
    r = min(1.0, max(0.0, tau / T_RAMP))
    return r * r * (3 - 2 * r)


def dramp(tau):
    if tau <= 0 or tau >= T_RAMP:
        return 0.0
    r = tau / T_RAMP
    return (6 * r - 6 * r * r) / T_RAMP


class Sway(MS.Phase):
    name = "sway"

    def __init__(self, dp, T):
        self.dp, self.T = dp, T

    def start(self, M, t):
        self.t1 = t + self.T
        M.refs.pay.move(t, self.T / 2, dp=self.dp)
        M.refs.pay.move(t + self.T / 2, self.T / 2, dp=-self.dp)

    def done(self, M, t):
        return t >= self.t1


def hand_displacement(tm, sign):
    """Actuator displacement for unit palm-normal displacements sign[k] of the grasps (pseudoinverse of the
    translational hand Jacobian over the arm joints)."""
    da = np.zeros(tm.mact)
    m, d = tm.m, tm.d
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for j, k in enumerate(tm.grasp_rows):
        _, i, s = tm.rows[k]
        r = tm.robots[i]
        sid = r.site[("hand", s)]
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        po, Ro = tm.mode.grasps[(i, s)]
        Rb = d.xmat[tm.beam].reshape(3, 3)
        n = Rb @ Ro[:, 2]                                      # palm normal (grasp-frame z, world)
        dofs = [r.dof[jn] for jn in ARM[s]]
        J = jp[:, dofs]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-8 * np.eye(3), sign[j] * n)
        for jn, v in zip(ARM[s], dq):
            da[32 * i + scene.JOINTS.index(jn)] = v
    return da


def antagonism(tm, K, wa):
    """alpha = Psi K (w_a - L du_a), du_a = (L^T K L)^{-1} L^T K w_a (Proposition 1(ii))."""
    L = tm.L
    LtK = L.T * K
    du = np.linalg.solve(LtK @ L, LtK @ wa)
    return tm.Psit @ (K * (wa - L @ du)), du


class Displacement:
    """Requests w_a from a displacement of actuator references (K_a = K: w_a = da) and d/dt of the fitted part."""

    def __init__(self, mode, K, t0, da0=None, psi_p=None, sign=None, scale=1.0):
        self.mode, self.K, self.t0 = mode, K, t0
        self.da0, self.psi_p, self.sign, self.scale = da0, psi_p, sign, scale

    def da(self, t, tm):
        if self.mode == "frozen":
            return self.da0
        if self.mode == "projected":
            tm.effort_split(self.K)
            return (tm.Psip @ self.psi_p) / self.K
        return self.scale * hand_displacement(tm, self.sign)

    def w_extra(self, t, tm):
        return ramp(t - self.t0) * self.da(t, tm)

    def dur_extra(self, t, tm, lu, LtK):
        if self.mode == "projected":
            return np.zeros(tm.d_ind)                       # du_a = 0 identically
        wa = self.w_extra(t, tm)
        wad = dramp(t - self.t0) * self.da(t, tm)           # change of da with q neglected (feedforward only)
        L = tm.L
        Ld = tm.Ed[tm.act, :]
        du = sla.cho_solve(lu, LtK @ wa)
        LdtK = Ld.T * self.K
        rhs = LdtK @ wa - (LdtK @ L + LtK @ Ld) @ du + LtK @ wad
        return sla.cho_solve(lu, rhs)


def run(state_path, variant, amp, T=3.0, D=0.02, nofriction=False):
    with open(state_path, "rb") as f:
        M = pickle.load(f)
    if nofriction:
        M.R.m.dof_frictionloss[:] = 0.0
    M.cfg.verbose = False
    tm, C = M.tm, M.C
    K = C.K
    t0 = M.R.t
    C.internal_active = False                              # internal-impedance state frozen in all runs
    tm.update(M.d.qpos.copy(), M.d.qvel.copy())
    tm.effort_split(K)
    # squeeze pattern: palm normal displacements with alternating signs, scaled so that the antagonism at the
    # initial configuration has palm normal components of at most amp
    sign = np.array([1.0 if (tm.rows[k][1] + (tm.rows[k][2] == "r")) % 2 == 0 else -1.0 for k in tm.grasp_rows])
    da_unit = hand_displacement(tm, sign)
    a_unit, _ = antagonism(tm, K, da_unit)
    scale = amp / np.max(np.abs(a_unit[2::6]))
    psi_p = tm.PI @ (scale * a_unit)
    info = dict(scale_rad_per_m=float(scale), psi_normal=[float(x) for x in psi_p[2::6]])
    if variant == "psi":
        base = M.refs.lam_d
        M.refs.lam_d = lambda t: (base(t) if base(t) is not None else 0.0) + ramp(t - t0) * psi_p
    elif variant in ("hand", "frozen", "projected"):
        da0 = (tm.Psip @ psi_p) / K if variant == "frozen" else None
        disp = Displacement(variant, K, t0, da0=da0, psi_p=psi_p, sign=sign, scale=scale)
        M.refs.w_extra = disp.w_extra
        M.refs.dur_extra = disp.dur_extra
        _, du0 = antagonism(tm, K, disp.da(t0, tm))
        info["du_a0_norm"] = float(np.linalg.norm(du0))
    rec = []
    orig = M._log

    def _log():
        orig()
        L = M.logs[-1]
        g = {(c[1], c[2]): c[4][:3].copy() for c in L["conn"] if c[0] == "grasp"}
        rec.append(dict(t=L["t"] - t0, beam=L["beam"].copy(), palms=g, sat=float(np.max(L["sat"])),
                        lamI=L.get("lamI", np.zeros(1)).copy(), pel=L["pel"].copy(),
                        cop=min(c[6][1] for c in L["conn"]), cops=[c[6][1] for c in L["conn"]],
                        cop_ids=[f"{c[0]}{c[1] + 1}{c[2]}" for c in L["conn"]],
                        cop_edge=[f"{c[0]}{c[1] + 1}{c[2]}" for c in L["conn"] if c[6][1] < 0.0]))
    M._log = _log
    M.run([Sway(np.array([D, D, 1.5 * D]), T), MS.Hold(1.0, "hold")])
    return rec, info


def qang(a, b):
    return 2.0 * np.arccos(min(1.0, abs(float(np.dot(a, b)))))


def main():
    state = sys.argv[1]
    amp = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    nof = len(sys.argv) > 3 and sys.argv[3] == "nofriction"
    suf = "_nofriction" if nof else ""
    runs, infos = {}, {}
    for v in ("nominal", "psi", "hand", "frozen", "projected"):
        runs[v], infos[v] = run(state, v, amp, nofriction=nof)
        print(v, "done", flush=True)
    r0 = runs["nominal"]
    out = dict(amp=amp, nofriction=nof, info=infos)
    arrays = {}
    for v in ("psi", "hand", "frozen", "projected"):
        r = runs[v]
        n = min(len(r), len(r0))
        t = np.array([r[k]["t"] for k in range(n)])
        dp = np.array([r[k]["beam"][:3] - r0[k]["beam"][:3] for k in range(n)])
        dth = np.array([qang(r[k]["beam"][3:], r0[k]["beam"][3:]) for k in range(n)])
        dpel = np.array([np.max(np.linalg.norm(r[k]["pel"] - r0[k]["pel"], axis=1)) for k in range(n)])
        dpal = np.array([max([abs(r[k]["palms"][kk][2] - r0[k]["palms"][kk][2])
                              for kk in set(r[k]["palms"]) & set(r0[k]["palms"])] or [0.0]) for k in range(n)])
        hold = t > t[-1] - 0.5
        out[v] = dict(max_dp_mm=float(1e3 * np.linalg.norm(dp, axis=1).max()),
                      end_dp_mm=float(1e3 * np.linalg.norm(dp[hold].mean(0))),
                      max_dtheta_deg=float(np.degrees(dth.max())), max_dpelvis_mm=float(1e3 * dpel.max()),
                      max_dpalm_N=float(dpal.max()), end_dpalm_N=float(dpal[hold].mean()),
                      sat_max=float(max(x["sat"] for x in r[:n])), cop_min_mm=float(1e3 * min(x["cop"] for x in r[:n])),
                      cop_edge=sorted(set(c for x in r[:n] for c in x["cop_edge"])),
                      cop_edge_time_s=float(sum(1 for x in r[:n] if x["cop_edge"]) * np.median(np.diff(t))),
                      cop_edge_first_s=next((float(x["t"]) for x in r[:n] if x["cop_edge"]), None))
        # flicker-robust statistics: palm forces averaged over 10 ms, CoP edges that last at least 10 ms
        dt = float(np.median(np.diff(t)))
        w = max(1, int(round(0.01 / dt)))
        keys = sorted(set(r[0]["palms"]) & set(r0[0]["palms"]))
        P1 = np.array([[r[k]["palms"].get(kk, np.zeros(3))[2] for kk in keys] for k in range(n)])
        P0 = np.array([[r0[k]["palms"].get(kk, np.zeros(3))[2] for kk in keys] for k in range(n)])
        avg = lambda X: np.array([np.convolve(np.pad(c, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w,
                                              mode="valid") for c in X.T]).T
        D10 = np.abs(avg(P1) - avg(P0))
        out[v]["max_dpalm_avg10ms_N"] = float(D10.max())
        out[v]["end_dpalm_avg10ms_N"] = float(D10[hold].max(axis=1).mean())
        C = np.array([r[k]["cops"] for k in range(n)])
        edge_t, edge_ids = 0.0, set()
        for j in range(C.shape[1]):
            neg = C[:, j] < 0
            k = 0
            while k < n:
                if neg[k]:
                    e = k
                    while e + 1 < n and neg[e + 1]:
                        e += 1
                    if e - k + 1 >= w:
                        edge_t += (e - k + 1) * dt
                        edge_ids.add(r[0]["cop_ids"][j])
                    k = e + 1
                else:
                    k += 1
        out[v]["cop_edge_10ms_time_s"] = float(edge_t)
        out[v]["cop_edge_10ms"] = sorted(edge_ids)
        arrays[f"t_{v}"], arrays[f"dp_{v}"], arrays[f"dpal_{v}"] = t, dp, dpal
        print(v, json.dumps(out[v]), flush=True)
    out["nominal"] = dict(sat_max=float(max(x["sat"] for x in r0)), cop_min_mm=float(1e3 * min(x["cop"] for x in r0)),
                          cop_edge=sorted(set(c for x in r0 for c in x["cop_edge"])),
                          girder_travel_mm=float(1e3 * max(np.linalg.norm(x["beam"][:3] - r0[0]["beam"][:3]) for x in r0)))
    np.savez(os.path.join(OUT, f"mujoco_refdisp_{int(amp)}{suf}.npz"), **arrays)
    with open(os.path.join(OUT, f"mujoco_refdisp_{int(amp)}{suf}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
