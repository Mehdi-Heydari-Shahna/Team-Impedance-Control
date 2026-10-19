"""Decoupling under contact physics (Theorem 1(d) outside its assumptions): from the post-lift state of the
N = 4 mission, the desired girder pose moves by (2, 2, 3) cm and back in double support (min-jerk, 3 s, then a
1 s hold), twice: without and with an additional internal command psi' in N(G) (palm normal-force
redistribution of amplitude amp N per palm, ramped in over 0.5 s and modulated by 1 + 0.5 sin(2 pi 0.7 t)).
Reports the largest differences of the girder pose and of the joint positions, the realized change of the palm
forces and of the internal wrench, the first actuator saturation, and the smallest CoP margin of the measured
connection wrenches (palms and soles) with the connection where it occurs.

Usage: python experiments/mujoco_decoupling.py <state.pkl> [amplitude_N] [nofriction]
("nofriction" removes the joint friction of the plant in both runs, as in mujoco_push.py)
(the state is produced by running the mission to the end of the lift and pickling the Mission object)
"""
import os
import sys
import json
import pickle
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from teamimp import mission as MS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mission")


def internal_pattern(tm, amp):
    """Unit internal wrench in N(G) dominated by palm normal forces (grasp-frame z), scaled to amp (N)."""
    ng = tm.ng
    best, best_frac = None, -1.0
    for signs in itertools.product((1.0, -1.0), repeat=ng - 1):
        raw = np.zeros(6 * ng)
        sg = (1.0,) + signs
        for j in range(ng):
            raw[6 * j + 2] = sg[j]
        lam = tm.PI @ raw
        nz = np.linalg.norm(lam[2::6])
        frac = nz / max(np.linalg.norm(lam), 1e-12)
        if frac > best_frac and nz > 1e-6:
            best, best_frac = lam, frac
    return amp * best / np.max(np.abs(best[2::6])), best_frac


class Sway(MS.Phase):
    """Double support: the desired girder pose moves by dp and back (min-jerk, T/2 each); feet and pelvises
    keep their references."""
    name = "sway"

    def __init__(self, dp, T):
        self.dp, self.T = dp, T

    def start(self, M, t):
        self.t1 = t + self.T
        M.refs.pay.move(t, self.T / 2, dp=self.dp)
        M.refs.pay.move(t + self.T / 2, self.T / 2, dp=-self.dp)

    def done(self, M, t):
        return t >= self.t1


def run(state_path, amp, T=3.0, D=0.02, nofriction=False):
    M = pickle.load(open(state_path, "rb"))
    if nofriction:
        M.R.m.dof_frictionloss[:] = 0.0
    M.cfg.verbose = False
    tm = M.tm
    t0 = M.R.t
    if amp > 0:
        psi_p, frac = internal_pattern(tm, amp)

        def lam_extra(t):
            tau = t - t0
            r = min(1.0, max(0.0, tau / 0.5))
            r = r * r * (3 - 2 * r)
            return r * (1.0 + 0.5 * np.sin(2 * np.pi * 0.7 * tau)) * psi_p
        M.refs.lam_d = lam_extra
    else:
        frac = None
    rec = []
    orig = M._log

    def _log():
        orig()
        L = M.logs[-1]
        g = {(c[1], c[2]): c[4][:3].copy() for c in L["conn"] if c[0] == "grasp"}
        rec.append(dict(t=L["t"], beam=L["beam"].copy(), qpos=L["qpos"].copy(), palms=g,
                        eI=np.linalg.norm(L["eI"]) if "eI" in L else np.nan,
                        lamI=L.get("lamI", np.zeros(1)).copy(), sat=float(np.max(L["sat"])),
                        sat_arg=int(np.argmax(L["sat"])), mode=L["mode"],
                        cop=[c[6][1] for c in L["conn"]],
                        cop_id=[f"{c[0]}{c[1]}{c[2]}" for c in L["conn"]]))
    M._log = _log
    ph = [Sway(np.array([D, D, 1.5 * D]), T), MS.Hold(1.0, "hold")]
    M.run(ph)
    return rec, frac, M


def main():
    state = sys.argv[1]
    amp = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
    nof = len(sys.argv) > 3 and sys.argv[3] == "nofriction"
    suf = "_nofriction" if nof else ""
    r0, _, M0 = run(state, 0.0, nofriction=nof)
    r1, frac, M1 = run(state, amp, nofriction=nof)
    n = min(len(r0), len(r1))
    t = np.array([r0[k]["t"] for k in range(n)])
    dp = np.array([np.linalg.norm(r1[k]["beam"][:3] - r0[k]["beam"][:3]) for k in range(n)])
    # girder orientation difference (angle of the relative rotation)
    def qang(a, b):
        d = abs(float(np.dot(a, b)))
        return 2.0 * np.arccos(min(1.0, d))
    dth = np.array([qang(r1[k]["beam"][3:], r0[k]["beam"][3:]) for k in range(n)])
    tm = M0.tm
    jq = []
    for r in tm.robots:
        jq += [r.qpos_adr[j] for j in r.qpos_adr]
    jq = np.array(sorted(set(jq)))
    dq = np.array([np.max(np.abs(r1[k]["qpos"][jq] - r0[k]["qpos"][jq])) for k in range(n)])
    dpal = []
    for k in range(n):
        keys = set(r0[k]["palms"]) & set(r1[k]["palms"])
        dpal.append(max([abs(r1[k]["palms"][kk][2] - r0[k]["palms"][kk][2]) for kk in keys] or [0.0]))
    dpal = np.array(dpal)
    dlamI = np.array([np.linalg.norm(r1[k]["lamI"] - r0[k]["lamI"]) if r1[k]["lamI"].shape == r0[k]["lamI"].shape
                      else np.nan for k in range(n)])
    out = dict(amp=amp, frac_normal=frac, T=float(t[-1] - t[0]), max_dp_mm=float(1e3 * dp.max()),
               max_dtheta_deg=float(np.degrees(dth.max())), max_dq_joint_rad=float(dq.max()),
               max_dpalm_N=float(dpal.max()), max_dlamI_N=float(np.nanmax(dlamI)),
               sat_max_nominal=float(max(r["sat"] for r in r0)), sat_max_psi=float(max(r["sat"] for r in r1)),
               modes_equal=bool(all(r0[k]["mode"] == r1[k]["mode"] for k in range(n))),
               girder_travel_mm=float(1e3 * np.linalg.norm(r0[n - 1]["beam"][:3] - r0[0]["beam"][:3])))
    print(json.dumps(out, indent=1))
    sat0 = np.array([r0[k]["sat"] for k in range(n)])
    sat1 = np.array([r1[k]["sat"] for k in range(n)])
    arg1 = np.array([r1[k]["sat_arg"] for k in range(n)])
    cop1 = np.array([min(r1[k]["cop"]) for k in range(n)])
    cop0 = np.array([min(r0[k]["cop"]) for k in range(n)])
    cop1_id = [r1[k]["cop_id"][int(np.argmin(r1[k]["cop"]))] for k in range(n)]
    k_neg = np.where(cop1 < 0)[0]
    k_sat = int(np.argmax(sat1 >= 0.999)) if (sat1 >= 0.999).any() else n
    out["t_first_saturation"] = float(t[k_sat] - t[0]) if k_sat < n else None
    out["actuator_first_saturation"] = (int(arg1[k_sat]) // 32, int(arg1[k_sat]) % 32) if k_sat < n else None
    out["max_dp_mm_before_saturation"] = float(1e3 * dp[:k_sat].max()) if k_sat > 0 else None
    out["max_dq_before_saturation"] = float(dq[:k_sat].max()) if k_sat > 0 else None
    out["max_dpalm_before_saturation"] = float(dpal[:k_sat].max()) if k_sat > 0 else None
    out["max_dlamI_before_saturation"] = float(np.nanmax(dlamI[:k_sat])) if k_sat > 0 else None
    out["min_cop_margin_psi_mm"] = float(1e3 * np.min(cop1))
    out["min_cop_margin_nominal_mm"] = float(1e3 * np.min(cop0))
    out["min_cop_margin_psi_at"] = cop1_id[int(np.argmin(cop1))]
    out["t_first_cop_edge"] = float(t[k_neg[0]] - t[0]) if len(k_neg) else None
    out["cop_edge_connections"] = sorted(set(cop1_id[k] for k in k_neg))
    out["time_at_cop_edge_s"] = float(len(k_neg) * (t[1] - t[0]))
    print(json.dumps({k: out[k] for k in out if k.startswith(("t_first", "actuator", "max_dp_mm_b", "max_dq_b",
                                                             "max_dpalm_b", "max_dlamI_b", "min_cop", "t_first_cop",
                                                             "cop_edge", "time_at_cop"))}, indent=1))
    # flicker-robust versions: palm forces averaged over 10 ms; a connection counts as at its CoP edge only if it
    # stays there for at least 10 ms (soft contacts of lightly loaded palms separate for single steps)
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(0.01 / dt)))
    keys = sorted(set(r0[0]["palms"]) & set(r1[0]["palms"]))
    P0 = np.array([[r0[k]["palms"].get(kk, np.zeros(3))[2] for kk in keys] for k in range(n)])
    P1 = np.array([[r1[k]["palms"].get(kk, np.zeros(3))[2] for kk in keys] for k in range(n)])
    avg = lambda X: np.array([np.convolve(np.pad(c, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w,
                                          mode="valid") for c in X.T]).T
    out["max_dpalm_avg10ms_N"] = float(np.max(np.abs(avg(P1) - avg(P0))))
    C1 = np.array([r1[k]["cop"] for k in range(n)])
    ids = r1[0]["cop_id"]
    edge_t, edge_ids = 0.0, set()
    for j in range(C1.shape[1]):
        neg = C1[:, j] < 0
        k = 0
        while k < n:
            if neg[k]:
                e = k
                while e + 1 < n and neg[e + 1]:
                    e += 1
                if e - k + 1 >= w:
                    edge_t += (e - k + 1) * dt
                    edge_ids.add(ids[j])
                k = e + 1
            else:
                k += 1
    out["time_at_cop_edge_10ms_s"] = float(edge_t)
    out["cop_edge_10ms_connections"] = sorted(edge_ids)
    print("flicker-robust:", out["max_dpalm_avg10ms_N"], out["time_at_cop_edge_10ms_s"], out["cop_edge_10ms_connections"])
    np.savez(os.path.join(OUT, f"mujoco_decoupling_{int(amp)}{suf}.npz"), t=t, dp=dp, dth=dth, dq=dq, dpal=dpal,
             dlamI=dlamI, sat0=sat0, sat1=sat1, arg1=arg1, cop0=cop0, cop1=cop1)
    json.dump(out, open(os.path.join(OUT, f"mujoco_decoupling_{int(amp)}{suf}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
