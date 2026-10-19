"""Pose bias caused by a payload-mass error of the controller model (Remark 3, Section V).

Prediction: at rest with constant references, the requests settle at
    u_q = -(L^T K L + E^T Dv E)^{-1} E^T tau~,   tau~ = tau_model - tau,
and bias xi by Gamma^{-1} u_q; for a payload-mass error, E^T tau~ is the gravity-force error
(m_model - m) g on the payload rows. It is evaluated at the double-support state at the end of the lift.

Measurement: mean of p_c - p (compliant frame minus girder position) over the last 1 s of each double-support
weight shift of the carry (the joints keep moving, so static joint friction does not hold the bias), for the
runs with a girder model of m_model kg and, as the reference, the exact-model run.

Usage: python experiments/model_error_offset.py [state.pkl] [kz0]
(the state is produced by experiments/save_state.py; default results/mission/state_lift_al_n4_kz0.pkl, the 63 kg
aluminium girder, runs with K_z = I; "kz0": the runs with K_z = 0 (model_error_offset_kz0.json); with the 183 kg
steel state, the steel runs are evaluated)
"""
import os
import sys
import json
import pickle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np

MI = os.path.join(ROOT, "results", "mission")
G0 = 9.81
RUNS_STEEL = [("n4_m200_kz0", 200.0), ("n4_m190_kz0", 190.0), ("n4_kz0", None), ("n4_m178_kz0", 178.0),
              ("n4_m174_kz0", 174.0), ("n4_m170_kz0", 170.0), ("n4_m165_kz0", 165.0), ("n4_m160_kz0", 160.0),
              ("n4_m155_kz0", 155.0), ("n4_merr_kz0", 150.0)]
# aluminium girder: K_z = I (with K_z = 0, the girder is lost for all model errors tested, see Section V-E);
# the bias does not depend on the internal channel (Theorem 1(d))
RUNS_AL = [("al_n4_m70", 70.0), ("al_n4_m66", 66.0), ("al_n4", None), ("al_n4_m60", 60.0), ("al_n4_m55", 55.0),
           ("al_n4_m50", 50.0), ("al_n4_m40", 40.0)]
RUNS_AL_KZ0 = [("al_n4_m70_kz0", 70.0), ("al_n4_m66_kz0", 66.0), ("al_n4_kz0", None), ("al_n4_m60_kz0", 60.0),
               ("al_n4_m55_kz0", 55.0), ("al_n4_m50_kz0", 50.0), ("al_n4_m40_kz0", 40.0), ("al_n4_m30_kz0", 30.0)]


def bias_per_kg(state_path):
    """Predicted p_c - p (m) and rotation (rad) per kg of model-mass deficit (the bias is linear in it)."""
    with open(state_path, "rb") as f:
        M = pickle.load(f)
    tm, R, C = M.tm, M.R, M.C
    tm.update(R.d.qpos.copy(), np.zeros(R.d.qvel.shape), need_dot=False)
    L, E, Dv = tm.L, tm.E, tm.Dv
    Mred = L.T @ (R.K[:, None] * L) + E.T @ Dv @ E           # L^T K L + E^T Dv E
    rhs = np.zeros(E.shape[1])
    rhs[2] = G0                                                # -E^T tau~ for 1 kg: payload rows first, force first
    uq = np.linalg.solve(Mred, rhs)
    return uq[:3] / float(C.P.gp), uq[3:6] / float(C.P.go), float(M.mass)


def measure(tag):
    path = os.path.join(MI, f"{tag}_logs.pkl")
    if not os.path.exists(path):
        return None, None
    with open(path, "rb") as f:
        D = pickle.load(f)
    L = D["logs"]
    ends = {}
    for x in L:
        if x["phase"].startswith("carry") and x["phase"].endswith("shift"):
            ends[x["phase"]] = max(ends.get(x["phase"], -np.inf), x["t"])
    tf = D["failed"][0] if D["failed"] else np.inf
    d = np.array([x["beam_c"] - x["beam"][:3] for x in L
                  if x["phase"] in ends and ends[x["phase"]] - 1.0 < x["t"] < min(ends[x["phase"]], tf) + 1e-9
                  and ends[x["phase"]] < tf and "beam_c" in x])
    res = None
    if len(d):
        res = dict(samples=int(len(d)), mean_mm=(1e3 * d.mean(0)).tolist(),
                   z_range_mm=[float(1e3 * d[:, 2].min()), float(1e3 * d[:, 2].max())])
    return res, D["failed"]


def main():
    args = [a for a in sys.argv[1:] if a != "kz0"]
    kz0 = "kz0" in sys.argv[1:]
    state = args[0] if args else os.path.join(MI, "state_lift_al_n4_kz0.pkl")
    dp, dr, m_true = bias_per_kg(state)
    RUNS = (RUNS_AL_KZ0 if kz0 else RUNS_AL) if m_true < 100.0 else RUNS_STEEL
    mm = [m for _, m in RUNS if m is not None] + [m_true]
    masses = np.linspace(min(mm) - 5.0, max(mm) + 5.0, 61)
    out = dict(true_mass=m_true, bias_per_kg_mm=(1e3 * dp).tolist(), rot_per_kg_deg=float(np.degrees(np.linalg.norm(dr))),
               prediction_curve=dict(mass=masses.tolist(), z_mm=(1e3 * dp[2] * (m_true - masses)).tolist()), runs=[])
    for tag, mhat in RUNS:
        mhat = m_true if mhat is None else mhat
        meas, failed = measure(tag)
        row = dict(tag=tag, model_mass=mhat, predicted_z_mm=float(1e3 * dp[2] * (m_true - mhat)),
                   predicted_mm=(1e3 * dp * (m_true - mhat)).tolist(), completed=failed is None,
                   fail_time=None if failed is None else float(failed[0]),
                   measured_mm=None if meas is None else meas["mean_mm"],
                   measured_z_mm=None if meas is None else meas["mean_mm"][2],
                   z_range_mm=None if meas is None else meas["z_range_mm"])
        if meas is not None or os.path.exists(os.path.join(MI, f"{tag}_logs.pkl")):
            out["runs"].append(row)
            print(f"{tag:12s} model {mhat:6.2f} kg: predicted z {row['predicted_z_mm']:6.2f} mm, measured "
                  f"{row['measured_z_mm'] if meas else None} mm, completed={row['completed']}", flush=True)
    with open(os.path.join(MI, "model_error_offset_kz0.json" if kz0 else "model_error_offset.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
