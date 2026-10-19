"""Remark 3 with contact physics: pose bias caused by a girder-mass error of the controller model, at rest.

From the post-lift state of the N = 4 mission (double support, h = 0), the girder model of the controller is
changed to m_hat (inertia scaled with the mass) and all references are held for T s. The measured bias of the
girder relative to the compliant frame (mean over the last second) is compared with the prediction of
Remark 3, xi = Gamma^{-1} u_q, u_q = -(L^T K L + E^T Dv E)^{-1} E^T tau~, evaluated at the initial state.

Usage: python experiments/mujoco_bias_sweep.py <state.pkl> [model masses, e.g. 183.36,178,173,168,163,158]
"""
import os
import sys
import json
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from teamimp import mission as MS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mission")
G0 = 9.81


def set_model_mass(tm, m_hat):
    m = tm.m
    b = tm.beam
    s = m_hat / float(m.body_mass[b])
    m.body_mass[b] *= s
    m.body_inertia[b] *= s
    m.body_subtreemass[b] = m.body_mass[b]
    m.body_subtreemass[0] = float(np.sum(m.body_mass))


def predict(M, m_hat):
    tm, R, C = M.tm, M.R, M.C
    tm.update(R.d.qpos.copy(), np.zeros(R.d.qvel.shape), need_dot=False)
    L, E, Dv = tm.L, tm.E, tm.Dv
    Mred = L.T @ (R.K[:, None] * L) + E.T @ Dv @ E
    rhs = np.zeros(E.shape[1])
    rhs[2] = (M.mass - m_hat) * G0
    uq = np.linalg.solve(Mred, rhs)
    return uq[:3] / float(C.P.gp), uq[3:6] / float(C.P.go)


def run(state_path, m_hat, T=6.0):
    with open(state_path, "rb") as f:
        M = pickle.load(f)
    M.cfg.verbose = False
    xi_p, xi_o = predict(M, m_hat)
    set_model_mass(M.tm, m_hat)
    rec = []
    orig = M._log

    def _log():
        orig()
        L = M.logs[-1]
        rec.append((L["t"], L["beam_c"] - L["beam"][:3], float(np.max(L["sat"]))))
    M._log = _log
    M.run([MS.Hold(T, "hold")])
    t = np.array([r[0] for r in rec])
    d = np.array([r[1] for r in rec])
    last = t > t[-1] - 1.0
    return dict(model_mass=m_hat, true_mass=float(M.mass), predicted_mm=(1e3 * xi_p).tolist(),
                predicted_rot_deg=float(np.degrees(np.linalg.norm(xi_o))),
                measured_mm=(1e3 * d[last].mean(0)).tolist(), measured_std_mm=(1e3 * d[last].std(0)).tolist(),
                sat_max=float(max(r[2] for r in rec))), t, d


def main():
    state = sys.argv[1]
    masses = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else \
        [183.36, 178.0, 173.0, 168.0, 163.0, 158.0]
    res, arrays = [], {}
    for mh in masses:
        r, t, d = run(state, mh)
        res.append(r)
        arrays[f"t_{mh:g}"], arrays[f"d_{mh:g}"] = t, d
        print(json.dumps(r), flush=True)
    np.savez(os.path.join(OUT, "mujoco_bias_sweep.npz"), **arrays)
    with open(os.path.join(OUT, "mujoco_bias_sweep.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
