"""Computation time of the controller (Python, one CPU core): from the post-lift state of the N = 4 mission
(double support), time 300 control steps (contact sensing, fit, required effort, effort split, internal
impedance), the MuJoCo steps in between, and the load-sharing QP (every 10th step). Also prints the dimensions of
the compiled model. Run it on an otherwise idle machine.

Usage: python experiments/timing.py <state.pkl> [repetitions=3]   (writes results/mission/timing.json;
medians over the steps of each repetition, then the median and range over the repetitions)
"""
import os
import sys
import json
import time
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mujoco

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mission")


def measure(path, steps=300):
    with open(path, "rb") as f:
        M = pickle.load(f)
    M.cfg.verbose = False
    tm, R = M.tm, M.R
    dims = dict(n=int(tm.m.nv), d=int(tm.d_ind), m=int(tm.mact), rows=int(tm.A.shape[0]), nI=int(tm.nI))
    tc, ts, tq = [], [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        R.control_step()
        t1 = time.perf_counter()
        for _ in range(R.nsub):
            mujoco.mj_step(R.m, R.d)
        R.t += R.dt_ctrl
        t2 = time.perf_counter()
        M.share.update(tm, M.C.last, R.frc_lim, M.cfg.dt_ctrl)
        t3 = time.perf_counter()
        tc.append(t1 - t0)
        ts.append(t2 - t1)
        tq.append(t3 - t2)
    tq = np.array(tq)
    qp = tq[tq > 1e-3]                                      # steps in which the QP ran (every 10th)
    return dims, 1e3 * np.median(tc), 1e3 * np.median(ts), 1e3 * np.median(qp), len(qp)


def main():
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    res = [measure(sys.argv[1]) for _ in range(reps)]
    dims = res[0][0]
    c, s, q = (np.array([r[k] for r in res]) for k in (1, 2, 3))
    out = dict(dims=dims, repetitions=reps, steps_per_repetition=300,
               control_step_ms_median=float(np.median(c)), control_step_ms_range=[float(c.min()), float(c.max())],
               plant_ms_median=float(np.median(s)), qp_ms_median=float(np.median(q)),
               qp_ms_range=[float(q.min()), float(q.max())], qp_calls_per_repetition=int(res[0][4]),
               threads=os.environ.get("OPENBLAS_NUM_THREADS"))
    print(json.dumps(out, indent=1))
    with open(os.path.join(OUT, "timing.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
