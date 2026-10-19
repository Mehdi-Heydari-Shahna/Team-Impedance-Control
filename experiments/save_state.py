"""Run the mission to the end of the lift and pickle the Mission object; this is the initial state of
experiments/mujoco_decoupling.py (identical to the state of the full mission at the end of the lift).

Usage: python experiments/save_state.py [--n 4] [--kz 0] [--profile "AL 280"] [--out <file>]
(default file: results/mission/state_lift_al_n4_kz0.pkl for the aluminium girder, state_lift_n4_kz0.pkl for steel)
"""
import os
import sys
import time
import pickle
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from teamimp import mission as MS

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=4)
p.add_argument("--kz", type=float, default=0.0)
p.add_argument("--profile", type=str, default="AL 280")
p.add_argument("--out", type=str, default=None)
a = p.parse_args()
pre = "al_" if a.profile.startswith("AL") else ""
out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mission",
                            f"state_lift_{pre}n{a.n}_kz{a.kz:g}.pkl")
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
M = MS.Mission(MS.Cfg(n=a.n, kz=a.kz, verbose=True, background=True, profile=a.profile))
ph = MS.full_mission(M, stop_after="lift")[:-1]          # up to the end of the lift (final hold dropped)
t0 = time.time()
M.run(ph)
print(f"done: sim {M.R.t:.3f} s, wall {time.time() - t0:.0f} s, failed={M.failed}", flush=True)
with open(out, "wb") as f:
    pickle.dump(M, f)
print("saved", out, flush=True)
