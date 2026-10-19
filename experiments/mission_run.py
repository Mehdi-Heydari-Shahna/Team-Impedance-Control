"""Construction-site mission in MuJoCo (contact physics): go, pick, carry, unload.

Usage: python experiments/mission_run.py --n 4 --tag al_n4 [--profile "AL 280"] [--mass 63] [--no-share-internal]
       [--internal impedance|feedforward|off] [--ctrl-mass 150] [--carry 4] [--app 2]
"""
import os
import sys
import argparse
import pickle
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from teamimp import mission as MS

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=4)
p.add_argument("--tag", type=str, default=None)
p.add_argument("--profile", type=str, default="AL 280", help='"AL 280" (63 kg aluminium) or "HEA 280" (183 kg steel)')
p.add_argument("--mass", type=float, default=None)
p.add_argument("--ctrl-mass", type=float, default=None)
p.add_argument("--internal", type=str, default="impedance")
p.add_argument("--no-share-internal", action="store_true")
p.add_argument("--kz", type=float, default=1.0)
p.add_argument("--carry", type=int, default=4)
p.add_argument("--app", type=int, default=2)
p.add_argument("--retreat", type=int, default=1)
p.add_argument("--no-background", action="store_true")
p.add_argument("--t-max", type=float, default=200.0)
p.add_argument("--stop-after", type=str, default=None)
p.add_argument("--noise-f", type=float, default=0.0, help="force-sensor noise, standard deviation (N)")
p.add_argument("--noise-m", type=float, default=0.0, help="moment-sensor noise, standard deviation (N m)")
p.add_argument("--noise-seed", type=int, default=0)
p.add_argument("--kin-err", type=float, default=0.0, help="kinematic model error: std of link offsets (m)")
p.add_argument("--kin-err-seed", type=int, default=0)
p.add_argument("--palm-weight", type=float, default=None, help="QP weight of the palm margin slacks (default 10)")
p.add_argument("--fmin-palm", type=float, default=None, help="QP minimum palm normal force (N, default 25)")
p.add_argument("--shift-frac-load", type=float, default=None,
               help="pelvis shift toward the stance foot when carrying (fraction of the lateral distance)")
a = p.parse_args()
tag = a.tag or f"n{a.n}"
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mission")
os.makedirs(out_dir, exist_ok=True)

cfg = MS.Cfg(n=a.n, profile=a.profile, mass=a.mass, ctrl_mass=a.ctrl_mass, internal_mode=a.internal,
             share_internal=not a.no_share_internal, kz=a.kz, background=not a.no_background,
             D_app=a.app * 0.25, YB=a.carry * 0.25, t_max=a.t_max,
             sensor_noise=(a.noise_f, a.noise_m) if (a.noise_f > 0 or a.noise_m > 0) else None,
             noise_seed=a.noise_seed, kin_err=a.kin_err, kin_err_seed=a.kin_err_seed,
             share_kw={k: v for k, v in (("w_soft_palm", a.palm_weight), ("fmin_palm", a.fmin_palm)) if v is not None},
             **({} if a.shift_frac_load is None else dict(shift_frac_load=a.shift_frac_load)))
t0 = time.time()
M = MS.Mission(cfg)
ph = MS.full_mission(M, n_app=a.app, n_carry=a.carry, retreat=a.retreat, stop_after=a.stop_after)
M.run(ph)
print(f"[{tag}] done: sim {M.R.t:.2f}s wall {time.time() - t0:.0f}s failed={M.failed}", flush=True)

# compact arrays for figures/video, full logs for analysis
L = M.logs
t = np.array([x["t"] for x in L])
qpos = np.array([x["qpos"] for x in L])
sat = np.array([x["sat"] for x in L])
sM = np.array([x["sM"] for x in L])
beam = np.array([x["beam"] for x in L])
phase = np.array([x["phase"] for x in L])
mode = np.array([x["mode"] for x in L])
np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), t=t, qpos=qpos, sat=sat, sM=sM, beam=beam, phase=phase,
                    mode=mode, failed=np.array([M.failed[0] if M.failed else -1.0]),
                    fail_reason=np.array([M.failed[1] if M.failed else ""]), mass=np.array([M.mass]),
                    n=np.array([a.n]))
with open(os.path.join(out_dir, f"{tag}_logs.pkl"), "wb") as f:
    pickle.dump(dict(logs=L, events=M.events, failed=M.failed, cfg=vars(cfg), mass=M.mass, W=M.W,
                     wall=M.t_wall), f)
print(f"[{tag}] saved", flush=True)
