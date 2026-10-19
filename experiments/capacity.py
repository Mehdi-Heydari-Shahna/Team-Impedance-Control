"""Usage: python experiments/capacity.py [masses, e.g. 100,120,183] [teams, e.g. 4,3,2]

Static capacity analysis (Corollary 1 with actuator limits): smallest achievable peak actuator utilization over
the internal channel, for team sizes N = 2, 3, 4 and girder masses, in double support and in single support."""
import sys, os, json, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
import numpy as np, mujoco
from teamimp import mission as MS, scene, team, setup, refs as RF, controller, share

def utilization(M, q, single=None, use_internal=True):
    tm = M.tm
    mode = setup.grasp_mode(tm, q, feet=True, grasps=True)
    if single:
        mode = team.Mode({k: v for k, v in mode.feet.items() if k[1] != single}, mode.grasps)
    tm.set_mode(mode); tm.update(q, np.zeros(tm.m.nv))
    Ct = controller.TeamImpedance(tm, setup.default_params(tm), RF.HoldRefs(tm, q)); Ct.init_states(0.0)
    Ct.effort(0.0, np.zeros(6))
    u, lam_d, fF, flag = share.min_max_utilization(tm, Ct.last, M.R.frc_lim, use_internal=use_internal)
    return u, flag

out = {}
masses = [float(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else "100,120,140,160,183,200".split(','))]
teams = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [4, 3, 2]
for N in teams:
    for mass in masses:
        t0 = time.time()
        cfg = MS.Cfg(n=N, background=False, verbose=False, D_app=0.0, profile="HEA 280", mass=mass)
        M = MS.Mission(cfg)
        q = M.q_pick
        res = {}
        for lab, single, ui in (("double", None, True), ("single", "l", True), ("double_noint", None, False),
                                ("single_noint", "l", False)):
            u, flag = utilization(M, q, single, ui)
            res[lab] = u
        out[f"{N}_{mass}"] = res
        print(f"N={N} mass={mass:6.1f}  " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()) + f"   ({time.time()-t0:.1f}s)", flush=True)
path = os.path.join(ROOT, 'results', 'capacity.json')
res = json.load(open(path)) if os.path.exists(path) else {}
res.update(out)                                   # merge with earlier masses
json.dump(res, open(path, 'w'), indent=1)
