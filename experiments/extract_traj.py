"""Trajectories of a mission run for the figures: girder (actual, compliant, desired), pelvises, soles (forward
kinematics of the logged configuration), planted feet, joint angles, connection wrenches and margins.

Usage: python experiments/extract_traj.py [tag]   (writes results/mission/traj_<tag>.npz)
"""
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mujoco

from teamimp import mission as MS, scene

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MI = os.path.join(ROOT, "results", "mission")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "al_n4_kz0"
    with open(os.path.join(MI, f"{tag}_logs.pkl"), "rb") as f:
        D = pickle.load(f)
    L = D["logs"]
    n = len(L[0]["pel"])
    M = MS.Mission(MS.Cfg(n=n, verbose=False, background=False))
    m = M.m
    d = mujoco.MjData(m)
    K = len(L)
    sole_sites = [[m.site(f"r{i}_{s}_sole").id for s in "lr"] for i in range(n)]
    out = dict(t=np.array([x["t"] for x in L]), phase=np.array([x["phase"] for x in L]),
               mode=np.array([x["mode"] for x in L]), beam=np.array([x["beam"] for x in L]),
               beam_c=np.array([x["beam_c"] if "beam_c" in x else np.full(3, np.nan) for x in L]),
               beam_d=np.array([x["beam_d"] if "beam_d" in x else np.full(3, np.nan) for x in L]),
               pel=np.array([x["pel"] for x in L]), pel_d=np.array([x["pel_d"] for x in L]))
    sole_p = np.zeros((K, n, 2, 3))
    sole_yaw = np.zeros((K, n, 2))
    planted = np.zeros((K, n, 2), bool)
    sole_fz = np.full((K, n, 2), np.nan)
    palm_fz = np.full((K, n, 2), np.nan)
    sole_cop = np.full((K, n, 2), np.nan)
    sole_cop_pred = np.full((K, n, 2), np.nan)
    palm_cop = np.full((K, n, 2), np.nan)
    palm_cop_pred = np.full((K, n, 2), np.nan)
    tm = M.tm
    jq = np.array([[tm.robots[i].qpos_adr[j] for j in scene.JOINTS] for i in range(n)])
    for k, x in enumerate(L):
        d.qpos[:] = x["qpos"]
        mujoco.mj_kinematics(m, d)
        for i in range(n):
            for js, sid in enumerate(sole_sites[i]):
                sole_p[k, i, js] = d.site_xpos[sid]
                R = d.site_xmat[sid].reshape(3, 3)
                sole_yaw[k, i, js] = np.arctan2(R[1, 0], R[0, 0])
        for c in x["conn"]:
            kind, i, s = c[0], c[1], c[2]
            js = 0 if s == "l" else 1
            if kind == "foot":
                planted[k, i, js] = True
                sole_fz[k, i, js] = c[4][2]
                sole_cop[k, i, js] = c[6][1]
                sole_cop_pred[k, i, js] = c[5][1]
            else:
                palm_fz[k, i, js] = c[4][2]
                palm_cop[k, i, js] = c[6][1]
                palm_cop_pred[k, i, js] = c[5][1]
    out.update(sole_p=sole_p, sole_yaw=sole_yaw, planted=planted, sole_fz=sole_fz, palm_fz=palm_fz,
               sole_cop=sole_cop, sole_cop_pred=sole_cop_pred, palm_cop=palm_cop, palm_cop_pred=palm_cop_pred,
               q=np.array([x["qpos"][jq] for x in L]), joints=np.array(scene.JOINTS),
               sat=np.array([x["sat"] for x in L]), mass=np.array([D["mass"]]),
               failed=np.array([D["failed"][0] if D["failed"] else -1.0]))
    np.savez_compressed(os.path.join(MI, f"traj_{tag}.npz"), **out)
    print("saved", os.path.join(MI, f"traj_{tag}.npz"))


if __name__ == "__main__":
    main()
