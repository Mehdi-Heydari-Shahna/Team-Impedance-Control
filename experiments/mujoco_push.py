"""External impedance with contact physics: from the post-lift state of the N = 4 mission (double support), a
horizontal force pushes the girder across its long axis (ramped in over 0.3 s, held, ramped out), twice: with
the push and without (reference run). The push is measured as part of the environment wrench h (Assumption 2:
e.g., a force sensor at the contact). Reports the girder and compliant-frame displacements caused by the push
and the static prediction of the external impedance (6), K_p^{-1} F.

Usage: python experiments/mujoco_push.py <state.pkl> [force_N] [nofriction]
(the state is produced by experiments/save_state.py; "nofriction" removes the joint friction of the plant, which
the controller model does not have)
"""
import os
import sys
import json
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from teamimp import mission as MS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "mission")


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Push(MS.Phase):
    """Double support, all references held; force F(t) * direction on the girder (at its center of mass)."""
    name = "push"

    def __init__(self, F, direction, T=5.0, t_on=0.5, t_off=3.0, ramp=0.3):
        self.F, self.dir, self.T, self.t_on, self.t_off, self.ramp = F, direction, T, t_on, t_off, ramp

    def force(self, tau):
        return self.F * (smooth((tau - self.t_on) / self.ramp) - smooth((tau - self.t_off) / self.ramp))

    def start(self, M, t):
        self.t0 = t

    def update(self, M, t):
        M.R.d.xfrc_applied[M.R.beam, :3] = self.force(t - self.t0) * self.dir

    def done(self, M, t):
        if t >= self.t0 + self.T:
            M.R.d.xfrc_applied[M.R.beam, :] = 0.0
            return True
        return False


def run(state_path, F, T=5.0, nofriction=False):
    M = pickle.load(open(state_path, "rb"))
    M.cfg.verbose = False
    if nofriction:
        M.R.m.dof_frictionloss[:] = 0.0
    R, d = M.R, M.R.d
    # long axis of the girder: body axis with the largest extent of the flange geoms
    Rb = d.xmat[R.beam].reshape(3, 3)
    g = M.R.m.geom("beam_bot").id
    ax_long = Rb[:, int(np.argmax(M.R.m.geom_size[g]))]
    direction = np.cross([0.0, 0.0, 1.0], ax_long)
    direction /= np.linalg.norm(direction)
    orig = R.contact_wrenches

    def contact_wrenches():                      # the push is part of the measured environment wrench h
        palm, sole, h = orig()
        f = d.xfrc_applied[R.beam, :3].copy()
        h = h.copy()
        h[:3] += f
        h[3:] += np.cross(d.xipos[R.beam] - d.xpos[R.beam], f)
        return palm, sole, h
    R.contact_wrenches = contact_wrenches
    ph = Push(F, direction, T=T)
    rec = []
    orig_log = M._log

    def _log():
        orig_log()
        L = M.logs[-1]
        rec.append(dict(t=L["t"], beam=L["beam"][:3].copy(), pc=L["beam_c"].copy(), hf=L["h_f"][:3].copy(),
                        sat=float(np.max(L["sat"])), mode=L["mode"],
                        cop=min(c[6][1] for c in L["conn"]), F=ph.force(L["t"] - ph.t0)))
    M._log = _log
    M.run([ph])
    return rec, direction, M


def main():
    state = sys.argv[1]
    F = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    nof = len(sys.argv) > 3 and sys.argv[3] == "nofriction"
    r0, dr, M0 = run(state, 0.0, nofriction=nof)
    r1, _, M1 = run(state, F, nofriction=nof)
    n = min(len(r0), len(r1))
    t = np.array([r1[k]["t"] for k in range(n)]) - r1[0]["t"]
    db = np.array([r1[k]["beam"] - r0[k]["beam"] for k in range(n)])          # girder displacement by the push
    dc = np.array([r1[k]["pc"] - r0[k]["pc"] for k in range(n)])              # compliant-frame displacement
    Ft = np.array([r1[k]["F"] for k in range(n)])
    hf = np.array([r1[k]["hf"] - r0[k]["hf"] for k in range(n)])
    Kp = float(np.diag(M1.C.P.Kp)[0]) if hasattr(M1.C, "P") else 6000.0
    along = db @ dr
    hold = (t > 2.2) & (t < 3.0)                                               # push held, transient decayed
    out = dict(F=F, direction=dr.tolist(), Kp=Kp, predicted_mm=1e3 * F / Kp,
               girder_mm_hold=float(1e3 * np.mean(along[hold])), compliant_mm_hold=float(1e3 * np.mean(dc[hold] @ dr)),
               girder_mm_max=float(1e3 * along.max()), girder_mm_end=float(1e3 * along[-1]),
               girder_off_axis_mm_max=float(1e3 * np.max(np.linalg.norm(db - np.outer(along, dr), axis=1))),
               h_measured_hold=float(np.mean(hf[hold] @ dr)),
               sat_max_push=float(max(r["sat"] for r in r1[:n])), sat_max_ref=float(max(r["sat"] for r in r0[:n])),
               cop_min_push_mm=float(1e3 * min(r["cop"] for r in r1[:n])),
               cop_min_ref_mm=float(1e3 * min(r["cop"] for r in r0[:n])),
               modes_equal=bool(all(r0[k]["mode"] == r1[k]["mode"] for k in range(n))))
    print(json.dumps(out, indent=1), flush=True)
    suf = "_nofriction" if nof else ""
    np.savez(os.path.join(OUT, f"mujoco_push_{int(F)}{suf}.npz"), t=t, db=db, dc=dc, F=Ft, hf=hf, dir=dr)
    json.dump(out, open(os.path.join(OUT, f"mujoco_push_{int(F)}{suf}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
