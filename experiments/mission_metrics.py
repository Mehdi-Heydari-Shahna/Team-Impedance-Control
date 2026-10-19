"""Metrics of the MuJoCo mission runs (Table III and the text of Section V)."""
import os
import sys
import json
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MI = os.path.join(ROOT, "results", "mission")
LIFTED = ("lift", "carry", "place")       # phases in which the team carries the girder alone (h = 0 after lift-off)


def metrics(tag):
    with open(os.path.join(MI, f"{tag}_logs.pkl"), "rb") as f:
        d = pickle.load(f)
    L = d["logs"]
    failed = d["failed"]
    tf = failed[0] if failed else np.inf
    t = np.array([x["t"] for x in L])
    ok = t < tf - 1e-9
    ph = [x["phase"] for x in L]
    sat = np.array([np.max(x["sat"]) for x in L])
    S = np.array([x["sat"] for x in L])
    carried = np.array([any(p.startswith(k) for k in LIFTED) for p in ph]) & ok
    # exclude the first 0.6 s of the lift (lift-off from the trestles: h != 0)
    t_lift = min([x["t"] for x in L if x["phase"] == "lift"] or [np.inf])
    carried &= t > t_lift + 0.6
    carry = np.array([p.startswith("carry") for p in ph]) & ok
    single = np.array([x["mode"][0] < 2 * len(x["pel"]) for x in L])
    perr = np.array([np.linalg.norm(x["beam"][:3] - x["beam_c"]) if "beam_c" in x else np.nan for x in L])
    # with sensor noise, the noise-free values are evaluated (eI_true, palm_true); eI_meas is what the controller saw
    eI = np.array([np.linalg.norm(x["eI_true"] if "eI_true" in x else x["eI"]) if "eI" in x else np.nan for x in L])
    eI_meas = np.array([np.linalg.norm(x["eI"]) if "eI" in x else np.nan for x in L])
    palm_min = np.array([min([(x["palm_true"][(c[1], c[2])][2] if "palm_true" in x else c[4][2])
                              for c in x["conn"] if c[0] == "grasp"] or [np.nan]) for x in L])
    shift = []
    for x, sg, ca in zip(L, single, carry):
        if sg and ca:
            g = {(c[1], c[2]): c[4][2] for c in x["conn"] if c[0] == "grasp"}
            shift += [abs(g[(i, "l")] - g[(i, "r")]) for i in range(len(x["pel"])) if (i, "l") in g and (i, "r") in g]
    sat_idx = np.where((S[ok] >= 0.999).any(0))[0]
    out = dict(tag=tag, n=int(len(L[0]["pel"])), completed=failed is None,
               fail_time=None if failed is None else float(failed[0]),
               fail_phase=None if failed is None else ph[int(np.searchsorted(t, failed[0]))],
               fail_reason=None if failed is None else failed[1], t_end=float(t[-1]),
               util_max=float(sat[ok].max()), util_p99=float(np.quantile(sat[ok], 0.99)),
               util_p99_carried=float(np.quantile(sat[carried], 0.99)) if carried.any() else None,
               sat_time_pct=float(100 * np.mean(sat[ok] >= 0.999)),
               sat_time_pct_carried=float(100 * np.mean(sat[carried] >= 0.999)) if carried.any() else None,
               saturated_actuators=[int(j) for j in sat_idx],
               perr_max_mm=float(1e3 * np.nanmax(perr[carried])) if carried.any() else None,
               perr_rms_mm=float(1e3 * np.sqrt(np.nanmean(perr[carried] ** 2))) if carried.any() else None,
               eI_rms=float(np.sqrt(np.nanmean(eI[carried] ** 2))) if carried.any() else None,
               eI_med=float(np.nanmedian(eI[carried])) if carried.any() else None,
               eI_p95=float(np.nanquantile(eI[carried], 0.95)) if carried.any() else None,
               eI_meas_rms=float(np.sqrt(np.nanmean(eI_meas[carried] ** 2))) if carried.any() else None,
               palm_min_carried=float(np.nanmin(palm_min[carried])) if carried.any() else None,
               hand_shift_single_max=float(np.max(shift)) if shift else None,
               hand_shift_single_mean=float(np.mean(shift)) if shift else None,
               mass=float(d["mass"]), ctrl_mass=d["cfg"].get("ctrl_mass"), kz=d["cfg"].get("kz"),
               internal=d["cfg"].get("internal_mode"), share=d["cfg"].get("share_internal"),
               wall=float(d.get("wall", np.nan)), sim_time=float(t[-1]))
    return out


if __name__ == "__main__":
    AL = ["al_n4_kz0", "al_n4", "al_n4_ff", "al_n4_noshare_kz0", "al_n3_kz0", "al_n2_kz0", "al_n4_noise5",
          "al_n4_noise15", "al_n4_kin_kz0", "al_n4_kin", "al_n4_kin_ff", "al_n4_m50_kz0", "al_n4_m50",
          "al_n4_m70_kz0", "al_n4_m66_kz0", "al_n4_m60_kz0", "al_n4_m55_kz0", "al_n4_m40_kz0", "al_n4_m30_kz0",
          "al_n4_m70", "al_n4_m66", "al_n4_m60", "al_n4_m55", "al_n4_m40", "al_n4_m70_ff"]
    SWEEP = ["n4_kz0", "n3_kz0", "n2_kz0", "sweep_n4_240", "sweep_n4_260", "sweep_n4_280", "sweep_n3_120",
             "sweep_n3_140", "sweep_n3_160", "sweep_n2_100", "sweep_n2_120", "sweep_n2_140"]
    tags = sys.argv[1:] or AL + SWEEP
    path = os.path.join(MI, "metrics.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for tg in tags:
        if os.path.exists(os.path.join(MI, f"{tg}_logs.pkl")):
            res[tg] = metrics(tg)
            r = res[tg]
            print(f"{tg:12s} done={r['completed']!s:5s} fail={r['fail_time']} {r['fail_phase']} util_max={r['util_max']:.3f} "
                  f"p99={r['util_p99']:.3f} p99c={r['util_p99_carried']} sat%={r['sat_time_pct']:.2f} "
                  f"perr={r['perr_max_mm']} rms={r['perr_rms_mm']} eI_rms={r['eI_rms']} med={r['eI_med']} "
                  f"palm_min={r['palm_min_carried']} shift={r['hand_shift_single_max']}", flush=True)
    json.dump(res, open(path, "w"), indent=1)
