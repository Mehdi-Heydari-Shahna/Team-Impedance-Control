"""Figures of the letter (IEEE two-column format, vector PDF).

  python experiments/figures.py exact      # exact compiled model: decoupling, sampling, rates, antagonism
  python experiments/figures.py mission    # mission overview (needs paper/figures/snap_*.png)
  python experiments/figures.py traj       # trajectories (needs results/mission/traj_n4_kz0.npz)
  python experiments/figures.py capacity   # capacity estimate and actuator utilization
  python experiments/figures.py contact    # push, internal command vs reference displacement, model-mass bias
  python experiments/figures.py all
"""
import os
import sys
import json
import pickle
import glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, LogLocator, NullFormatter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EX = os.path.join(ROOT, "results", "exact")
MI = os.path.join(ROOT, "results", "mission")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
    "legend.fontsize": 6.3, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "lines.linewidth": 0.9, "legend.frameon": False,
    "legend.handlelength": 1.6, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.25,
    "axes.titlepad": 2.0, "axes.labelpad": 1.5, "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
})
# categorical slots (fixed order; validated for color-vision deficiency up to three series on all pairs) and
# neutrals; a fourth series is drawn in gray with a dash pattern
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, GRAY, LGRAY = "#1a1a1a", "#6e6e6e", "#c4c4c4"
PHASE_COL = dict(pick=AQUA, carry=BLUE, unload=ORANGE)


def label_box(ax, s, x=0.015, y=0.97, ha="left", va="top"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))


# ------------------------------------------------------------------------------------------------ exact model
def fig_exact(path=os.path.join(FIG, "fig_exact.pdf")):
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.38))
    plt.subplots_adjust(left=0.065, right=0.995, bottom=0.255, top=0.88, wspace=0.42)
    # (a) decoupling: motion difference with and without internal command
    ax = axs[0]
    d = np.load(os.path.join(EX, "decoupling.npz"))
    t = d["t"]
    zs = sorted(glob.glob(os.path.join(EX, "zoh_*.npz")), key=lambda p: float(np.load(p)["Ts"]))
    for p in zs:
        z = np.load(p)
        Ts = float(z["Ts"]) * 1e3
        if Ts in (1.0, 4.0):
            ax.semilogy(z["t"], np.maximum(z["dq"], 1e-17), color=GRAY if Ts == 1.0 else ORANGE, lw=0.8,
                        ls="-" if Ts == 4.0 else (0, (3, 1)), label=rf"ZOH, $T={Ts:g}$ ms")
    ax.semilogy(t, np.maximum(d["dq_exact"], 1e-17), color=BLUE, lw=0.7, label="exact model")
    ax.semilogy(t, np.maximum(d["dq_inexact"], 1e-17), color=AQUA, lw=0.7, ls="--", label="inexact model")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\max_k|\Delta q_k|$ (m, rad)")
    ax.set_ylim(1e-16, 1e-2)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0, 2.6)
    ax.legend(loc="center right", fontsize=6.0, handlelength=1.6, bbox_to_anchor=(1.0, 0.57))
    ax.set_title(r"(a) $\psi$ does not move the team", loc="left", fontsize=7)
    # (b) sampled-data deviation versus period
    ax = axs[1]
    Ts = np.array([float(np.load(p)["Ts"]) for p in zs]) * 1e3
    dq = np.array([np.load(p)["dq"].max() for p in zs])
    dp = np.array([np.load(p)["dp"].max() for p in zs])
    ax.loglog(Ts, dq, "o-", color=ORANGE, ms=2.8, label=r"joints (rad)")
    ax.loglog(Ts, dp, "s-", color=BLUE, ms=2.6, label="payload (m)")
    ax.loglog(Ts, dq[0] * Ts / Ts[0], ":", color=GRAY, lw=0.8, label=r"$\propto T$")
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "4"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.42, 4.8)
    ax.set_ylim(1e-6, 2e-4)
    ax.set_xlabel(r"hold period $T$ (ms)")
    ax.set_ylabel("max. deviation")
    ax.legend(loc="upper left", fontsize=6.0)
    ax.set_title("(b) sampled data: $O(T)$", loc="left", fontsize=7)
    # (c) Theorem 2: measured decay versus certified rate
    ax = axs[2]
    r = np.load(os.path.join(EX, "rates.npz"))
    t = r["t"]
    ks = float(r["kappa_s"])
    ax.semilogy(t, r["sM"] / r["sM"][0], color=BLUE, label=r"$\|s\|_M$")
    ax.semilogy(t, np.exp(-ks * t), ":", color=GRAY, lw=0.9, label=rf"$e^{{-\kappa_s t}}$")
    ax.semilogy(t, r["xi"] / r["xi"][0], color=AQUA, label=r"$\|\xi\|$")
    ax.semilogy(t, r["eI"] / r["eI"][0], color=ORANGE, label=r"$\|e_I\|$")
    ax.semilogy(r["tl"], r["chi"] / r["chi"][0], color=INK, ls="-.", lw=0.7, label=r"$\|\chi\|$")
    ax.set_ylim(1e-4, 60)
    ax.set_xlim(0, t[-1])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("time (s)")
    ax.set_ylabel("normalized error")
    ax.legend(loc="upper right", ncol=3, fontsize=6.0, columnspacing=0.6, handlelength=1.3, handletextpad=0.4,
              borderaxespad=0.2)
    ax.set_title("(c) Theorem 2 rates", loc="left", fontsize=7)
    # (d) Proposition 1(iv): antagonism rejection
    ax = axs[3]
    a = np.load(os.path.join(EX, "antagonism.npz"))
    ainf = float(a["a_inf"][0])
    ax.plot(a["t_ff"], a["eI_ff"], color=ORANGE, label=r"$\psi\equiv\lambda_d$")
    ax.plot(a["t_kz1"], a["eI_kz1"], color=AQUA, label=r"$K_z=I$")
    ax.plot(a["t_int"], a["eI_int"], color=BLUE, label=r"$K_z=0$")
    for lev, c in ((ainf, ORANGE), (0.5 * ainf, AQUA), (0.0, BLUE)):
        ax.axhline(lev, color=c, ls=":", lw=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\|e_I\|$ (N)")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-5, 118)
    ax.legend(loc="center right", fontsize=6.0, bbox_to_anchor=(1.0, 0.62), borderaxespad=0.2)
    ax.set_title("(d) antagonism, Prop. 1(iv)", loc="left", fontsize=7)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


# ------------------------------------------------------------------------------------------------ mission
PHASE_GROUPS = [("go", ("settle", "go")), ("pick", ("pre-reach", "reach-in", "contact", "load-transfer", "lift")),
                ("carry", ("carry",)), ("unload", ("place", "unload", "release", "stow")),
                ("back", ("back", "done"))]


def phase_group(ph):
    for g, keys in PHASE_GROUPS:
        if any(ph.startswith(k) for k in keys):
            return g
    return "other"


def load_mission(tag):
    with open(os.path.join(MI, f"{tag}_logs.pkl"), "rb") as f:
        d = pickle.load(f)
    L = d["logs"]
    t = np.array([x["t"] for x in L])
    out = dict(t=t, failed=d["failed"], W=d["W"], mass=d["mass"])
    out["sat"] = np.array([np.max(x["sat"]) for x in L])
    out["phase"] = [x["phase"] for x in L]
    out["group"] = [phase_group(p) for p in out["phase"]]
    pal = np.full((len(L), 2 * 4), np.nan)
    for k, x in enumerate(L):
        g = [(c[1], c[2], c[4][2]) for c in x["conn"] if c[0] == "grasp"]
        for (i, s, fz) in g:
            pal[k, 2 * i + (0 if s == "l" else 1)] = fz
    out["palm"] = pal
    out["hz"] = np.array([x["h_f"][2] if "h_f" in x else np.nan for x in L])
    out["perr"] = np.array([np.linalg.norm(x["beam"][:3] - x["beam_c"]) if "beam_c" in x else np.nan for x in L])
    out["eI"] = np.array([np.linalg.norm(x["eI_true"] if "eI_true" in x else x["eI"]) if "eI" in x else np.nan
                          for x in L])
    out["mode"] = np.array([x["mode"] for x in L])
    return out


def shade_phases(ax, t, groups, alpha=0.08):
    k0 = 0
    for k in range(1, len(t) + 1):
        if k == len(t) or groups[k] != groups[k0]:
            g = groups[k0]
            if g in PHASE_COL:
                ax.axvspan(t[k0], t[min(k, len(t) - 1)], color=PHASE_COL[g], alpha=alpha, lw=0)
            k0 = k


def nice_top(v, steps=(1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)):
    """Smallest 'nice' number >= v."""
    e = 10 ** np.floor(np.log10(max(v, 1e-9)))
    return next(s * e for s in steps if s * e >= v - 1e-12)


def fig_mission(path=os.path.join(FIG, "fig_mission.pdf"), runs=None, snaps=None, labels=None, main="al_n4_kz0",
                fails=None, eI_runs=None):
    import matplotlib.image as mpimg
    runs = runs or {}
    snaps = snaps or [os.path.join(FIG, f"snap_{k}.png") for k in range(6)]
    labels = labels or ["(i) go", "(ii) pick", "(iii) lift", "(iv) carry", "(v) place", "(vi) step back"]
    fails = fails or [("al_n4_noshare_kz0", GRAY, (0, (3, 1.2)), r"$\lambda_d=0$"),
                      ("al_n2_kz0", AQUA, "-", "$N=2$"), ("al_n3_kz0", ORANGE, "-", "$N=3$")]
    eI_runs = eI_runs or []
    W_in, H_in = 7.16, 1.52
    fig = plt.figure(figsize=(W_in, H_in))
    ns = len(snaps)
    gap = 0.004
    w = (1.0 - gap * (ns - 1)) / ns
    img0 = mpimg.imread(snaps[0]) if os.path.exists(snaps[0]) else np.zeros((450, 640, 3))
    h = w * W_in * img0.shape[0] / img0.shape[1] / H_in
    for k, p in enumerate(snaps):
        ax = fig.add_axes([k * (w + gap), 1.0 - h, w, h])
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        ax.set_axis_off()
        ax.text(0.03, 0.05, labels[k], transform=ax.transAxes, fontsize=6.4, color="white", fontweight="bold",
                va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.15", fc=(0, 0, 0, 0.45), ec="none"))
    y0 = 0.18
    hb = 1.0 - h - y0 - 0.055
    xs = [0.062, 0.392, 0.724]
    wb = 0.266
    axs = [fig.add_axes([x, y0, wb, hb]) for x in xs]
    r4 = runs.get(main)
    tmin = 20.0
    # (a) loads of the N = 4 run
    ax = axs[0]
    if r4 is not None:
        t = r4["t"]
        shade_phases(ax, t, r4["group"])
        ax.plot(t, r4["palm"][:, 0], color=BLUE, lw=0.6, label="left palm")
        ax.plot(t, r4["palm"][:, 1], color=ORANGE, lw=0.6, label="right palm")
        ax.plot(t, r4["hz"] / 8.0, color=INK, lw=0.7, ls="--", label=r"trestles $h_z/8$")
    ax.set_xlim(tmin, 82)
    if r4 is not None:
        sel = (r4["t"] >= tmin) & (r4["t"] <= 82)
        top = nice_top(1.4 * np.nanmax(np.r_[r4["palm"][sel, :2].ravel(), r4["hz"][sel] / 8.0]))
        ax.set_ylim(0, top)
        ax.set_yticks([0, top / 2, top])
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel("force (N)")
    ax.legend(loc="upper center", fontsize=6.0, ncol=3, columnspacing=0.6, handlelength=1.2,
              bbox_to_anchor=(0.5, 1.04))
    ax.text(0.985, 0.06, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, fontweight="bold")
    # (b) peak actuator utilization for team sizes and the no-sharing ablation
    ax = axs[1]
    for tag, col, ls, lab in fails + [(main, BLUE, "-", "$N=4$")]:
        r = runs.get(tag)
        if r is None:
            continue
        t = r["t"]
        sel = t >= tmin
        if r["failed"]:
            sel &= t <= r["failed"][0]
        ax.plot(t[sel], r["sat"][sel], color=col, ls=ls, lw=0.65 if tag == main else 0.6, label=lab)
        if r["failed"]:
            ax.plot([r["failed"][0]], [1.07], marker="x", color=col, ms=3.5, mew=0.9, clip_on=False)
    if r4 is not None:
        shade_phases(ax, r4["t"], r4["group"])
    ax.set_xlim(tmin, 82)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel(r"peak $|f_i|/f_i^{\max}$")
    h_, l_ = ax.get_legend_handles_labels()
    order = list(range(len(h_)))[::-1]
    ax.legend([h_[i] for i in order], [l_[i] for i in order], loc="lower center", fontsize=6.0, ncol=4,
              columnspacing=0.6, handlelength=1.1, handletextpad=0.4, bbox_to_anchor=(0.5, -0.02),
              frameon=False)
    ax.text(0.985, 0.95, "(b)", transform=ax.transAxes, ha="right", va="top", fontsize=7, fontweight="bold")
    # (c) internal-wrench error of internal-channel variants
    ax = axs[2]
    med = {}
    for tag, col, lab, ls in eI_runs:
        r = runs.get(tag)
        if r is None:
            continue
        k = max(1, int(round(0.5 / np.median(np.diff(r["t"])))))       # 0.5 s moving median
        e = np.pad(r["eI"], (k // 2, k - 1 - k // 2), mode="edge")
        med[tag] = np.array([np.nanmedian(e[i:i + k]) for i in range(len(r["eI"]))])
        ax.plot(r["t"], med[tag], color=col, lw=0.7, ls=ls, label=lab)
    if r4 is not None:
        shade_phases(ax, r4["t"], r4["group"])
    ax.set_xlim(tmin, 82)
    vals = np.concatenate([med[tg][(runs[tg]["t"] >= tmin)] for tg in med] or [[1.0]])
    vals = vals[np.isfinite(vals)]
    top = nice_top(1.3 * np.max(vals))
    ax.set_ylim(0, top)
    print(f"  Fig. 4(c): y-range 0-{top:g} N, largest moving median {vals.max():.0f} N", flush=True)
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel(r"$\|e_I\|$ (N)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", fontsize=6.0, ncol=3, columnspacing=0.7, handlelength=1.2, frameon=True,
                  facecolor="white", edgecolor="none", framealpha=0.92, borderpad=0.25, bbox_to_anchor=(0.56, 1.02))
    ax.text(0.015, 0.95, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=7, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.9))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)


MISSION_TAGS = ["al_n4_kz0", "al_n3_kz0", "al_n2_kz0", "al_n4_noshare_kz0", "al_n4_kin_kz0", "al_n4_kin",
                "al_n4_kin_ff"]


def fig_mission_paper():
    runs = {tag: load_mission(tag) for tag in MISSION_TAGS if os.path.exists(os.path.join(MI, f"{tag}_logs.pkl"))}
    # internal-wrench error under kinematic model errors (constant antagonism, Proposition 1(iv))
    eI = [("al_n4_kin_ff", ORANGE, r"$\psi\equiv\lambda_d$", "-"), ("al_n4_kin", AQUA, r"$K_z=I$", "-"),
          ("al_n4_kin_kz0", BLUE, r"$K_z=0$", "-")]
    fig_mission(runs=runs, eI_runs=eI)


# ------------------------------------------------------------------------------------------------ trajectories
def runs_of(mask):
    """(start, end) index pairs of the True runs of a boolean array."""
    out, k = [], 0
    n = len(mask)
    while k < n:
        if mask[k]:
            j = k
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((k, j))
            k = j + 1
        else:
            k += 1
    return out


def fig_traj(path=os.path.join(FIG, "fig_traj.pdf"), tag="al_n4_kz0", cycle=(41.6, 51.8), robot=0):
    z = np.load(os.path.join(MI, f"traj_{tag}.npz"), allow_pickle=True)
    t, beam, bc, bd = z["t"], z["beam"], z["beam_c"], z["beam_d"]
    pel, sp, yaw, planted = z["pel"], z["sole_p"], z["sole_yaw"], z["planted"]
    groups = [phase_group(p) for p in z["phase"]]
    n = pel.shape[1]
    fig = plt.figure(figsize=(7.16, 2.02))
    # (a) top view
    ax = fig.add_axes([0.058, 0.13, 0.29, 0.845])
    hb, Lb = 0.28, 2.4
    for y0, lab in ((0.0, "start"), (1.0, "end")):
        ax.add_patch(Rectangle((-Lb / 2, y0 - hb / 2), Lb, hb, fc="none", ec=LGRAY, lw=0.6, ls="--", zorder=1))
    for x in (-0.95, 0.95):
        for y in (0.0, 1.0):
            ax.add_patch(Rectangle((x - 0.09, y - 0.32), 0.18, 0.64, fc="none", ec="#8a6d4b", lw=0.35, ls=":",
                                   zorder=0))
            ax.add_patch(Rectangle((x - 0.07, y - 0.25), 0.14, 0.50, fc="#d9c3a5", ec="#8a6d4b", lw=0.4, zorder=0))
    ax.text(-0.95, -0.36, "trestles", ha="center", va="top", fontsize=6.0, color=INK)
    for i in range(n):
        for js in range(2):
            for (a, b) in runs_of(planted[:, i, js]):
                if b - a < 5:
                    continue
                k = (a + b) // 2
                p, th = sp[k, i, js], yaw[k, i, js]
                L, W = 0.20, 0.085
                c, s = np.cos(th), np.sin(th)
                corner = p[:2] - c * np.array([L / 2, 0]) - s * np.array([0, L / 2]) \
                    - np.array([-s * W / 2, c * W / 2])
                ax.add_patch(Rectangle(tuple(corner), L, W, angle=np.degrees(th), fc="#e9e9e9", ec=GRAY, lw=0.35,
                                       zorder=2))
        ax.plot(pel[:, i, 0], pel[:, i, 1], color=INK, lw=0.6, zorder=3)
        ax.plot(pel[0, i, 0], pel[0, i, 1], "o", color=INK, ms=2.6, zorder=4)
        ax.plot(pel[-1, i, 0], pel[-1, i, 1], "s", color=INK, ms=2.4, mfc="white", zorder=4)
        ax.text(pel[0, i, 0] + (0.36 if pel[0, i, 0] > 0 else -0.36), pel[0, i, 1], f"robot {i + 1}",
                fontsize=6.0, ha="center", va="center")
    ok = ~np.isnan(bc[:, 0])
    ax.plot(beam[:, 0], beam[:, 1], color=BLUE, lw=1.2, zorder=5, label="girder center")
    ax.plot([], [], color=INK, lw=0.6, label="pelvises")
    ax.add_patch(Rectangle((0, 0), 0, 0, fc="#e9e9e9", ec=GRAY, lw=0.35, label="footholds"))
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.05, 1.95)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.legend(loc="upper center", fontsize=6.0, ncol=3, columnspacing=0.8, handlelength=1.3,
              bbox_to_anchor=(0.5, 1.01))
    ax.text(0.02, 0.02, "(a)", transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom")
    # right block
    x1, x2, wr = 0.425, 0.737, 0.258
    yt, yb, hr = 0.64, 0.13, 0.31
    # (b) girder height
    ax = fig.add_axes([x1, yt, wr, hr])
    shade_phases(ax, t, groups)
    ax.plot(t, bd[:, 2], color=INK, ls="--", lw=0.7, label=r"desired $p_d$")
    ax.plot(t, bc[:, 2], color=ORANGE, lw=0.8, label=r"compliant $p_c$")
    ax.plot(t, beam[:, 2], color=BLUE, lw=0.7, label=r"girder $p$")
    ax.set_xlim(20, 82)
    ax.set_ylim(0.74, 1.24)
    ax.set_ylabel("height $z$ (m)")
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.legend(loc="lower center", fontsize=6.0, ncol=1, handlelength=1.4, bbox_to_anchor=(0.55, 0.02))
    label_box(ax, "(b)", x=0.985, ha="right")
    # (c) pose error components
    ax = fig.add_axes([x2, yt, wr, hr])
    shade_phases(ax, t, groups)
    e = 1e3 * (bc - beam[:, :3])
    for j, (col, lab) in enumerate(((BLUE, "$x$"), (ORANGE, "$y$"), (AQUA, "$z$"))):
        ax.plot(t, e[:, j], color=col, lw=0.6, label=lab)
    ax.axhline(0, color=GRAY, lw=0.4)
    ax.set_xlim(20, 82)
    win = (t >= 20) & (t <= 82)
    lim = nice_top(1.25 * np.nanmax(np.abs(e[win])))
    ax.set_ylim(-lim, lim)
    ax.set_ylabel(r"$p_c-p$ (mm)")
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.legend(loc="lower center", fontsize=6.0, ncol=3, columnspacing=0.8, handlelength=1.3)
    label_box(ax, "(c)", x=0.985, ha="right")
    # gait cycle window
    sel = (t >= cycle[0]) & (t <= cycle[1])
    ss = planted[:, robot, :].sum(axis=1) == 1
    # (d) leg joint angles of one robot
    ax = fig.add_axes([x1, yb, wr, hr])
    J = list(z["joints"])
    q = z["q"][:, robot, :]
    for (a, b) in runs_of(ss & sel):
        ax.axvspan(t[a], t[b], color=LGRAY, alpha=0.35, lw=0)
    for jn, col, ls, lab in (("l_hip_fe", BLUE, "-", "hip L"), ("l_knee_fe", ORANGE, "-", "knee L"),
                             ("l_ankle_pd", AQUA, "-", "ankle L"), ("r_knee_fe", ORANGE, "--", "knee R")):
        ax.plot(t[sel], q[sel, J.index(jn)], color=col, ls=ls, lw=0.75, label=lab)
    ax.set_xlim(*cycle)
    ax.set_ylim(-1.0, 2.1)
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel("joint angle (rad)")
    ax.legend(loc="upper left", fontsize=6.0, ncol=4, columnspacing=0.7, handlelength=1.3,
              bbox_to_anchor=(0.0, 1.0))
    label_box(ax, "(d)", x=0.985, y=0.04, ha="right", va="bottom")
    # (e) load shares of one robot: feet and palms
    ax = fig.add_axes([x2, yb, wr, hr])
    for (a, b) in runs_of(ss & sel):
        ax.axvspan(t[a], t[b], color=LGRAY, alpha=0.35, lw=0)
    sf = np.nan_to_num(z["sole_fz"][:, robot, :])
    pf = np.nan_to_num(z["palm_fz"][:, robot, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        fs = 100 * sf / sf.sum(axis=1, keepdims=True)
        fp = 100 * pf / pf.sum(axis=1, keepdims=True)
    ax.plot(t[sel], fs[sel, 0], color=BLUE, lw=0.75, label="sole L")
    ax.plot(t[sel], fs[sel, 1], color=BLUE, lw=0.75, ls="--", label="sole R")
    ax.plot(t[sel], fp[sel, 0], color=ORANGE, lw=0.75, label="palm L")
    ax.plot(t[sel], fp[sel, 1], color=ORANGE, lw=0.75, ls="--", label="palm R")
    ax.set_xlim(*cycle)
    ax.set_ylim(-3, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel("load share (%)")
    ax.legend(loc="upper left", fontsize=6.0, ncol=4, columnspacing=0.6, handlelength=1.3,
              bbox_to_anchor=(0.0, 1.0))
    label_box(ax, "(e)", x=0.985, y=0.04, ha="right", va="bottom")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("saved", path)


# ------------------------------------------------------------------------------------------------ capacity
JOINT_GROUPS = [("torso", ("torso_yaw", "torso_roll", "torso_pitch")),
                ("shoulder", ("shoulder_aa", "shoulder_ie", "shoulder_fe")), ("elbow", ("elbow_fe",)),
                ("wrist", ("wrist_roll", "wrist_yaw", "wrist_pitch")), ("hip", ("hip_ie", "hip_aa", "hip_fe")),
                ("knee", ("knee_fe",)), ("ankle", ("ankle_ie", "ankle_pd"))]


def group_utilization(tag, q=0.99):
    from teamimp import scene
    J = scene.JOINTS
    with open(os.path.join(MI, f"{tag}_logs.pkl"), "rb") as f:
        D = pickle.load(f)
    L = D["logs"]
    tf = D["failed"][0] if D["failed"] else np.inf
    n = len(L[0]["pel"])
    sel = [x for x in L if x["phase"].startswith(("load-transfer", "lift", "carry", "place")) and x["t"] < tf]
    S = np.array([x["sat"] for x in sel])
    out = {}
    for g, names in JOINT_GROUPS:
        idx = [k for k, j in enumerate(J) if j in names or j[2:] in names]
        cols = [32 * i + k for i in range(n) for k in idx]
        out[g] = float(np.quantile(S[:, cols].max(axis=1), q))
    return out


RATED = 25.0        # published payload of one Apollo (kg)


def fig_capacity(path=os.path.join(FIG, "fig_capacity.pdf")):
    cap = json.load(open(os.path.join(ROOT, "results", "capacity.json")))
    met = json.load(open(os.path.join(MI, "metrics.json")))
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.45), gridspec_kw=dict(width_ratios=[1.0, 1.05]))
    plt.subplots_adjust(left=0.115, right=0.995, bottom=0.265, top=0.97, wspace=0.36)
    ax = axs[0]
    cols = {4: BLUE, 3: ORANGE, 2: AQUA}
    for N in (4, 3, 2):
        ks = sorted([k for k in cap if k.startswith(f"{N}_")], key=lambda k: float(k.split("_")[1]))
        ms = np.array([float(k.split("_")[1]) for k in ks])
        ax.plot(ms / N, [cap[k]["single"] for k in ks], color=cols[N], lw=0.9, label=f"$N={N}$")
        ax.plot(ms / N, [cap[k]["double"] for k in ks], color=cols[N], lw=0.6, ls=":")
    ax.axvline(RATED, color=INK, lw=0.6, ls="--")
    ax.text(RATED + 1.5, 1.52, "rated", fontsize=6.0, va="top", ha="left", color=INK)
    runs = {(4, 183.36): "n4_kz0", (3, 183.36): "n3_kz0", (2, 183.36): "n2_kz0",
            (4, 63.07): "al_n4_kz0", (3, 63.07): "al_n3_kz0", (2, 63.07): "al_n2_kz0"}
    for k in met:
        if k.startswith("sweep_n"):
            N, mass = int(k[7]), float(k.split("_")[2])
            runs[(N, mass)] = k
    for (N, mass), tag in runs.items():
        key = f"{N}_{mass}" if f"{N}_{mass}" in cap else f"{N}_{mass:.1f}"
        if key not in cap or tag not in met:
            continue
        u = cap[key]["single"]
        if met[tag]["completed"]:
            ax.plot(mass / N, u, "o", color=cols[N], ms=3.0, mec="white", mew=0.4, zorder=5)
        else:
            ax.plot(mass / N, u, "x", color=cols[N], ms=3.6, mew=1.0, zorder=5)
    ax.axhline(1.0, color=GRAY, lw=0.6)
    ax.set_xlim(10, 100)
    ax.set_ylim(0.2, 1.6)
    ax.set_xlabel("girder mass per robot (kg)")
    ax.set_ylabel(r"capacity estimate $U^\star$")
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.3)
    label_box(ax, "(a)", x=0.985, y=0.97, ha="right")
    ax = axs[1]
    names = [g for g, _ in JOINT_GROUPS]
    x = np.arange(len(names))
    w = 0.38
    # completed 63 kg runs (N = 3 loses the girder, and its utilization saturates in the loss)
    for k, (N, tag) in enumerate(((4, "al_n4_kz0"), (2, "al_n2_kz0"))):
        u = group_utilization(tag)
        ax.bar(x + (k - 0.5) * w, [u[g] for g in names], w * 0.92, color=cols[N], lw=0, label=f"$N={N}$")
    ax.axhline(1.0, color=GRAY, lw=0.6)
    ax.axhline(0.85, color=GRAY, lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=50, ha="right", rotation_mode="anchor", fontsize=6.0)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r"$|f_i|/f_i^{\max}$, 99th pct.")
    ax.legend(loc="upper right", fontsize=6.0, ncol=2, columnspacing=0.6, handlelength=1.0)
    label_box(ax, "(b)", x=0.015, y=0.985)
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


# ------------------------------------------------------------------------------------------------ contact tests
REFDISP_AMP = 25          # amplitude (N) of the palm-load shift in the internal-command/reference-displacement test


def fig_contact(path=os.path.join(FIG, "fig_contact.pdf")):
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.30))
    plt.subplots_adjust(left=0.058, right=0.995, bottom=0.285, top=0.95, wspace=0.42)
    # (a) push response
    ax = axs[0]
    p = np.load(os.path.join(MI, "mujoco_push_100.npz"))
    js = json.load(open(os.path.join(MI, "mujoco_push_100.json")))
    t, dr = p["t"], p["dir"]
    ax.plot(t, 1e3 * p["F"] / js["Kp"], color=INK, ls="--", lw=0.7, label=r"$K_p^{-1}F$")
    ax.plot(t, 1e3 * (p["dc"] @ dr), color=ORANGE, lw=0.8, label=r"$p_c$")
    ax.plot(t, 1e3 * (p["db"] @ dr), color=BLUE, lw=0.8, label=r"$p$")
    pn = os.path.join(MI, "mujoco_push_100_nofriction.npz")
    if os.path.exists(pn):
        q = np.load(pn)
        ax.plot(q["t"], 1e3 * (q["db"] @ q["dir"]), color=BLUE, lw=0.8, ls=(0, (1, 1.2)),
                label=r"$p$, no friction")
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-1.5, 31)
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel("displacement (mm)")
    ax.legend(loc="upper center", fontsize=6.0, handlelength=1.4, ncol=2, columnspacing=0.6,
              bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
    label_box(ax, "(a)", x=0.985, y=0.22, ha="right", va="bottom")
    # (b), (c) internal command versus reference displacement
    r = np.load(os.path.join(MI, f"mujoco_refdisp_{REFDISP_AMP}.npz"))
    var = (("psi", BLUE, "-", r"internal command $\psi'$"), ("hand", ORANGE, "-", "hand displacement [6]"))
    ax = axs[1]
    for v, c, ls, lab in var:
        ax.semilogy(r[f"t_{v}"], np.maximum(1e3 * np.linalg.norm(r[f"dp_{v}"], axis=1), 1e-5), color=c, ls=ls,
                    lw=0.8, label=lab)
    pn = os.path.join(MI, f"mujoco_refdisp_{REFDISP_AMP}_nofriction.npz")
    if os.path.exists(pn):                     # dotted: without joint friction
        rn = np.load(pn)
        for v, c, ls, lab in var:
            ax.semilogy(rn[f"t_{v}"], np.maximum(1e3 * np.linalg.norm(rn[f"dp_{v}"], axis=1), 1e-5), color=c,
                        ls=(0, (1, 1.2)), lw=0.8)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(1e-4, 20)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel(r"girder deviation (mm)")
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.4)
    label_box(ax, "(b)")
    ax = axs[2]
    ytop = 0.0
    for v, c, ls, lab in var:
        tt, yy = r[f"t_{v}"], r[f"dpal_{v}"]
        k = max(1, int(round(0.05 / np.median(np.diff(tt)))))      # 50 ms moving average
        ys = np.convolve(np.pad(yy, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
        ax.plot(tt, ys, color=c, ls=ls, lw=0.8, label=lab)
        ytop = max(ytop, float(np.max(ys[tt <= 4.0])))
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, nice_top(1.12 * ytop))
    ax.set_xlabel("time (s)", labelpad=0.5)
    ax.set_ylabel("palm-force change (N)")
    label_box(ax, "(c)")
    # (d) Remark 3: bias versus model mass
    ax = axs[3]
    mb = json.load(open(os.path.join(MI, "model_error_offset.json")))
    pr = mb["prediction_curve"]
    ax.plot(pr["mass"], pr["z_mm"], color=INK, ls="--", lw=0.7, label="predicted")
    lost = [row["model_mass"] for row in mb["runs"] if not row["completed"]]
    for row in mb["runs"]:
        if row["measured_z_mm"] is not None:
            ax.plot(row["model_mass"], row["measured_z_mm"], "o", color=BLUE, ms=3.2, mec="white", mew=0.4,
                    zorder=5)
    zs = np.array(pr["z_mm"])
    ylo, yhi = 1.15 * min(zs.min(), 0.0) - 2.0, 1.15 * max(zs.max(), 0.0) + 2.0
    if lost:                                   # girder lost: no measurement
        ax.axvspan(min(lost) - 2.5, max(lost) + 2.5, color=LGRAY, alpha=0.45, lw=0, zorder=0)
        ax.text(0.5 * (min(lost) + max(lost)), ylo + 0.08 * (yhi - ylo), "girder\nlost", ha="center",
                va="bottom", fontsize=6.0, color=INK)
    ax.plot([], [], "o", color=BLUE, ms=3.2, mec="white", mew=0.4, label="measured")
    ax.set_xlim(min(pr["mass"]), max(pr["mass"]))
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("girder model mass (kg)", labelpad=0.5)
    ax.set_ylabel(r"bias $(p_c-p)_z$ (mm)")
    ax.legend(loc="center right", fontsize=6.0, handlelength=1.4, bbox_to_anchor=(1.0, 0.62))
    label_box(ax, "(d)", x=0.985, y=0.95, ha="right", va="top")
    fig.savefig(path)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("exact", "all"):
        fig_exact()
    if what in ("mission", "all"):
        fig_mission_paper()
    if what in ("traj", "all"):
        fig_traj()
    if what in ("capacity", "all"):
        fig_capacity()
    if what in ("contact", "all"):
        fig_contact()
