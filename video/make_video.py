"""Render the mission video from logged MuJoCo states (qpos at 100 Hz).

python video/make_video.py all                      # renders all segments and assembles the video
python video/make_video.py cards,go,pick,closeup    # renders selected segments (PNG) into video/frames/<name>/
python video/make_video.py assemble                 # concatenates the segments into video/team_impedance_mission.mp4
"""
import os
import sys
import json
import pickle
import subprocess
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from teamimp import mission as MS, render, scene

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results", "mission")
FR = os.path.join(ROOT, "video", "frames")
W, H = 1280, 720
FPS = 25
from matplotlib import font_manager
FONT = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
FONTB = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))


def font(sz, bold=False):
    try:                                              # DejaVu Sans (any system with fonts-dejavu)
        return ImageFont.truetype(FONTB if bold else FONT, sz)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", sz)
        except OSError:
            return ImageFont.load_default()


# --------------------------------------------------------------------------------------------- data
def load(tag):
    d = np.load(os.path.join(RES, f"{tag}.npz"), allow_pickle=True)
    with open(os.path.join(RES, f"{tag}_logs.pkl"), "rb") as f:
        lg = pickle.load(f)
    hud = []
    for L in lg["logs"]:
        palms = [c[4][2] for c in L["conn"] if c[0] == "grasp"]
        rho = [c[6][0] for c in L["conn"]]
        hud.append(dict(palms=palms, sat=float(np.max(L["sat"])),
                        eI=float(np.linalg.norm(L["eI"])) if "eI" in L else None,
                        share=(sum(palms) / lg["W"]) if palms else None, phase=L["phase"], mode=L["mode"]))
    return d, lg, hud


def model_for(tag, d, lg):
    cfg = MS.Cfg(**{k: v for k, v in lg["cfg"].items() if k in MS.Cfg().__dict__})
    cfg.background = True
    cfg.verbose = False
    M = MS.Mission(cfg)
    return M


# --------------------------------------------------------------------------------------------- cameras
def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class CamPath:
    """Keyframed free camera: list of (t, lookat(3), distance, azimuth, elevation); 'track' adds the girder
    position (x, y) to the lookat."""

    def __init__(self, keys, track=None):
        self.keys = keys
        self.track = track

    def __call__(self, t, beam_xy=None):
        k = self.keys
        if t <= k[0][0]:
            a = b = k[0]
            s = 0.0
        elif t >= k[-1][0]:
            a = b = k[-1]
            s = 0.0
        else:
            j = max(i for i in range(len(k)) if k[i][0] <= t)
            a, b = k[j], k[j + 1]
            s = smooth((t - a[0]) / (b[0] - a[0]))
        look = (1 - s) * np.array(a[1]) + s * np.array(b[1])
        if self.track is not None and beam_xy is not None:
            look = look + self.track * np.r_[beam_xy, 0.0]
        dist = (1 - s) * a[2] + s * b[2]
        az = (1 - s) * a[3] + s * b[3]
        el = (1 - s) * a[4] + s * b[4]
        return render.free_cam(look, dist, az, el)


# --------------------------------------------------------------------------------------------- overlays
PHASE_TEXT = [
    ("settle", "GO", "team gait: every step is a compiled support mode"),
    ("go", "GO", "team gait: every step is a compiled support mode"),
    ("pre-reach", "PICK", "palms go under the bottom flange"),
    ("reach-in", "PICK", "palms go under the bottom flange"),
    ("contact", "PICK", "all palms touch: grasps close, the girder joins the compiled team"),
    ("load-transfer", "PICK", "external impedance takes the load from the trestles"),
    ("lift", "LIFT", "0.10 m lift; internal impedance on"),
    ("carry1-step", "CARRY", "single support: the internal channel shifts the hand loads"),
    ("carry2-step", "CARRY", "single support: the internal channel shifts the hand loads"),
    ("carry3-step", "CARRY", "single support: the internal channel shifts the hand loads"),
    ("carry4-step", "CARRY", "single support: the internal channel shifts the hand loads"),
    ("carry", "CARRY", "double support: the team moves the girder 0.25 m per cycle"),
    ("place", "PLACE", "lowered onto rubber pads by the external impedance"),
    ("unload", "UNLOAD", "external impedance hands the load to the trestles"),
    ("release", "RELEASE", "grasps open"),
    ("stow", "RELEASE", "grasps open"),
    ("back", "DONE", "team steps back"),
    ("done", "DONE", "mission complete"),
    ("hold", "", ""),
]


def phase_caption(ph):
    for key, head, text in PHASE_TEXT:
        if ph.startswith(key):
            return head, text
    return "", ""


def draw_overlay(img, info):
    im = Image.fromarray(img).convert("RGBA")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    # title (top-left)
    tw = max(dr.textlength(info["title"], font=font(24, True)), dr.textlength(info["subtitle"], font=font(17)))
    dr.rounded_rectangle([18, 16, 18 + max(560, tw + 30), 16 + 66], radius=10, fill=(15, 18, 24, 170))
    dr.text((32, 22), info["title"], font=font(24, True), fill=(255, 255, 255, 255))
    dr.text((32, 52), info["subtitle"], font=font(17), fill=(210, 220, 230, 255))
    # time (top-right)
    s = f"t = {info['t']:5.1f} s    ×{info['speed']:g}"
    dr.rounded_rectangle([W - 250, 16, W - 18, 56], radius=10, fill=(15, 18, 24, 170))
    dr.text((W - 236, 24), s, font=font(20, True), fill=(255, 255, 255, 255))
    # caption (bottom-left)
    head, text = info.get("caption", ("", ""))
    if head or text:
        w = 30 + max(dr.textlength(text, font=font(18)), dr.textlength(head, font=font(24, True)), 200)
        dr.rounded_rectangle([18, H - 92, 18 + min(w, W - 380), H - 18], radius=10, fill=(15, 18, 24, 175))
        col = info.get("head_color", (255, 196, 40, 255))
        dr.text((32, H - 86), head, font=font(24, True), fill=col)
        dr.text((32, H - 52), text, font=font(18), fill=(235, 238, 240, 255))
    # HUD (bottom-right): palm loads and certificates
    hud = info.get("hud")
    if hud is not None:
        x0, y0, x1, y1 = W - 340, H - 210, W - 18, H - 18
        dr.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=(15, 18, 24, 175))
        dr.text((x0 + 14, y0 + 8), "palm loads (N)", font=font(15, True), fill=(230, 230, 230, 255))
        palms = hud["palms"]
        n = max(len(palms), 1)
        bw = (x1 - x0 - 28) / max(n, 8)
        base = y0 + 112
        for k, f in enumerate(palms):
            hgt = min(max(f, 0.0) / 450.0, 1.0) * 76
            c = (90, 200, 255, 255) if f > 8 else (255, 90, 80, 255)
            dr.rectangle([x0 + 14 + k * bw + 2, base - hgt, x0 + 14 + (k + 1) * bw - 2, base], fill=c)
        if not palms:
            dr.text((x0 + 14, y0 + 60), "no grasp", font=font(15), fill=(180, 180, 180, 255))
        dr.line([x0 + 14, base, x1 - 14, base], fill=(150, 150, 150, 255), width=1)
        sat = hud["sat"]
        col = (120, 220, 120, 255) if sat < 0.9 else ((255, 196, 40, 255) if sat < 1.0 else (255, 80, 70, 255))
        dr.text((x0 + 14, base + 8), f"peak torque  {100 * sat:5.1f} %", font=font(16), fill=col)
        if hud.get("eI") is not None:
            dr.text((x0 + 14, base + 32), f"internal-wrench error  {hud['eI']:6.1f} N", font=font(16),
                    fill=(230, 230, 230, 255))
        if hud.get("share") is not None:
            dr.text((x0 + 14, base + 56), f"girder weight on palms  {100 * hud['share']:5.1f} %", font=font(16),
                    fill=(230, 230, 230, 255))
    if info.get("banner"):
        txt, colr = info["banner"]
        fw = font(44, True)
        tw = dr.textlength(txt, font=fw)
        dr.rounded_rectangle([W / 2 - tw / 2 - 24, 110, W / 2 + tw / 2 + 24, 180], radius=12, fill=colr)
        dr.text((W / 2 - tw / 2, 118), txt, font=fw, fill=(255, 255, 255, 255))
    return np.array(Image.alpha_composite(im, ov).convert("RGB"))


def clear_dir(out):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".png"):
            os.remove(os.path.join(out, f))


def card(lines, out, n_frames, sub=None):
    clear_dir(out)
    im = Image.new("RGB", (W, H), (14, 18, 26))
    dr = ImageDraw.Draw(im)
    y = H / 2 - 40 * len(lines) / 2 - (30 if sub else 0)
    for k, (txt, sz, bold, col) in enumerate(lines):
        f = font(sz, bold)
        tw = dr.textlength(txt, font=f)
        while tw > W - 80 and sz > 12:                  # shrink lines that do not fit the frame
            sz -= 1
            f = font(sz, bold)
            tw = dr.textlength(txt, font=f)
        dr.text(((W - tw) / 2, y), txt, font=f, fill=col)
        y += sz * 1.55
    arr = np.array(im)
    for k in range(n_frames):
        Image.fromarray(arr).save(os.path.join(out, f"{k:05d}.png"))


# --------------------------------------------------------------------------------------------- segments
def render_run(tag, name, t0, t1, speed, cam, title, subtitle, banner=None, banner_t=None, caption=None,
               hud=True, fail_banner=True):
    d, lg, hudv = load(tag)
    M = model_for(tag, d, lg)
    out = os.path.join(FR, name)
    clear_dir(out)
    S = render.Snap(M.m, W, H)
    t = d["t"]
    t1 = min(t1, t[-1])
    nfr = int((t1 - t0) / speed * FPS)
    fail_t = float(d["failed"][0])
    tic = time.time()
    for k in range(nfr):
        ts = t0 + k * speed / FPS
        j = int(np.clip(np.searchsorted(t, ts), 0, len(t) - 1))
        q = d["qpos"][j]
        bxy = d["beam"][j][:2]
        img = S(q, cam(ts, bxy))
        ph = str(d["phase"][j])
        info = dict(t=ts, speed=speed, title=title, subtitle=subtitle,
                    caption=caption(ts, ph) if callable(caption) else phase_caption(ph),
                    hud=hudv[j] if hud else None)
        if fail_banner and fail_t > 0 and ts >= fail_t:
            info["banner"] = ("FAILURE: " + str(d["fail_reason"][0]), (190, 40, 40, 220))
            info["caption"] = ("", "")
        if banner and banner_t and banner_t[0] <= ts <= banner_t[1]:
            info["banner"] = banner
        img = draw_overlay(img, info)
        Image.fromarray(img).save(os.path.join(out, f"{k:05d}.png"))
        if k % 50 == 0:
            print(f"  {name}: frame {k}/{nfr}  ({(time.time() - tic) / max(k, 1):.2f} s/frame)", flush=True)
    S.close()


def assemble(names, out_path):
    lst = os.path.join(FR, "list.txt")
    parts = []
    for nm in names:
        mp4 = os.path.join(FR, f"{nm}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
                        os.path.join(FR, nm, "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                        "-preset", "medium", mp4], check=True)
        parts.append(mp4)
    with open(lst, "w") as f:
        for p in parts:
            escaped = Path(p).resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                    out_path], check=True)
    print("video:", out_path)


# --------------------------------------------------------------------------------------------- script
def phase_times(tag):
    d = np.load(os.path.join(RES, f"{tag}.npz"), allow_pickle=True)
    t, ph = d["t"], d["phase"]
    out = {}
    for k in range(len(t)):
        p = str(ph[k])
        if p not in out:
            out[p] = [t[k], t[k]]
        out[p][1] = t[k]
    return out


MAIN = "al_n4_kz0"       # N = 4, integral internal impedance (K_z = 0), 63 kg aluminium girder
N3 = "al_n3_kz0"         # N = 3, same girder
NOSHARE = "al_n4_noshare_kz0"
WHITE = (255, 255, 255)
GREY = (190, 200, 210)
GOLD = (255, 196, 40)
RATED = 25.0             # published payload of one Apollo (kg)


def outcome(tag):
    """(exists, completed, mass, n) of a mission run."""
    p = os.path.join(RES, f"{tag}.npz")
    if not os.path.exists(p):
        return False, False, None, None
    d = np.load(p, allow_pickle=True)
    return True, float(d["failed"][0]) < 0, float(d["mass"][0]), int(d["n"][0])


def capacity(n, mass):
    cap = json.load(open(os.path.join(ROOT, "results", "capacity.json")))
    for key in (f"{n}_{mass}", f"{n}_{round(mass, 2)}", f"{n}_{mass:.1f}"):
        if key in cap:
            return cap[key]
    return None


def seg_cards():
    ok, done, m, n = outcome(MAIN)
    u = capacity(4, 63.07) or {}
    card([("External and Internal Impedance for Robot Teams", 38, True, WHITE),
          ("Sharing a Load: Exact Decoupling and Decay Certificates", 38, True, WHITE),
          ("supplementary video", 24, False, GREY),
          ("", 20, False, GREY),
          (f"MuJoCo contact physics: four Apollo humanoids carry a {m:.0f} kg aluminium girder", 24, False, GOLD),
          (f"{m / 4:.0f} kg per robot, within the {RATED:.0f} kg payload published for Apollo", 22, False, GREY),
          ("go  -  pick  -  lift  -  carry 1 m  -  place  -  unload  -  step back", 22, False, GREY)],
         os.path.join(FR, "c0_title"), 5 * FPS)
    card([("Team of four (N = 4)", 44, True, WHITE),
          (f"capacity estimate (Corollary 1 with torque limits): U* = {u.get('double', float('nan')):.2f} in double, "
           f"{u.get('single', float('nan')):.2f} in single support", 24, False, GREY),
          ("internal channel: QP over the internal wrench and the foot efforts with the margins of Corollary 1",
           22, False, GREY),
          ("integral internal impedance (K_z = 0); controller 500 Hz, MuJoCo 1 kHz", 22, False, GREY)],
         os.path.join(FR, "c1_n4"), int(4 * FPS))
    ok3, done3, _, _ = outcome(N3)
    okn, donen, _, _ = outcome(NOSHARE)
    lines = []
    if ok3 and done3:
        lines.append((f"N = 3 carries the same girder ({m / 3:.0f} kg per robot)", 24, False, GREY))
    if okn and not donen:
        lines.append(("N = 4 without the internal command (lambda_d = 0): the girder is lost", 24, False, GREY))
    if lines:
        head = "Team size and the internal command" if (ok3 and done3) else "Without the internal command"
        card([(head, 44, True, WHITE)] + lines, os.path.join(FR, "c2_fail"), int(4 * FPS))
    end = [("Summary", 44, True, WHITE),
           ("Exact-model, fixed-mode analysis: decoupling and certified decay rates", 24, False, GREY),
           ("under the local theorem assumptions, with a measured environment wrench", 24, False, GREY),
           (f"MuJoCo contact simulation: N = 4 carries the {m:.0f} kg girder", 24, False, GOLD)]
    if okn and not donen:
        end.append(("without the internal command (lambda_d = 0), the girder is lost", 24, False, GREY))
    card(end, os.path.join(FR, "c9_end"), 5 * FPS)


def seg_n4(parts=("go", "pick", "carry", "unload")):
    pt = phase_times(MAIN)
    t_pick = pt["pre-reach"][0]
    t_carry = pt["carry1-step-l"][0]
    t_place = pt["place"][0]
    t_end = pt["done"][1]
    track = np.array([1.0, 1.0, 0.0])
    cam = CamPath([(0.0, (0.0, 0.0, 0.80), 7.2, 152.0, -24.0),
                   (t_pick - 2.0, (0.0, 0.0, 0.85), 5.0, 132.0, -17.0),
                   (t_pick + 3.0, (0.0, -0.05, 0.85), 3.5, 112.0, -9.0),
                   (t_carry - 2.0, (0.0, 0.0, 0.88), 3.8, 118.0, -12.0),
                   (t_carry + 4.0, (0.0, 0.0, 0.85), 5.0, 120.0, -17.0),
                   (t_carry + 22.0, (0.0, 0.0, 0.85), 5.2, 88.0, -20.0),
                   (t_place, (0.0, 0.0, 0.85), 4.8, 60.0, -15.0),
                   (t_end, (0.0, -0.2, 0.80), 6.4, 28.0, -20.0)], track=track)
    title = f"N = 4 Apollo humanoids, {outcome(MAIN)[2]:.0f} kg aluminium girder"
    sub = "MuJoCo contact physics, 1 kHz; controller 500 Hz"
    if "go" in parts:
        render_run(MAIN, "s1_n4_go", 0.0, t_pick, 3.0, cam, title, sub)
    if "pick" in parts:
        render_run(MAIN, "s2_n4_pick", t_pick, t_carry, 1.5, cam, title, sub)
    if "carry" in parts:
        render_run(MAIN, "s3_n4_carry", t_carry, t_place, 3.0, cam, title, sub)
    if "unload" in parts:
        render_run(MAIN, "s5_n4_unload", t_place, t_end, 2.5, cam, title, sub)


def seg_n4_closeup():
    pt = phase_times(MAIN)
    t0 = pt["carry1-step-l"][0] + 0.3
    t1 = pt["carry1-shift"][0] + 0.6
    track = np.array([1.0, 1.0, 0.0])
    cam = CamPath([(t0, (-0.2, 0.0, 0.95), 2.6, 170.0, -12.0), (t1, (-0.2, 0.0, 0.95), 2.6, 190.0, -12.0)],
                  track=track)
    render_run(MAIN, "s4_n4_closeup", t0, t1, 1.0, cam, "single support, close-up",
               "the internal channel redistributes palm loads with little girder motion")


def seg_fail(tag, name, title, sub, t0_phase="lift", speed=1.0, banner=None, t_after=1.5, hold=1.5):
    """Failure run from phase t0_phase until t_after s after the failure is detected; the last frame is held for
    hold s. (After a fall, only the soles collide with the ground, so the rest of a fallen robot is not shown.)"""
    d = np.load(os.path.join(RES, f"{tag}.npz"), allow_pickle=True)
    pt = phase_times(tag)
    t0 = pt[t0_phase][0]
    tf = float(d["failed"][0])
    t1 = min(float(d["t"][-1]), tf + t_after)
    track = np.array([1.0, 1.0, 0.0])
    cam = CamPath([(t0, (0.0, 0.0, 0.85), 5.0, 128.0, -16.0), (t1, (0.0, 0.0, 0.85), 4.6, 105.0, -14.0)], track=track)
    render_run(tag, name, t0, t1, speed, cam, title, sub,
               banner=(banner, (40, 60, 120, 210)) if banner else None, banner_t=(t0, tf - 0.3) if banner else None)
    out = os.path.join(FR, name)
    frames = sorted(f for f in os.listdir(out) if f.endswith(".png"))
    last = Image.open(os.path.join(out, frames[-1]))
    for k in range(len(frames), len(frames) + int(hold * FPS)):
        last.save(os.path.join(out, f"{k:05d}.png"))


def seg_n3():
    """N = 3 with the same girder: pick to the end of the carry."""
    pt = phase_times(N3)
    t0, t1 = pt["pre-reach"][0], pt["place"][0]
    track = np.array([1.0, 1.0, 0.0])
    cam = CamPath([(t0, (0.0, 0.0, 0.85), 5.2, 128.0, -17.0), (t1, (0.0, 0.0, 0.85), 5.0, 100.0, -16.0)],
                  track=track)
    render_run(N3, "s6_n3", t0, t1, 3.0, cam, f"N = 3, same girder ({outcome(N3)[2]:.0f} kg)",
               "the compiled team model changes with the team; the controller is the same")


def main(which):
    todo = set(which.split(","))
    allp = "all" in todo
    if allp or "cards" in todo:
        seg_cards()
    parts = [p for p in ("go", "pick", "carry", "unload") if allp or "n4" in todo or p in todo]
    if parts:
        seg_n4(parts)
    if allp or "closeup" in todo:
        seg_n4_closeup()
    ok3, done3, _, _ = outcome(N3)
    if (allp or "n3" in todo) and ok3 and done3:
        seg_n3()
    okn, donen, _, _ = outcome(NOSHARE)
    if (allp or "noshare" in todo) and okn and not donen:
        seg_fail(NOSHARE, "s8_noshare", "N = 4 without the internal command",
                 "lambda_d = 0: minimum-norm load distribution", t0_phase="load-transfer", speed=1.5,
                 banner="lambda_d = 0: contact cones violated", t_after=0.3)
    if allp or "assemble" in todo:
        names = ["c0_title", "c1_n4", "s1_n4_go", "s2_n4_pick", "s3_n4_carry", "s4_n4_closeup", "s5_n4_unload",
                 "c2_fail", "s6_n3", "s8_noshare", "c9_end"]
        names = [n for n in names if os.path.isdir(os.path.join(FR, n))]
        assemble(names, os.path.join(ROOT, "video", "team_impedance_mission.mp4"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
