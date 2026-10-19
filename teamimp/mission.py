"""Construction-site mission with a humanoid team: go (walk to the girder), pick (reach under it, close the
grasps, transfer the load from the trestles), carry (quasi-static team gait), and unload (place it on the
target trestles, release, step back).

Every phase runs in a compiled support/grasp mode; contact events switch the mode (Remark 6: the theory
applies within each mode). Load sharing between the robots and between the feet of each robot is chosen by
the internal channel (share.LoadShare) under the feasibility margins of Corollary 1.
"""
import time
import numpy as np
import mujoco

from . import scene, team, ik, controller, setup, sim_mj, share
from .traj import PoseSchedule, RelPose, VecSchedule
from .so3 import rot_z, log, hat


class Cfg:
    def __init__(self, **kw):
        self.n = 4
        self.profile = "AL 280"            # aluminium I-beam with the HEA 280 cross-section (63 kg at 2.4 m)
        self.beam_len = 2.4
        self.mass = None                  # None: from the profile table
        self.ctrl_mass = None             # payload mass assumed by the controller (None: true mass)
        self.zA = 0.94                    # top of the pickup trestle pads (girder bottom)
        self.zB = 0.94                    # top of the target trestle pads
        self.YB = 1.0                     # target position of the girder (y)
        self.x_sup = 0.95                 # trestle positions along the girder
        self.y_off = 0.36
        self.x_sp = 0.8
        self.hand_dx = 0.26
        self.foot_dy = 0.11
        self.inset = 0.07
        self.D_app = 0.5                  # approach distance
        self.step = 0.25
        self.dz_lift = 0.10
        self.z_walk = 1.0
        self.walk_hand = (0.34, 0.24, -0.12)   # palm position relative to the pelvis while walking
        self.z_pick = 0.91
        self.w_com_pick = 0.0             # CoM placement in the pick posture (lean back against the load)
        self.com_dx_pick = 0.0
        self.T_ss = 1.5                   # weight shift before lift-off
        self.T_unload = 0.8               # unloading of the swing foot when the CoM is already over the stance
        self.T_swing = 1.0
        self.T_land = 0.6
        self.T_settle = 0.3               # after touch-down, before moving the pelvis
        self.T_reload = 0.6               # load ramp of a new foot
        self.T_back = 1.5
        self.T_body = 2.6
        self.apex = 0.06
        self.delta_free = 0.100           # (unused) lateral pelvis shift in single support (unloaded)
        self.delta_load = 0.06            # (unused)
        self.shift_frac_free = 0.95       # pelvis over the stance foot (fraction of the lateral distance)
        self.shift_frac_load = 0.25       # carrying: the internal channel shifts the hand loads
        self.share_internal = True        # internal channel used for load sharing
        self.share_kw = {}                # options of share.LoadShare (e.g., w_soft_palm, fmin_palm)
        self.internal_mode = "impedance"  # "impedance" | "feedforward" | "off"
        self.kz = 1.0                     # stiffness-type internal impedance (K_z = I); 0: integral action
        self.z_max = 250.0
        self.blend_tau = 0.0              # optional bumpless transfer of the effort at mode switches (s); off
        self.background = True
        self.dt_ctrl = 0.002
        self.log_dt = 0.01
        self.gain_scale = dict(setup.GAIN_SCALE)
        self.stop_on_fail = True
        self.t_max = 200.0
        self.verbose = True
        self.Kp = (6000.0, 6000.0, 6000.0)
        self.sensor_noise = None          # (sigma_force N, sigma_moment N m) of the palm, sole and h measurements
        self.noise_seed = 0
        self.kin_err = 0.0                # kinematic model error: std (m) of the link offsets of the controller model
        self.kin_err_seed = 0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)


def perturb_kinematics(mc, sigma, seed=0):
    """Kinematic model error: random offsets (std sigma per axis) of the positions of all robot links relative to
    their parents in the controller model (the root links keep their free joints)."""
    rng = np.random.default_rng(seed)
    for b in range(1, mc.nbody):
        name = mc.body(b).name
        if not name.startswith("r") or mc.body_parentid[b] == 0:
            continue
        mc.body_pos[b] += sigma * rng.standard_normal(3)


class MissionRefs(controller.Refs):
    def __init__(self):
        self.pel = {}
        self.jnt = {}
        self.eef = {}
        self.pay = None

    def payload(self, t):
        return self.pay(t)

    def pelvis(self, i, t):
        return self.pel[i](t)

    def joints(self, i, t):
        return self.jnt[i](t)

    def ee(self, i, kind, s, t):
        return self.eef[(i, kind, s)](t)


WALK = dict(scene.STAND)
WALK.update(l_shoulder_aa=0.10, r_shoulder_aa=-0.10, l_shoulder_fe=-0.30, r_shoulder_fe=-0.30,
            l_elbow_fe=-1.40, r_elbow_fe=-1.40, l_wrist_roll=1.50, r_wrist_roll=-1.50, l_wrist_yaw=0.0, r_wrist_yaw=0.0,
            l_wrist_pitch=0.0, r_wrist_pitch=0.0, l_knee_fe=0.9, r_knee_fe=0.9, l_hip_fe=-0.45, r_hip_fe=-0.45,
            l_ankle_pd=-0.45, r_ankle_pd=-0.45)
PICK_SEED = dict(ik.CARRY)
PICK_SEED.update(l_shoulder_fe=-0.7, r_shoulder_fe=-0.7, l_elbow_fe=-1.3, r_elbow_fe=-1.3, l_wrist_roll=1.3,
                 r_wrist_roll=-1.3, l_wrist_pitch=0.0, r_wrist_pitch=0.0, l_wrist_yaw=0.0, r_wrist_yaw=0.0,
                 l_shoulder_aa=0.1, r_shoulder_aa=-0.1, l_knee_fe=0.8, r_knee_fe=0.8, l_hip_fe=-0.4,
                 r_hip_fe=-0.4, l_ankle_pd=-0.4, r_ankle_pd=-0.4)
RED = scene.TORSO + scene.NECK + [scene.ARM_RED["l"], scene.ARM_RED["r"]]


class Mission:
    def __init__(self, cfg):
        self.cfg = c = cfg
        n = c.n
        h, b = scene.PROFILES[c.profile][:2]
        self.h, self.b = h, b
        self.mass = c.mass if c.mass is not None else scene.PROFILES[c.profile][4] * c.beam_len
        self.W = self.mass * 9.81
        self.lay = scene.Layout(n, x_sp=c.x_sp, y_off=c.y_off)
        self.beamA = np.array([0.0, 0.0, c.zA + h / 2 + 0.0005])
        sup = [("A0", -c.x_sup, 0.0, c.zA, "wood"), ("A1", c.x_sup, 0.0, c.zA, "wood"),
               ("B0", -c.x_sup, c.YB, c.zB, "rubber"), ("B1", c.x_sup, c.YB, c.zB, "rubber")]
        kr = setup.rate_gains(c.gain_scale)
        cm = self.mass if c.ctrl_mass is None else c.ctrl_mass
        self.R = sim_mj.MjRunner(n, kr, dt_ctrl=c.dt_ctrl, spec_kw=dict(
            supports=sup, beam_pos=tuple(self.beamA), profile=c.profile, beam_len=c.beam_len, mass=self.mass,
            background=c.background), payload_scale=cm / self.mass,
            ctrl_model_scale=(lambda mc: perturb_kinematics(mc, c.kin_err, c.kin_err_seed)) if c.kin_err > 0 else None)
        self.m, self.d, self.tm = self.R.m, self.R.d, self.R.tm
        self.R.blend_tau = c.blend_tau
        if c.sensor_noise is not None:
            self.R.noise = tuple(float(x) for x in c.sensor_noise)
            self.R.rng = np.random.default_rng(c.noise_seed)
        self.side = [self.lay.stations[i][1] for i in range(n)]
        self.fwd = [np.array([0.0, -self.side[i], 0.0]) for i in range(n)]    # towards the girder
        self._keyframes()
        self._init_state()
        self.phase_name = "init"
        self.events = []
        self.logs = []
        self.failed = None
        self.t_wall = 0.0

    # ------------------------------------------------------------------ keyframes (IK)
    def _keyframes(self):
        c, tm, mc = self.cfg, self.tm, self.R.mc
        h, b = self.h, self.b
        # pick posture: palms under the bottom face at A (crouched)
        q_pick, _ = ik.team_config(mc, tm, self.beamA, self.lay, iters=600, hand_dx=c.hand_dx, grasp="under",
                                   beam_half_w=b / 2, beam_half_h=h / 2, posture=PICK_SEED, foot_dy=c.foot_dy,
                                   pelvis_z=c.z_pick, inset=c.inset, w_com=c.w_com_pick, com_dx=c.com_dx_pick)
        # walk posture at the approach start: palms up in front of the hips, below the girder's bottom face
        q_walk = self._walk_ik(mc, tm, c.D_app)
        d = mujoco.MjData(mc)
        self.kf = {}
        for name, q in (("pick", q_pick), ("walk", q_walk)):
            d.qpos[:] = q
            mujoco.mj_kinematics(mc, d)
            mujoco.mj_comPos(mc, d)
            kf = {}
            for r in tm.robots:
                pp, Rp = d.xpos[r.pelvis].copy(), d.xmat[r.pelvis].reshape(3, 3).copy()
                feet = {s: (d.site_xpos[r.site[("foot", s)]].copy(), d.site_xmat[r.site[("foot", s)]].reshape(3, 3).copy())
                        for s in "lr"}
                hands = {s: (d.site_xpos[r.site[("hand", s)]].copy(), d.site_xmat[r.site[("hand", s)]].reshape(3, 3).copy())
                         for s in "lr"}
                mid = 0.5 * (feet["l"][0] + feet["r"][0])
                Ry = rot_z(-self.side[r.i] * np.pi / 2)
                kf[r.i] = dict(pel=(pp, Rp), feet=feet, hands=hands, mid=mid,
                               pel_off=Ry.T @ (pp - mid),              # pelvis offset from the feet midpoint
                               hand_rel={s: (Rp.T @ (hands[s][0] - pp), Rp.T @ hands[s][1]) for s in "lr"},
                               jq=np.array([q[r.qpos_adr[j]] for j in RED]),
                               com=d.subtree_com[r.pelvis].copy())
            self.kf[name] = kf
        self.q_walk, self.q_pick = q_walk, q_pick

    def _walk_ik(self, mc, tm, back):
        c = self.cfg
        q = mc.qpos0.copy()
        q[tm.beam_qadr:tm.beam_qadr + 3] = self.beamA
        q[tm.beam_qadr + 3:tm.beam_qadr + 7] = [1, 0, 0, 0]
        for r in tm.robots:
            x, y, yaw = self.lay.base_xy_yaw(r.i, self.beamA[:2], back=back)
            Ry = rot_z(yaw)
            q[r.qadr:r.qadr + 3] = [x, y, 1.0]
            q[r.qadr + 3:r.qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            for jn in scene.JOINTS:
                q[r.qpos_adr[jn]] = WALK[jn]
            feet = []
            tasks = []
            for s, sg in (("l", 1), ("r", -1)):
                pf = np.array([x, y, 0.0]) + Ry @ np.array([-0.015, sg * c.foot_dy, 0.0])
                feet.append(pf)
                tasks.append((r.site[("foot", s)], pf, Ry, 30.0, np.array([10.0, 10.0, 10.0])))
            mid = np.mean(feet, axis=0)
            pel = np.array([mid[0], mid[1], c.z_walk])
            fwd = Ry @ np.array([1.0, 0, 0])
            Rp = np.column_stack([np.cross(-fwd, [0, 0, 1.0]), -fwd, [0, 0, 1.0]])   # palm up, fingers forward
            for s, sg in (("l", 1), ("r", -1)):
                ph = pel + Ry @ np.array([c.walk_hand[0], sg * c.walk_hand[1], c.walk_hand[2]])
                tasks.append((r.site[("hand", s)], ph, Rp, 20.0, np.array([10.0, 10.0, 3.0])))
            tasks.append((mc.site(f"r{r.i}_pelvis").id, pel, Ry, 10.0, np.array([5.0, 5.0, 5.0])))
            post = np.array([WALK[j] for j in scene.JOINTS])
            q, _ = ik._solve_with_partial(mc, q, r, tasks, 500, post, mid[:2], 0.0)
        return q

    def _init_state(self):
        c, R, tm = self.cfg, self.R, self.tm
        d = self.d
        d.qpos[:] = self.q_walk
        d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, d)
        mode = setup.grasp_mode(tm, self.q_walk, feet=True, grasps=False)
        tm.set_mode(mode)
        refs = MissionRefs()
        kf = self.kf["walk"]
        self.pel_center = {}
        for r in tm.robots:
            pp, Rp = kf[r.i]["pel"]
            refs.pel[r.i] = PoseSchedule(pp, Rp)
            refs.jnt[r.i] = VecSchedule(kf[r.i]["jq"])
            self.pel_center[r.i] = pp.copy()
            for s in "lr":
                rr, Rr = kf[r.i]["hand_rel"][s]
                refs.eef[(r.i, "hand", s)] = RelPose(refs.pel[r.i], rr, Rr)
        pb, Rb = d.xpos[R.beam].copy(), d.xmat[R.beam].reshape(3, 3).copy()
        refs.pay = PoseSchedule(pb, Rb)
        self.refs = refs
        P = setup.default_params(tm, Kp=list(c.Kp), kz=c.kz)
        C = controller.TeamImpedance(tm, P, refs)
        C.internal_mode = c.internal_mode
        C.z_max = c.z_max
        tm.update(d.qpos.copy(), d.qvel.copy())
        C.init_states(0.0)
        self.share = share.LoadShare(**c.share_kw)
        self.share.use_internal = c.share_internal
        C.share = self.share
        R.ctrl = C
        self.C = C
        self.P = P

    # ------------------------------------------------------------------ measurements
    def foot_fz(self, i, s):
        return self.R.foot_wrench(self.R._last["sole"], i, s)[2] if hasattr(self.R, "_last") else 0.0

    def foot_ncon(self, i, s):
        sole = self.R._last["sole"] if hasattr(self.R, "_last") else {}
        return len(sole[(i, s)][2]) if (i, s) in sole else 0

    def palm_fz(self):
        palm = self.R._last["palm"] if hasattr(self.R, "_last") else {}
        out = {}
        for r in self.tm.robots:
            for s in "lr":
                out[(r.i, s)] = palm[(r.i, s)][0][2] if (r.i, s) in palm else 0.0    # force on the girder, upwards
        return out

    def site_pose(self, i, kind, s):
        sid = self.tm.robots[i].site[(kind, s)]
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()


    def pelvis_target(self, i, over=None, frac=1.0, kf="pick"):
        """Pelvis target: the planned neutral position (tied to the girder plan), shifted laterally by `frac` of
        the measured lateral distance to the stance foot `over`."""
        base = self.pel_center[i].copy()
        if over is not None:
            po = self.site_pose(i, "foot", over)[0]
            lat = rot_z(-self.side[i] * np.pi / 2)[:, 1]
            base = base + frac * ((po - base) @ lat) * lat
        return base

    def foot_target(self, i, s, center, kf="pick"):
        """Nominal sole position for a neutral pelvis at `center` (offsets from the keyframe)."""
        K = self.kf[kf][i]
        Ry = rot_z(-self.side[i] * np.pi / 2)
        off = Ry.T @ (K["feet"][s][0] - K["pel"][0])
        p = center + Ry @ np.array([off[0], off[1], 0.0])
        p[2] = 0.0
        return p

    # ------------------------------------------------------------------ mode switching
    def switch(self, feet=None, grasps=None, why=""):
        tm, C, d = self.tm, self.C, self.d
        old = tm.mode
        new = team.Mode(old.feet if feet is None else feet, old.grasps if grasps is None else grasps)
        had_payload = old.payload
        self.R.blend_pending = True
        tm.set_mode(new)
        tm.update(d.qpos.copy(), d.qvel.copy())
        if new.payload and not had_payload:
            p, Rb, v, w = tm.payload_state()
            C.ext = [p.copy(), Rb.copy(), np.r_[v, w]]
        if not new.payload:
            C.ext = None
        if set(new.grasps) != set(old.grasps):
            C.z = np.zeros(tm.nI)
            C.zd = np.zeros(tm.nI)
        self.share.reset()
        self.events.append((self.R.t, why, len(new.feet), len(new.grasps)))
        if self.cfg.verbose:
            print(f"    [{self.R.t:7.3f}] mode: {why}  feet={len(new.feet)} grasps={len(new.grasps)}", flush=True)

    def lift_feet(self, keys):
        feet = dict(self.tm.mode.feet)
        for (i, s) in keys:
            feet.pop((i, s), None)
            p, Rf = self.site_pose(i, "foot", s)
            self.refs.eef[(i, "foot", s)] = PoseSchedule(p, Rf)
        self.switch(feet=feet, why=f"lift-off {keys}")

    def plant_feet(self, keys):
        feet = dict(self.tm.mode.feet)
        for (i, s) in keys:
            p, Rf = self.site_pose(i, "foot", s)
            feet[(i, s)] = (p, Rf)
        self.switch(feet=feet, why=f"touch-down {keys}")

    def close_grasps(self):
        d, tm = self.d, self.tm
        pb, Rb = d.xpos[self.R.beam].copy(), d.xmat[self.R.beam].reshape(3, 3).copy()
        G = {}
        for r in tm.robots:
            for s in "lr":
                ph, Rh = self.site_pose(r.i, "hand", s)
                G[(r.i, s)] = (Rb.T @ (ph - pb), Rb.T @ Rh)
        self.switch(grasps=G, why="grasps closed")
        # desired pose at equilibrium with the measured support wrench: K_p (p_d - p_c) = -f_h
        hf = self.R.h_f[:3]
        pd = pb - np.linalg.solve(self.P.Kp, hf)
        self.refs.pay = PoseSchedule(pd, Rb)

    def open_grasps(self):
        for r in self.tm.robots:
            for s in "lr":
                p, Rh = self.site_pose(r.i, "hand", s)
                self.refs.eef[(r.i, "hand", s)] = PoseSchedule(p, Rh)
        self.switch(grasps={}, why="grasps opened")

    # ------------------------------------------------------------------ phases
    def run(self, phases):
        c = self.cfg
        tic = time.time()
        self.next_log = 0.0
        for ph in phases:
            self.phase_name = ph.name
            if c.verbose:
                print(f"[{self.R.t:7.3f}] phase {ph.name}  (wall {time.time() - tic:6.1f}s)", flush=True)
            ph.start(self, self.R.t)
            while not ph.done(self, self.R.t):
                ph.update(self, self.R.t)
                self.R.step(1)
                self.share.update(self.tm, self.C.last, self.R.frc_lim, self.cfg.dt_ctrl)
                if self.R.t >= self.next_log - 1e-9:
                    self._log()
                    self.next_log += c.log_dt
                if self._check_fail():
                    if c.stop_on_fail:
                        break
                if self.R.t > c.t_max:
                    break
            if self.failed and c.stop_on_fail:
                # keep simulating a little to show the failure, without control updates of the plan
                t_end = self.R.t + 2.5
                while self.R.t < t_end:
                    try:
                        self.R.step(1)
                        self.share.update(self.tm, self.C.last, self.R.frc_lim, self.cfg.dt_ctrl)
                    except Exception as e:      # numerical breakdown of the controller model
                        self.R.d.ctrl[:] = 0.0
                        mujoco.mj_step(self.m, self.d)
                        self.R.t += self.m.opt.timestep
                    if self.R.t >= self.next_log - 1e-9:
                        self._log()
                        self.next_log += c.log_dt
                break
        self.t_wall = time.time() - tic
        return self

    def _check_fail(self):
        if self.failed:
            return True
        d = self.d
        for r in self.tm.robots:
            if d.xpos[r.pelvis][2] < 0.6:
                self.failed = (self.R.t, f"robot {r.i} fell")
        if self.tm.mode.payload and self.C.ext is not None:
            pb = d.xpos[self.R.beam]
            if np.linalg.norm(pb - self.C.ext[0]) > 0.15:
                self.failed = (self.R.t, "girder lost (pose error > 0.15 m)")
        if not np.all(np.isfinite(d.qacc)):
            self.failed = (self.R.t, "numerical")
        if self.failed and self.cfg.verbose:
            print(f"    FAILURE at {self.failed[0]:.3f}: {self.failed[1]}", flush=True)
        return self.failed is not None

    def _log(self):
        R, tm, C, d = self.R, self.tm, self.C, self.d
        last = C.last
        L = dict(t=R.t, phase=self.phase_name, qpos=d.qpos.copy(), mode=(len(tm.mode.feet), len(tm.mode.grasps)))
        L["sat"] = R.saturation()
        L["tau"] = d.actuator_force.copy()
        L["beam"] = np.r_[d.xpos[R.beam], d.xquat[R.beam]]
        if tm.mode.payload:
            pd, Rd = self.refs.pay(R.t)[:2]
            L["beam_d"] = pd.copy()
            L["beam_c"] = C.ext[0].copy()
            L["h"] = R._last["h"].copy()
            L["h_f"] = R.h_f.copy()
        s = tm.v - last["vr"]
        L["sM"] = float(np.sqrt(max(s @ tm.M @ s, 0.0)))
        # connection wrenches: measured vs predicted (lambda* of Corollary 1, with the commanded lambda_d)
        lam_pred = last["lam_r"] - tm.Mmu @ (last["f_int"] + last["fF"])
        conn = []
        for k, (kind, i, s_) in enumerate(tm.rows):
            sl = slice(6 * k, 6 * k + 6)
            if kind == "foot":
                wm = R.foot_wrench(R._last["sole"], i, s_)
                spec = share.SOLE
            else:
                wm = R.grasp_wrench(R._last["palm"], i, s_)
                spec = share.PALM
            rp = share.margin(lam_pred[sl], spec)
            rm = share.margin(wm, spec)
            conn.append((kind, i, s_, lam_pred[sl].copy(), wm.copy(), rp, rm))
        L["conn"] = conn
        if tm.ng:
            lamI_meas = tm.PI @ R._last["lam_g"]
            L["eI"] = tm.VIp @ (lamI_meas - last["lam_d"])
            L["lam_d"] = last["lam_d"].copy()
            L["lamI"] = lamI_meas
            L["z"] = C.z.copy()
            if getattr(R, "noise", None) is not None and hasattr(R, "_true"):
                # noise-free values (evaluation only; the controller uses the measured ones)
                lam_g_true = np.concatenate([R.grasp_wrench(R._true["palm"], i, s_) for (kind, i, s_) in tm.rows
                                             if kind == "grasp"])
                L["eI_true"] = tm.VIp @ (tm.PI @ lam_g_true - last["lam_d"])
                L["palm_true"] = {(i, s_): R.grasp_wrench(R._true["palm"], i, s_)
                                  for (kind, i, s_) in tm.rows if kind == "grasp"}
        L["qp"] = dict(self.share.info) if self.share.info else {}
        L["pel"] = np.array([d.xpos[r.pelvis] for r in tm.robots])
        L["pel_d"] = np.array([self.refs.pel[r.i](R.t)[0] for r in tm.robots])
        self.logs.append(L)


# ---------------------------------------------------------------------- phase objects
class Phase:
    name = "phase"

    def start(self, M, t):
        pass

    def update(self, M, t):
        pass

    def done(self, M, t):
        return True


class Hold(Phase):
    def __init__(self, T, name="hold"):
        self.T = T
        self.name = name

    def start(self, M, t):
        self.t0 = t

    def done(self, M, t):
        return t >= self.t0 + self.T


class Step(Phase):
    """All robots in `group` step foot `s` by world displacement D[i] (quasi-static single support)."""

    def __init__(self, group, s, D, loaded, name=None, pre_shift=True, back=True, ahead=True):
        self.group, self.s, self.D, self.loaded = group, s, D, loaded
        self.pre_shift, self.back, self.ahead = pre_shift, back, ahead
        self.name = name or f"step-{s}"

    def start(self, M, t):
        c = M.cfg
        self.t0 = t
        self.state = "shift"
        self.swing = [(i, self.s) for i in self.group]
        self.stance = {i: ("r" if self.s == "l" else "l") for i in self.group}
        frac = c.shift_frac_load if self.loaded else c.shift_frac_free
        if self.pre_shift:
            for i in self.group:
                tgt = M.pelvis_target(i, over=self.stance[i], frac=frac, kf="pick" if self.loaded else "walk")
                M.refs.pel[i].move(t, c.T_ss, p1=tgt)
        self.planted = {}
        self.extended = set()

    def update(self, M, t):
        c = M.cfg
        if self.state == "shift":
            Tu = c.T_ss if self.pre_shift else c.T_unload
            a = min(1.0, (t - self.t0) / (0.85 * Tu))
            cap = (1.0 - a) * 1500.0 + a * 40.0
            for k in self.swing:
                M.share.caps[k] = cap
            if t >= self.t0 + Tu:
                ok = all(M.foot_fz(*k) < 90.0 for k in self.swing)
                if ok or t >= self.t0 + Tu + 0.6:
                    M.lift_feet(self.swing)
                    for k in self.swing:
                        M.share.caps.pop(k, None)
                    self.t_lift = t
                    for (i, s) in self.swing:
                        p, Rf = M.site_pose(i, "foot", s)
                        ctr = M.pel_center[i] + (self.D[i] if self.ahead else 0.0)
                        tgt = M.foot_target(i, s, ctr, kf="pick" if self.loaded else "walk")
                        tgt[2] = 0.012
                        yaw = -M.side[i] * np.pi / 2
                        ps = M.refs.eef[(i, "foot", s)]
                        ps.move(t, c.T_swing, p1=tgt, R1=rot_z(yaw), bump=c.apex)
                        tgt2 = tgt.copy()
                        tgt2[2] = -0.008
                        ps.move(t + c.T_swing, c.T_land, p1=tgt2)
                    self.state = "swing"
        elif self.state == "swing":
            if t >= self.t_lift + c.T_swing:
                new = []
                for k in self.swing:
                    if k in self.planted:
                        continue
                    if M.foot_fz(*k) > 15.0:
                        new.append(k)
                    elif t >= self.t_lift + c.T_swing + c.T_land and k not in self.extended:
                        # no contact yet: keep descending slowly
                        M.refs.eef[(k[0], "foot", k[1])].move(t, 0.6, dp=np.array([0, 0, -0.02]))
                        self.extended.add(k)
                if new:
                    M.plant_feet(new)
                    for k in new:
                        self.planted[k] = t
                        M.share.caps[k] = 40.0
                if len(self.planted) == len(self.swing):
                    self.state = "back"
                    self.t_back = t
                    if self.back:
                        for i in self.group:
                            M.refs.pel[i].move(t + c.T_settle, c.T_back, p1=M.pel_center[i])
        if self.state == "back" or (self.state == "swing" and self.planted):
            # reload the new feet gradually
            for k, tp in self.planted.items():
                a = min(1.0, (t - tp) / c.T_reload)
                if a >= 1.0:
                    M.share.caps.pop(k, None)
                else:
                    M.share.caps[k] = 40.0 + a * 1500.0

    def done(self, M, t):
        if self.state == "back" and t >= self.t_back + (M.cfg.T_settle + M.cfg.T_back if self.back
                                                         else M.cfg.T_reload):
            for k in self.swing:
                M.share.caps.pop(k, None)
            return True
        return False


class BodyShift(Phase):
    """All pelvises (and the girder, if carried) move by D[i] in double support. With `to_side`, the pelvis
    ends shifted laterally over that foot (diagonal transfer of the support in a quasi-static gait)."""

    def __init__(self, D, D_payload=None, T=None, name="shift", to_side=None, loaded=False):
        self.D, self.Dp, self.T = D, D_payload, T
        self.name = name
        self.to_side, self.loaded = to_side, loaded

    def start(self, M, t):
        T = self.T or M.cfg.T_body
        self.t1 = t + T
        frac = M.cfg.shift_frac_load if self.loaded else M.cfg.shift_frac_free
        for i in M.pel_center:
            M.pel_center[i] = M.pel_center[i] + self.D[i]
            if self.to_side is not None:
                tgt = M.pelvis_target(i, over=self.to_side, frac=frac, kf="pick" if self.loaded else "walk")
            else:
                tgt = M.pel_center[i].copy()
            M.refs.pel[i].move(t, T, p1=tgt)
        if self.Dp is not None and M.tm.mode.payload:
            M.refs.pay.move(t, T, dp=self.Dp)

    def done(self, M, t):
        return t >= self.t1


class PreReach(Phase):
    """Crouch to the pick posture; hands to pre-grasp poses below and outside the bottom face."""
    name = "pre-reach"

    def __init__(self, T=2.6):
        self.T = T

    def start(self, M, t):
        c, kf = M.cfg, M.kf["pick"]
        self.t1 = t + self.T
        pb = M.d.xpos[M.R.beam].copy()
        for r in M.tm.robots:
            i = r.i
            mid = 0.5 * (M.site_pose(i, "foot", "l")[0] + M.site_pose(i, "foot", "r")[0])
            Ry = rot_z(-M.side[i] * np.pi / 2)
            tgt = mid + Ry @ kf[i]["pel_off"]
            M.pel_center[i] = tgt.copy()
            M.refs.pel[i].move(t, self.T, p1=tgt, R1=kf[i]["pel"][1])
            M.refs.jnt[i].move(t, self.T, kf[i]["jq"])
            for s in "lr":
                ref = M.refs.eef[(i, "hand", s)](t)
                ps = PoseSchedule(ref[0], ref[1])
                ph, Rh = kf[i]["hands"][s]
                ph = ph + (pb - M.beamA) * np.array([1, 1, 0])          # girder may sit slightly off
                pre = ph + np.array([0, M.side[i] * 0.03, -0.04])
                ps.move(t, self.T, p1=pre, R1=Rh)
                M.refs.eef[(i, "hand", s)] = ps
                M.grasp_tgt = getattr(M, "grasp_tgt", {})
                M.grasp_tgt[(i, s)] = (ph, Rh)

    def done(self, M, t):
        return t >= self.t1


class ReachIn(Phase):
    name = "reach-in"

    def __init__(self, T=1.4):
        self.T = T

    def start(self, M, t):
        self.t1 = t + self.T
        for (i, s), (ph, Rh) in M.grasp_tgt.items():
            M.refs.eef[(i, "hand", s)].move(t, self.T, p1=ph + np.array([0, 0, -0.03]), R1=Rh)

    def done(self, M, t):
        return t >= self.t1


class RaiseContact(Phase):
    """Raise the palms until every palm touches the girder (light preload), then close the grasps."""
    name = "contact"

    def __init__(self, T=1.2):
        self.T = T

    def start(self, M, t):
        self.t0 = t
        self.touch = {}
        for (i, s), (ph, Rh) in M.grasp_tgt.items():
            M.refs.eef[(i, "hand", s)].move(t, self.T, p1=ph + np.array([0, 0, 0.012]), R1=Rh)
        self.extended = False

    def done(self, M, t):
        fz = M.palm_fz()
        for k, v in fz.items():
            if v > 4.0 and k not in self.touch:
                self.touch[k] = t
        if len(self.touch) == len(fz) and t >= max(self.touch.values()) + 0.15:
            M.close_grasps()
            return True
        if t >= self.t0 + self.T and not self.extended:
            for (i, s), (ph, Rh) in M.grasp_tgt.items():
                if (i, s) not in self.touch:
                    M.refs.eef[(i, "hand", s)].move(t, 1.0, dp=np.array([0, 0, 0.02]))
            self.extended = True
        if t >= self.t0 + self.T + 2.0:
            M.close_grasps()
            M.events.append((t, "grasp closed without contact at all palms", 0, 0))
            return True
        return False


class Transfer(Phase):
    """Move the desired girder pose (external impedance) to dz relative to the current pose."""

    def __init__(self, dz, T, pelvis_dz=0.0, T_pel=None, delay_pel=0.0, extra=None, name="transfer",
                 internal=None):
        self.dz, self.T, self.pdz, self.Tp, self.delay, self.extra = dz, T, pelvis_dz, T_pel, delay_pel, extra
        self.name = name
        self.internal = internal          # switch the internal impedance on/off at the start of the phase

    def start(self, M, t):
        self.t1 = t + self.T
        if self.internal is not None:
            M.C.internal_active = self.internal
        if self.extra == "unload":
            # continue below contact until the support carries the weight (K_p (p_c - p_d) = W)
            M.refs.pay.move(t, self.T, dp=np.array([0, 0, self.dz - M.W / M.P.Kp[2, 2]]))
        elif self.extra == "load":
            # raise the desired pose to the current girder pose: the support unloads (K_p (p_d - p_c) = -h)
            M.refs.pay.move(t, self.T, p1=M.d.xpos[M.R.beam].copy() + np.array([0, 0, self.dz]))
        else:
            M.refs.pay.move(t, self.T, dp=np.array([0, 0, self.dz]))
        if self.pdz:
            Tp = self.Tp or self.T
            for i in M.pel_center:
                M.pel_center[i] = M.pel_center[i] + np.array([0, 0, self.pdz])
                M.refs.pel[i].move(t + self.delay, Tp, p1=M.pel_center[i])
            self.t1 = max(self.t1, t + self.delay + Tp)

    def done(self, M, t):
        return t >= self.t1


class Release(Phase):
    name = "release"

    def start(self, M, t):
        self.t0 = t
        self.opened = False

    def update(self, M, t):
        if not self.opened:
            hz = M.R.h_f[2]
            if hz > 0.85 * M.W or t > self.t0 + 3.0:
                M.open_grasps()
                self.opened = True
                c = M.cfg
                self.t1 = t + 0.7
                for r in M.tm.robots:
                    for s in "lr":
                        ps = M.refs.eef[(r.i, "hand", s)]
                        ps.move(t, 0.7, dp=np.array([0, 0, -0.05]))
                        ps.move(t + 0.7, 0.9, dp=np.array([0, M.side[r.i] * 0.16, 0.0]))
                self.t1 = t + 1.6

    def done(self, M, t):
        return self.opened and t >= self.t1


class Stow(Phase):
    """Hands back to the walking pose relative to the pelvis; pelvis back to walking height."""
    name = "stow"

    def __init__(self, T=1.6):
        self.T = T

    def start(self, M, t):
        kf = M.kf["walk"]
        self.t1 = t + self.T
        for r in M.tm.robots:
            i = r.i
            M.pel_center[i] = M.pel_center[i].copy()
            M.pel_center[i][2] = kf[i]["pel"][0][2]
            M.refs.pel[i].move(t, self.T, p1=M.pel_center[i], R1=kf[i]["pel"][1])
            M.refs.jnt[i].move(t, self.T, kf[i]["jq"])
            pe, Re = M.refs.pel[i].end_pose()
            for s in "lr":
                rr, Rr = kf[i]["hand_rel"][s]
                M.refs.eef[(r.i, "hand", s)].move(t, self.T, p1=pe + Re @ rr, R1=Re @ Rr)

    def done(self, M, t):
        if t >= self.t1:
            kf = M.kf["walk"]
            for r in M.tm.robots:
                for s in "lr":
                    rr, Rr = kf[r.i]["hand_rel"][s]
                    M.refs.eef[(r.i, "hand", s)] = RelPose(M.refs.pel[r.i], rr, Rr)
            return True
        return False


def walk_cycle(M, D, loaded, Dp=None, tag=""):
    """Quasi-static gait: [CoM over right foot, step left] -> [diagonal body shift by D to over the left foot]
    -> [step right, CoM back to the middle]."""
    g = list(range(M.cfg.n))
    return [Step(g, "l", D, loaded, name=f"{tag}step-l", pre_shift=True, back=False, ahead=True),
            BodyShift(D, Dp, name=f"{tag}shift", to_side="l", loaded=loaded),
            Step(g, "r", D, loaded, name=f"{tag}step-r", pre_shift=False, back=True, ahead=False)]


def full_mission(M, n_app=None, n_carry=None, retreat=1, stop_after=None):
    c = M.cfg
    n_app = n_app if n_app is not None else int(round(c.D_app / c.step))
    n_carry = n_carry if n_carry is not None else int(round(c.YB / c.step))
    ph = [Hold(0.8, "settle")]
    Dapp = {i: c.step * M.fwd[i] for i in range(c.n)}
    for k in range(n_app):
        ph += walk_cycle(M, Dapp, False, tag=f"go{k + 1}-")
    ph += [PreReach(), ReachIn(), RaiseContact(),
           Transfer(0.0, 3.5, extra="load", name="load-transfer", internal=False),
           Transfer(c.dz_lift, 2.4, pelvis_dz=c.dz_lift, name="lift", internal=True)]
    if stop_after == "lift":
        return ph + [Hold(2.0, "hold")]
    Dc = {i: c.step * np.array([0, 1.0, 0]) for i in range(c.n)}
    for k in range(n_carry):
        ph += walk_cycle(M, Dc, True, Dp=c.step * np.array([0, 1.0, 0]), tag=f"carry{k + 1}-")
    if stop_after == "carry":
        return ph + [Hold(2.0, "hold")]
    ph += [Transfer(-c.dz_lift - 0.004, 2.6, pelvis_dz=-c.dz_lift, name="place"),
           Transfer(0.0, 2.6, extra="unload", name="unload", internal=False), Release(), Stow()]
    Dret = {i: -c.step * M.fwd[i] for i in range(c.n)}
    for k in range(retreat):
        ph += walk_cycle(M, Dret, False, tag=f"back{k + 1}-")
    ph += [Hold(1.0, "done")]
    return ph
