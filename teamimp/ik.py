"""Whole-body inverse kinematics (damped least squares) for one robot of the team."""
import numpy as np
import mujoco

from . import scene
from .so3 import log, hat


def site_pose(m, d, sid):
    return d.site_xpos[sid].copy(), d.site_xmat[sid].reshape(3, 3).copy()


def solve_robot(m, q0, robot, tasks, iters=200, damping=1e-3, posture=None, w_post=0.02, tol=1e-9,
                w_com=0.0, com_xy=None):
    """tasks: list of (site_id, p_target or None, R_target or None, w_pos, w_rot_vec(3) in target frame).
    Returns q with only robot's coordinates changed."""
    d = mujoco.MjData(m)
    q = q0.copy()
    dofs = np.array(robot.dofs)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    post = posture if posture is not None else np.array([scene.STAND[j] for j in scene.JOINTS])
    jq = np.array([robot.qpos_adr[j] for j in scene.JOINTS])
    jdof = np.array([robot.dof[j] for j in scene.JOINTS])
    lo = m.jnt_range[[m.joint(f"r{robot.i}_{j}").id for j in scene.JOINTS], 0]
    hi = m.jnt_range[[m.joint(f"r{robot.i}_{j}").id for j in scene.JOINTS], 1]
    err_hist = []
    for it in range(iters):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rows, errs = [], []
        for (sid, pt, Rt, wp, wr) in tasks:
            p, R = site_pose(m, d, sid)
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            if pt is not None:
                rows.append(wp * jp[:, dofs])
                errs.append(wp * (pt - p))
            if Rt is not None:
                e = log(Rt @ R.T)                 # world-frame rotation error
                Wr = Rt @ np.diag(wr) @ Rt.T      # weights in target axes
                rows.append(Wr @ jr[:, dofs])
                errs.append(Wr @ e)
        if w_com > 0 and com_xy is not None:
            jc = np.zeros((3, m.nv))
            mujoco.mj_jacSubtreeCom(m, d, jc, robot.pelvis)
            c = d.subtree_com[robot.pelvis]
            rows.append(w_com * jc[:2, dofs])
            errs.append(w_com * (np.asarray(com_xy) - c[:2]))
        # posture regularization on joints
        Jp = np.zeros((len(jdof), len(dofs)))
        Jp[np.arange(len(jdof)), [list(dofs).index(k) for k in jdof]] = 1.0
        rows.append(w_post * Jp)
        errs.append(w_post * (post - q[jq]))
        J = np.vstack(rows)
        e = np.concatenate(errs)
        err_hist.append(np.linalg.norm(e))
        dx = np.linalg.solve(J.T @ J + damping * np.eye(J.shape[1]), J.T @ e)
        dv = np.zeros(m.nv)
        dv[dofs] = dx
        mujoco.mj_integratePos(m, q, dv, 1.0)
        q[jq] = np.clip(q[jq], lo + 1e-3, hi - 1e-3)
        if np.linalg.norm(dx) < tol:
            break
    return q, err_hist


def task_errors(m, q, tasks):
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    out = []
    for (sid, pt, Rt, wp, wr) in tasks:
        p, R = site_pose(m, d, sid)
        ep = 0.0 if pt is None else np.linalg.norm(pt - p)
        er = 0.0 if Rt is None else np.linalg.norm(np.diag(np.sign(wr)) @ Rt.T @ log(Rt @ R.T))
        out.append((ep, er))
    return out


W_WRIST_YAW = 1.5
CARRY = dict(scene.STAND)
CARRY.update(l_shoulder_aa=0.15, r_shoulder_aa=-0.15, l_shoulder_fe=-0.35, r_shoulder_fe=-0.35,
             l_elbow_fe=-1.45, r_elbow_fe=-1.45, l_wrist_pitch=0.6, r_wrist_pitch=-0.6)


def palm_target(side_sign, xh, beam_p, half_w):
    """Palm frame on the beam face. side_sign=+1: robot on +y side (palm normal -y)."""
    z = np.array([0.0, -side_sign, 0.0])           # palm normal, pointing into the beam
    y = np.array([0.0, 0.0, 1.0])                  # fingers down  <=> site y axis up
    x = np.cross(y, z)
    R = np.column_stack([x, y, z])
    p = np.array([xh, beam_p[1] + side_sign * half_w, beam_p[2]])
    return p, R


def palm_target_under(side_sign, xh, beam_p, half_w, half_h, inset=0.07, yaw=0.0):
    """Palm under the bottom face, facing up, fingers pointing towards the beam axis.
    side_sign=+1: robot on the +y side. yaw: in-plane rotation of the palm about the vertical."""
    from .so3 import rot_z
    z = np.array([0.0, 0.0, 1.0])
    y = np.array([0.0, side_sign, 0.0])            # site y: from the fingertips towards the wrist
    x = np.cross(y, z)
    R = rot_z(yaw) @ np.column_stack([x, y, z])
    p = np.array([xh, beam_p[1] + side_sign * (half_w - inset), beam_p[2] - half_h])
    return p, R


def team_config(m, tm, beam_p, layout, q_init=None, hand_dx=0.2, pelvis_z=None, iters=300, beam_half_w=0.15,
                foot_dy=0.153, hands=True, foot_back=0.0, posture=None, w_com=0.0, torso_pitch=None, pelvis_dx=0.0,
                grasp="side", beam_half_h=0.2, inset=0.07, w_yaw=0.0, pelvis_dy=0.0, feet_override=None,
                com_dx=0.0):
    """IK for all robots: soles flat at their station, palms on the girder faces (if hands).
    grasp="side": palms on the vertical faces at mid height; "under": palms up under the bottom face."""
    from .so3 import rot_z
    q = (q_init if q_init is not None else m.qpos0).copy()
    q[tm.beam_qadr:tm.beam_qadr + 3] = beam_p
    q[tm.beam_qadr + 3:tm.beam_qadr + 7] = [1, 0, 0, 0]
    info = []
    for i, r in enumerate(tm.robots):
        x, y, yaw = layout.base_xy_yaw(i, beam_p[:2], back=foot_back)
        side = layout.stations[i][1]
        Ry = rot_z(yaw)
        if q_init is None:
            q[r.qadr:r.qadr + 3] = [x, y, 1.0]
            q[r.qadr + 3:r.qadr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            for jn in scene.JOINTS:
                q[r.qpos_adr[jn]] = (posture or CARRY)[jn] if isinstance(posture or CARRY, dict) else scene.STAND[jn]
        tasks = []
        feet_c = []
        for s, sg in (("l", 1), ("r", -1)):
            pf = np.array([x, y, 0.0]) + Ry @ np.array([-0.015, sg * foot_dy, 0.0])
            if feet_override is not None and (i, s) in feet_override:
                pf = np.asarray(feet_override[(i, s)], float)
            feet_c.append(pf)
            tasks.append((r.site[("foot", s)], pf, Ry, 30.0, np.array([10.0, 10.0, 10.0])))
        if hands:
            for s, sg in (("l", 1), ("r", -1)):
                # robot's left is +x for side +1 (yaw -90deg) and -x for side -1
                xh = x + sg * (Ry @ np.array([0, 1.0, 0]))[0] * hand_dx
                if grasp == "under":
                    pt, Rt = palm_target_under(side, xh, beam_p, beam_half_w, beam_half_h, inset)
                    tasks.append((r.site[("hand", s)], pt, Rt, 30.0, np.array([10.0, 10.0, w_yaw])))
                else:
                    pt, Rt = palm_target(side, xh, beam_p, beam_half_w)
                    tasks.append((r.site[("hand", s)], pt, Rt, 30.0, np.array([10.0, 10.0, 0.0])))
        sid_p = m.site(f"r{i}_pelvis").id
        Rp = Ry
        pdy = pelvis_dy[i] if isinstance(pelvis_dy, (list, tuple, np.ndarray)) else pelvis_dy
        pc = np.mean(feet_c, axis=0) + Ry @ np.array([pelvis_dx, pdy, 0.0])
        tgt = np.array([pc[0], pc[1], np.nan if pelvis_z is None else pelvis_z])
        tasks.append((sid_p, tgt, Rp, 5.0, np.array([3.0, 3.0, 3.0])))
        com_xy = np.mean(feet_c, axis=0)[:2] + (Ry @ np.array([com_dx, 0.0, 0.0]))[:2]
        pv = posture or CARRY
        post = np.array([pv[j] for j in scene.JOINTS]) if isinstance(pv, dict) else pv
        qq, hist = _solve_with_partial(m, q, r, tasks, iters, post, com_xy, w_com)
        q = qq
        info.append((hist[-1] if hist else None))
    return q, info


def _solve_with_partial(m, q, r, tasks, iters, posture, com_xy, w_com):
    """Handle NaN components of a position target (free directions)."""
    d = mujoco.MjData(m)
    fixed = []
    for (sid, pt, Rt, wp, wr) in tasks:
        if pt is not None and np.any(np.isnan(pt)):
            fixed.append((sid, pt, Rt, wp, wr, ~np.isnan(pt)))
        else:
            fixed.append((sid, pt, Rt, wp, wr, None))
    dofs = np.array(r.dofs)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    jc = np.zeros((3, m.nv))
    post = posture if posture is not None else np.array([scene.STAND[j] for j in scene.JOINTS])
    jq = np.array([r.qpos_adr[j] for j in scene.JOINTS])
    jdof = [list(dofs).index(r.dof[j]) for j in scene.JOINTS]
    ids = [m.joint(f"r{r.i}_{j}").id for j in scene.JOINTS]
    lo, hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
    hist = []
    for it in range(iters):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rows, errs = [], []
        for (sid, pt, Rt, wp, wr, mask) in fixed:
            p, R = site_pose(m, d, sid)
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            if pt is not None and wp > 0:
                if mask is None:
                    rows.append(wp * jp[:, dofs])
                    errs.append(wp * (pt - p))
                else:
                    rows.append(wp * jp[mask][:, dofs])
                    errs.append(wp * (pt[mask] - p[mask]))
            if Rt is not None:
                e = log(Rt @ R.T)
                Wr = Rt @ np.diag(wr) @ Rt.T
                rows.append(Wr @ jr[:, dofs])
                errs.append(Wr @ e)
        if w_com > 0:
            mujoco.mj_jacSubtreeCom(m, d, jc, r.pelvis)
            c = d.subtree_com[r.pelvis]
            rows.append(w_com * 10 * jc[:2, dofs])
            errs.append(w_com * 10 * (com_xy - c[:2]))
        Jp = np.zeros((len(jdof), len(dofs)))
        Jp[np.arange(len(jdof)), jdof] = 1.0
        wj = np.array([1.5 if (j in scene.TORSO or j in scene.NECK) else (0.5 if j in scene.ARM_RED.values()
                       else (0.25 if ("elbow" in j or "shoulder_fe" in j) else (W_WRIST_YAW if "wrist_yaw" in j
                       else 0.05))) for j in scene.JOINTS])
        rows.append(wj[:, None] * Jp)
        errs.append(wj * (post - q[jq]))
        # joint-limit barrier (soft): push away when within 0.1 rad of a limit
        margin_lo = q[jq] - lo
        margin_hi = hi - q[jq]
        wb = np.where((margin_lo < 0.12) | (margin_hi < 0.12), 2.0, 0.0)
        if wb.any():
            mid = 0.5 * (lo + hi)
            rows.append(wb[:, None] * Jp)
            errs.append(wb * np.clip(mid - q[jq], -0.05, 0.05))
        J = np.vstack(rows)
        e = np.concatenate(errs)
        hist.append(np.linalg.norm(e))
        dx = np.linalg.solve(J.T @ J + 1e-4 * np.eye(J.shape[1]), J.T @ e)
        step = min(1.0, 0.2 / max(np.abs(dx).max(), 1e-12))
        dv = np.zeros(m.nv)
        dv[dofs] = step * dx
        mujoco.mj_integratePos(m, q, dv, 1.0)
        q[jq] = np.clip(q[jq], lo + 0.02, hi - 0.02)
        if np.linalg.norm(step * dx) < 1e-10:
            break
    return q, hist
