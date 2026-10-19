"""Scene construction: Apptronik Apollo humanoids, a steel box girder, and a construction site.

Two variants are built from the same robot description:
  * kind="ctrl": robots + payload only, no contacts (model used by the controller and the exact plant);
  * kind="plant": adds the site, contact geometry, actuators with torque limits.
"""
import os
import numpy as np
import mujoco

from . import textures

APOLLO_XML = os.environ.get(
    "APOLLO_XML", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "apollo", "apptronik_apollo.xml"))

# Actuated joints of one robot, in qvel order (after the 6 free-joint dofs).
JOINTS = [
    "torso_yaw", "torso_roll", "torso_pitch", "neck_yaw", "neck_roll", "neck_pitch",
    "l_shoulder_aa", "l_shoulder_ie", "l_shoulder_fe", "l_elbow_fe", "l_wrist_roll", "l_wrist_yaw", "l_wrist_pitch",
    "r_shoulder_aa", "r_shoulder_ie", "r_shoulder_fe", "r_elbow_fe", "r_wrist_roll", "r_wrist_yaw", "r_wrist_pitch",
    "l_hip_ie", "l_hip_aa", "l_hip_fe", "l_knee_fe", "l_ankle_ie", "l_ankle_pd",
    "r_hip_ie", "r_hip_aa", "r_hip_fe", "r_knee_fe", "r_ankle_ie", "r_ankle_pd",
]
TORSO = ["torso_yaw", "torso_roll", "torso_pitch"]
NECK = ["neck_yaw", "neck_roll", "neck_pitch"]
ARM = {s: [f"{s}_shoulder_aa", f"{s}_shoulder_ie", f"{s}_shoulder_fe", f"{s}_elbow_fe",
           f"{s}_wrist_roll", f"{s}_wrist_yaw", f"{s}_wrist_pitch"] for s in "lr"}
ARM_RED = {s: f"{s}_shoulder_ie" for s in "lr"}          # redundancy coordinate of each arm
LEG = {s: [f"{s}_hip_ie", f"{s}_hip_aa", f"{s}_hip_fe", f"{s}_knee_fe", f"{s}_ankle_ie", f"{s}_ankle_pd"]
       for s in "lr"}
SOLE_HALF = (0.100, 0.0425)      # rectangular sole, half sizes (m)
PALM_HALF = (0.041, 0.083)       # palm plate, half sizes (m)
STAND = dict(torso_yaw=0, torso_roll=0, torso_pitch=0, neck_yaw=0, neck_roll=0, neck_pitch=0,
             l_shoulder_aa=0.2, l_shoulder_ie=0, l_shoulder_fe=0.1, l_elbow_fe=-0.4,
             l_wrist_roll=0, l_wrist_yaw=0, l_wrist_pitch=0,
             r_shoulder_aa=-0.2, r_shoulder_ie=0, r_shoulder_fe=0.1, r_elbow_fe=-0.4,
             r_wrist_roll=0, r_wrist_yaw=0, r_wrist_pitch=0,
             l_hip_ie=0.08, l_hip_aa=0.1, l_hip_fe=-0.477, l_knee_fe=1.033, l_ankle_ie=-0.03, l_ankle_pd=-0.58,
             r_hip_ie=-0.08, r_hip_aa=-0.1, r_hip_fe=-0.477, r_knee_fe=1.033, r_ankle_ie=0.03, r_ankle_pd=-0.58)

SHADOW_SIZE = 4096
MSAA = 4

# collision bit masks
B_GROUND, B_FOOT, B_PALM, B_LOAD = 1, 2, 4, 8


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def robot_spec(kind, mu_foot=0.8, mu_palm=0.7):
    """Apollo spec with sole/palm sites; contacts only on soles and palm plates (plant)."""
    c = mujoco.MjSpec.from_file(APOLLO_XML)
    c.option.timestep = 0.001
    c.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    for a in list(c.actuators):
        c.delete(a)
    for k in list(c.keys):
        c.delete(k)
    for b in list(c.worldbody.bodies):
        if b.name == "world_link":
            c.delete(b)
    for g in c.geoms:
        g.contype = 0
        g.conaffinity = 0
    if kind == "ctrl":
        # the controller model needs inertia and kinematics only (explicit <inertial> elements)
        for g in list(c.geoms):
            c.delete(g)
        for msh in list(c.meshes):
            c.delete(msh)
    c.body("base_link").add_site(name="pelvis", size=[0.01, 0, 0], group=5)
    c.body("torso_link").add_site(name="chest", pos=[0.0, 0, 0.30], size=[0.01, 0, 0], group=5)
    for s in "lr":
        sole_quat = np.array([0.987443, -0.0797175, -0.132426, -0.0326442])
        sole_pos = np.array([0.0646931, -0.00550529 if s == "l" else 0.00550529, -0.03843])
        if s == "r":
            sole_quat = np.array([0.987443, 0.0797175, -0.132426, 0.0326442])
        R = _quat_to_mat(sole_quat)
        pos = sole_pos + R @ np.array([0, 0, -0.009])
        c.body(f"{s}_foot_link").add_site(name=f"{s}_sole", pos=pos, quat=sole_quat, size=[0.01, 0, 0],
                                          group=5)
        sgn = -1.0 if s == "l" else 1.0
        # palm frame: z = palm normal (out of the palm), y = from the fingertips towards the wrist
        q = np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0]) if s == "l" else np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        c.body(f"{s}_wrist_pitch_link").add_site(name=f"{s}_palm", pos=[0.0075, sgn * 0.045, -0.112], quat=q,
                                                 size=[0.01, 0, 0], group=5)
        if kind == "plant":
            sole = c.geom(f"collision_{s}_sole")
            sole.contype, sole.conaffinity = B_FOOT, B_GROUND
            sole.condim = 4
            sole.friction = [mu_foot, 0.02, 0.0001]
            sole.solref = [0.008, 1.0]
            plate = c.geom(f"collision_{s}_hand_plate")
            plate.contype, plate.conaffinity = B_PALM, B_LOAD
            plate.condim = 4
            plate.friction = [mu_palm, 0.02, 0.0001]
            plate.solref = [0.006, 1.0]
            plate.margin = 0.002
            plate.gap = 0.002
    return c


class Layout:
    """Robot stations around the girder. Robots on the +y side face -y and vice versa."""

    def __init__(self, n, x_sp=0.8, y_off=0.58):
        if n == 4:
            st = [(-x_sp / 2, +1), (x_sp / 2, +1), (-x_sp / 2, -1), (x_sp / 2, -1)]
        elif n == 3:
            st = [(-x_sp / 2, +1), (x_sp / 2, +1), (0.0, -1)]
        elif n == 2:
            st = [(0.0, +1), (0.0, -1)]
        else:
            raise ValueError(n)
        self.n = n
        self.stations = st
        self.y_off = y_off

    def base_xy_yaw(self, i, beam_xy=(0.0, 0.0), back=0.0):
        x, side = self.stations[i]
        return (beam_xy[0] + x, beam_xy[1] + side * (self.y_off + back), -side * np.pi / 2)


def beam_inertia(L, W, H, t, mass):
    """Thin-walled rectangular hollow section with end plates, scaled to `mass`."""
    def box(a, b, c):
        v = a * b * c
        return v, v / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    vo, io = box(L, W, H)
    vi, ii = box(L, W - 2 * t, H - 2 * t)
    vol = vo - vi
    I = io - ii
    return mass * I / vol


def add_payload(s, mass, pos, size=(1.2, 0.15, 0.20), visual=True):
    L, W, H = 2 * size[0], 2 * size[1], 2 * size[2]
    b = s.worldbody.add_body(name="beam", pos=pos)
    b.add_freejoint(name="beam_free")
    b.mass = mass
    b.inertia = beam_inertia(L, W, H, 0.005, mass)
    b.ipos = [0, 0, 0]
    b.explicitinertial = True
    g = b.add_geom(name="beam_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), mass=0,
                   contype=B_LOAD, conaffinity=B_GROUND | B_PALM, condim=4, friction=[0.6, 0.02, 0.0001],
                   solref=[0.006, 1.0])
    if visual:
        g.material = "primer"
        # top label and darker end plates
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.55, size[1] - 0.004, 0.0006],
                   pos=[0, 0, size[2] + 0.0005], material="label", contype=0, conaffinity=0, mass=0)
        for sx in (-1, 1):
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.006, size[1] + 0.012, size[2] + 0.012],
                       pos=[sx * (size[0] + 0.006), 0, 0], material="steel_dark", contype=0, conaffinity=0, mass=0)
            for sy in (-1, 1):
                for sz in (-1, 1):
                    b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.004],
                               pos=[sx * (size[0] + 0.014), sy * 0.08, sz * 0.12],
                               quat=[np.sqrt(0.5), 0, np.sqrt(0.5), 0], material="bolt",
                               contype=0, conaffinity=0, mass=0)
        # lifting lugs on top (visual)
        for sx in (-0.9, 0.9):
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.04, 0.008, 0.035],
                       pos=[sx, 0, size[2] + 0.035], material="primer", contype=0, conaffinity=0, mass=0)
    b.add_site(name="beam_center", size=[0.02, 0, 0], group=5)
    return b



# Hot-rolled steel wide-flange profiles (EN 10365): h, b, tw, tf (m), mass per length (kg/m). "AL 280": aluminium
# I-beam with the cross-section of an HEA 280 (mass per length scaled by the densities 2700/7850 kg/m^3).
PROFILES = {
    "HEA 240": (0.230, 0.240, 0.0075, 0.012, 60.3),
    "HEA 260": (0.250, 0.260, 0.0075, 0.0125, 68.2),
    "HEA 280": (0.270, 0.280, 0.008, 0.013, 76.4),
    "HEA 300": (0.290, 0.300, 0.0085, 0.014, 88.3),
    "AL 280": (0.270, 0.280, 0.008, 0.013, 76.4 * 2700.0 / 7850.0),
}


def girder_material(profile):
    """Visual material of the girder: aluminium for the "AL" profiles, red-oxide primer for steel."""
    return "aluminium" if profile.startswith("AL") else "steel"


def ibeam_inertia(L, h, b, tw, tf, mass):
    """Principal inertia (about the centroid) of an I-section bar of length L along x, scaled to `mass`."""
    parts = [((L, b, tf), (0, 0, h / 2 - tf / 2)), ((L, b, tf), (0, 0, -h / 2 + tf / 2)),
             ((L, tw, h - 2 * tf), (0, 0, 0))]
    vol = sum(a * bb * c for (a, bb, c), _ in parts)
    I = np.zeros(3)
    for (a, bb, c), (x, y, z) in parts:
        m = mass * a * bb * c / vol
        I += m / 12 * np.array([bb * bb + c * c, a * a + c * c, a * a + bb * bb])
        I += m * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
    return I


def add_payload_ibeam(s, pos, profile="AL 280", L=2.4, mass=None, visual=True, label=None):
    """Wide-flange beam (steel or aluminium) lying with its web vertical, axis along x. Body origin at the
    centroid."""
    h, b, tw, tf, mpl = PROFILES[profile]
    mass = mpl * L if mass is None else mass
    bd = s.worldbody.add_body(name="beam", pos=pos)
    bd.add_freejoint(name="beam_free")
    bd.mass = mass
    bd.inertia = ibeam_inertia(L, h, b, tw, tf, mass)
    bd.ipos = [0, 0, 0]
    bd.explicitinertial = True
    col = dict(contype=B_LOAD, conaffinity=B_GROUND | B_PALM, condim=4, friction=[0.6, 0.02, 0.0001],
               solref=[0.004, 1.0], mass=0)
    gmat = "alu" if girder_material(profile) == "aluminium" else "primer"
    mat = dict(material=gmat) if visual else {}
    bd.add_geom(name="beam_bot", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, b / 2, tf / 2],
                pos=[0, 0, -h / 2 + tf / 2], **col, **mat)
    bd.add_geom(name="beam_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, b / 2, tf / 2],
                pos=[0, 0, h / 2 - tf / 2], **col, **mat)
    bd.add_geom(name="beam_web", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, tw / 2, h / 2 - tf],
                pos=[0, 0, 0], **col, **mat)
    if visual:
        # stencil on both faces of the web, fillets (visual), end cuts
        for sy in (-1, 1):
            bd.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.42, 0.0004, (h / 2 - tf) * 0.62],
                        pos=[0.25 * sy, sy * (tw / 2 + 0.0004), 0], material="label", contype=0, conaffinity=0,
                        mass=0, quat=[1, 0, 0, 0] if sy > 0 else [0, 0, 0, 1])
            for sz in (-1, 1):
                bd.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2 - 0.0005, 0.006, 0.006],
                            pos=[0, sy * (tw / 2 + 0.004), sz * (h / 2 - tf - 0.004)],
                            quat=[np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0], material=gmat,
                            contype=0, conaffinity=0, mass=0)
    bd.add_site(name="beam_center", size=[0.02, 0, 0], group=5)
    return bd, dict(L=L, h=h, b=b, tw=tw, tf=tf, mass=mass, profile=profile)


ENV_TEMPLATE = """
<mujoco model="construction_site">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="20" iterations="60"
          ls_iterations="20" noslip_iterations="4"/>
  <statistic extent="3.0" center="0 0.5 0.8"/>
  <visual>
    <global offwidth="1920" offheight="1080" azimuth="140" elevation="-18" fovy="40"/>
    <quality shadowsize="{SHADOW}" offsamples="{MSAA}"/>
    <headlight ambient="0.30 0.30 0.32" diffuse="0.22 0.22 0.22" specular="0.05 0.05 0.05"/>
    <map znear="0.01" zfar="50" fogstart="6" fogend="40" shadowclip="2.6" shadowscale="0.6"/>
    <rgba haze="0.80 0.85 0.90 1" fog="0.78 0.83 0.88 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.40 0.60 0.82" rgb2="0.86 0.90 0.94"
             width="512" height="3072"/>
    <texture name="ground" type="2d" file="{T}/ground.png"/>
    <texture name="concrete" type="2d" file="{T}/concrete.png"/>
    <texture name="wood" type="2d" file="{T}/wood.png"/>
    <texture name="primer" type="2d" file="{T}/steel_primer.png"/>
    <texture name="alu" type="2d" file="{T}/aluminium.png"/>
    <texture name="label" type="2d" file="{T}/{LABEL}"/>
    <texture name="container" type="2d" file="{T}/container.png"/>
    <texture name="hazard" type="2d" file="{T}/hazard.png"/>
    <texture name="cone" type="2d" file="{T}/cone.png"/>
    <material name="ground" texture="ground" texrepeat="9 9" texuniform="true" specular="0.05" shininess="0.05"/>
    <material name="concrete" texture="concrete" texrepeat="1 1" texuniform="true" specular="0.08"/>
    <material name="wood" texture="wood" texrepeat="1 1" texuniform="true" specular="0.05"/>
    <material name="primer" texture="primer" texrepeat="2 2" texuniform="true" specular="0.25" shininess="0.35"/>
    <material name="alu" texture="alu" texrepeat="2 2" texuniform="true" specular="0.55" shininess="0.6"
              reflectance="0.04"/>
    <material name="label" texture="label" texrepeat="1 1" specular="0.2" shininess="0.3"/>
    <material name="steel_dark" rgba="0.30 0.12 0.08 1" specular="0.3" shininess="0.4"/>
    <material name="bolt" rgba="0.55 0.55 0.55 1" specular="0.6" shininess="0.8"/>
    <material name="yellow_steel" rgba="0.93 0.70 0.10 1" specular="0.3" shininess="0.4"/>
    <material name="grey_steel" rgba="0.52 0.54 0.56 1" specular="0.45" shininess="0.6" reflectance="0.05"/>
    <material name="rubber" rgba="0.08 0.08 0.08 1" specular="0.05"/>
    <material name="container" texture="container" texrepeat="1 1" texuniform="true" specular="0.3"/>
    <material name="hazard" texture="hazard" texrepeat="1 1" texuniform="true"/>
    <material name="cone" texture="cone" specular="0.2"/>
    <material name="white" rgba="0.92 0.92 0.90 1"/>
    <material name="glass" rgba="0.35 0.45 0.55 0.7" specular="0.9" shininess="0.9"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="2.2 -1.2 4.9" dir="-0.45 0.35 -0.82" diffuse="1.0 0.93 0.82"
           specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="fill" directional="true" pos="0 0 30" dir="0.5 -0.4 -0.6" diffuse="0.22 0.24 0.28"
           specular="0 0 0" castshadow="false"/>
    <geom name="ground" type="plane" size="60 60 0.1" material="ground" contype="{BG}" conaffinity="{BCG}"
          condim="3" friction="0.8 0.02 0.0001" solref="0.008 1" solimp="0.9 0.97 0.002"/>
    {PROPS}
  </worldbody>
</mujoco>
"""


def _box(name, pos, size, mat, coll=False, quat=None, extra=""):
    q = "" if quat is None else 'quat="%g %g %g %g"' % tuple(quat)
    c = 'contype="%d" conaffinity="%d" condim="4" friction="0.7 0.02 0.0001" solref="0.006 1"' % (
        B_GROUND, B_FOOT | B_LOAD) if coll else 'contype="0" conaffinity="0"'
    nm = f'name="{name}"' if name else ""
    return (f'<geom {nm} type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
            f'size="{size[0]:.4f} {size[1]:.4f} {size[2]:.4f}" material="{mat}" {q} {c} {extra}/>')


def _cyl(pos, r, h, mat, quat=None):
    q = "" if quat is None else 'quat="%g %g %g %g"' % tuple(quat)
    return (f'<geom type="cylinder" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" size="{r:.4f} {h:.4f}" '
            f'material="{mat}" {q} contype="0" conaffinity="0"/>')


def _ibeam(x, y, z0, h, mat, flange=0.16, web=0.012, depth=0.18, vertical=True, length=None, axis="x"):
    """I-section column (vertical) or beam (horizontal along `axis`), as three boxes."""
    out = []
    if vertical:
        hz = h / 2
        out.append(_box("", (x, y, z0 + hz), (web / 2, depth / 2 - 0.012, hz), mat))
        for s in (-1, 1):
            out.append(_box("", (x + s * (0 if True else 0), y + s * (depth / 2 - 0.006), z0 + hz),
                            (flange / 2, 0.006, hz), mat))
    else:
        L = length / 2
        if axis == "x":
            out.append(_box("", (x, y, z0), (L, web / 2, depth / 2 - 0.012), mat))
            for s in (-1, 1):
                out.append(_box("", (x, y, z0 + s * (depth / 2 - 0.006)), (L, flange / 2, 0.006), mat))
        else:
            out.append(_box("", (x, y, z0), (web / 2, L, depth / 2 - 0.012), mat))
            for s in (-1, 1):
                out.append(_box("", (x, y, z0 + s * (depth / 2 - 0.006)), (flange / 2, L, 0.006), mat))
    return out


def _trestle(name, x, y, top, pad="rubber", half_y=0.26):
    """Steel A-frame trestle; its top bar runs along y, a pad on top carries the load (collision)."""
    p = []
    bar_t = 0.05
    zt = top - 0.03                                      # top of the steel bar (pad 0.03 thick above it)
    p.append(_box(f"{name}_bar", (x, y, zt - bar_t / 2), (0.05, half_y, bar_t / 2), "yellow_steel"))
    hleg = zt - bar_t
    for sy in (-1, 1):
        for sx in (-1, 1):
            # splayed legs (A-frame in the y-z plane, slight splay in x)
            yb, yt = y + sy * (half_y + 0.06), y + sy * (half_y - 0.04)
            xb, xt = x + sx * 0.09, x + sx * 0.035
            c = np.array([(xb + xt) / 2, (yb + yt) / 2, hleg / 2])
            dv = np.array([xt - xb, yt - yb, hleg])
            ln = np.linalg.norm(dv)
            zax = dv / ln
            ax = np.cross([0, 0, 1.0], zax)
            ang = np.arccos(np.clip(zax[2], -1, 1))
            if np.linalg.norm(ax) > 1e-9:
                ax = ax / np.linalg.norm(ax)
                q = (np.cos(ang / 2), *(np.sin(ang / 2) * ax))
            else:
                q = (1, 0, 0, 0)
            p.append(_box("", tuple(c), (0.02, 0.02, ln / 2), "yellow_steel", quat=q))
        # cross brace
        p.append(_box("", (x, y + sy * (half_y + 0.01), hleg * 0.35), (0.075, 0.012, 0.012), "yellow_steel"))
    if pad == "rubber":
        p.append(f'<geom name="{name}_pad" type="box" pos="{x:.3f} {y:.3f} {top - 0.015:.4f}" '
                 f'size="0.07 {half_y - 0.01:.3f} 0.015" material="rubber" contype="{B_GROUND}" '
                 f'conaffinity="{B_LOAD}" condim="4" friction="0.9 0.02 0.0001" solref="0.02 1.0" '
                 f'solimp="0.9 0.95 0.003"/>')
    else:
        p.append(f'<geom name="{name}_pad" type="box" pos="{x:.3f} {y:.3f} {top - 0.015:.4f}" '
                 f'size="0.07 {half_y - 0.01:.3f} 0.015" material="wood" contype="{B_GROUND}" '
                 f'conaffinity="{B_LOAD}" condim="4" friction="0.7 0.02 0.0001" solref="0.01 1.0" '
                 f'solimp="0.9 0.95 0.002"/>')
    return p


def env_props(supports=None, background=True):
    """supports: list of (name, x, y, top, pad). Background: site props."""
    p = []
    for (nm, x, y, top, pad) in (supports or []):
        p += _trestle(nm, x, y, top, pad)
    if not background:
        return "\n    ".join(p)
    # background: steel frame under construction on a concrete slab
    p.append(_box("", (0.0, -7.5, 0.15), (7.0, 3.5, 0.15), "concrete"))
    for ix in range(5):
        for iy in range(2):
            x, y = -6 + 3 * ix, -10 + 4.5 * iy
            p += _ibeam(x, y, 0.3, 7.2, "primer")
    for lev in (3.6, 7.5):
        for iy in range(2):
            p += _ibeam(0, -10 + 4.5 * iy, lev, 0, "primer", vertical=False, length=12.4, axis="x", depth=0.3)
        for ix in range(5):
            p += _ibeam(-6 + 3 * ix, -7.75, lev, 0, "primer", vertical=False, length=4.5, axis="y", depth=0.25)
    # stack of steel beams on timber at the side
    for j in range(3):
        p.append(_box("", (-4.6, 1.4 + 0.5 * j, 0.06), (0.08, 1.4, 0.06), "wood"))
    for lay in range(2):
        for j in range(4 - lay):
            yy = 1.4 + 0.30 * (j - 1.5 + 0.5 * lay)
            z0 = 0.12 + 0.27 * lay
            p += _ibeam(-4.6, yy, z0 + 0.135, 0, "primer", vertical=False, length=3.0, axis="x", depth=0.27,
                        flange=0.28)
    # site container / office
    p.append(_box("", (6.2, 5.2, 1.3), (3.0, 1.2, 1.3), "container"))
    p.append(_box("", (6.2, 3.99, 1.5), (0.6, 0.01, 0.4), "glass"))
    # jersey barriers along the access road
    for j in range(6):
        p.append(_box("", (-7.0 + 2.1 * j, 7.5, 0.4), (1.0, 0.2, 0.4), "concrete"))
        p.append(_box("", (-7.0 + 2.1 * j, 7.29, 0.62), (1.0, 0.01, 0.06), "hazard"))
    # pallets with blocks
    for j in range(2):
        p.append(_box("", (3.9 + 1.3 * j, -2.9, 0.06), (0.55, 0.45, 0.06), "wood"))
        p.append(_box("", (3.9 + 1.3 * j, -2.9, 0.42), (0.5, 0.4, 0.3), "concrete"))
    # tower crane in the far background (mast lattice approximated by four chords + bracing)
    cx, cy, H = 11.0, -16.0, 24.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            p.append(_box("", (cx + sx * 0.8, cy + sy * 0.8, H / 2), (0.06, 0.06, H / 2), "yellow_steel"))
    for k in range(16):
        z = 1.0 + k * 1.45
        for sy in (-1, 1):
            p.append(_box("", (cx, cy + sy * 0.8, z), (0.8, 0.03, 0.03), "yellow_steel",
                          quat=(np.cos(0.3), 0, np.sin(0.3), 0)))
        for sx in (-1, 1):
            p.append(_box("", (cx + sx * 0.8, cy, z + 0.7), (0.03, 0.8, 0.03), "yellow_steel",
                          quat=(np.cos(0.3), np.sin(0.3), 0, 0)))
    p.append(_box("", (cx - 12.0, cy, H + 0.6), (16.0, 0.55, 0.55), "yellow_steel"))
    p.append(_box("", (cx + 7.0, cy, H + 0.4), (1.6, 0.9, 0.9), "concrete"))
    p.append(_box("", (cx, cy, H + 3.0), (0.25, 0.25, 2.4), "yellow_steel"))
    p.append(_box("", (cx - 0.5, cy + 1.0, H - 1.0), (0.9, 0.7, 0.8), "glass"))
    # traffic cones and a fence line
    for j, (x, y) in enumerate([(2.7, 2.9), (2.7, -1.6), (-2.7, 3.0), (-2.7, -1.8), (3.3, 0.6), (-3.3, 0.6)]):
        p.append(f'<geom type="mesh" mesh="cone" pos="{x} {y} 0" material="cone" contype="0" conaffinity="0"/>')
        p.append(_box("", (x, y, 0.012), (0.2, 0.2, 0.012), "rubber"))
    for j in range(9):
        x = -8.0 + 2.0 * j
        p.append(_cyl((x, 9.0, 0.6), 0.03, 0.6, "grey_steel"))
        p.append(_box("", (x + 1.0, 9.0, 1.0), (1.0, 0.006, 0.05), "hazard"))
    return "\n    ".join(p)


def _cone_mesh_xml():
    n = 24
    v = [0, 0, 0.75]
    for k in range(n):
        a = 2 * np.pi * k / n
        v += [0.16 * np.cos(a), 0.16 * np.sin(a), 0.02]
    for k in range(n):
        a = 2 * np.pi * k / n
        v += [0.045 * np.cos(a), 0.045 * np.sin(a), 0.70]
    verts = " ".join(f"{x:.4f}" for x in v)
    return f'<mesh name="cone" vertex="{verts}"/>'


def build_spec(n, kind="plant", layout=None, beam_pos=(0.0, 0.0, 0.75), robot_xy_yaw=None,
               with_env=True, mu_foot=0.8, mu_palm=0.7, rate_gains=None, supports=None, profile="AL 280",
               beam_len=2.4, mass=None, background=True, label=None):
    """Return an MjSpec of the team. robot_xy_yaw: list of (x, y, yaw) of each pelvis (z from stand)."""
    h, b, tw, tf, mpl = PROFILES[profile]
    mass = mpl * beam_len if mass is None else mass
    label = label or f"{profile}   {mass:.0f} kg"
    material = girder_material(profile)
    textures.make_all(label=label, material=material)
    if kind == "plant" and with_env:
        xml = ENV_TEMPLATE.replace("{SHADOW}", str(SHADOW_SIZE)).replace("{MSAA}", str(MSAA))
        xml = xml.replace("{LABEL}", textures.label_name(label, material))
        xml = xml.replace("{T}", textures.TEX_DIR).replace("{BG}", str(B_GROUND)).replace(
            "{BCG}", str(B_FOOT | B_LOAD)).replace("{PROPS}", env_props(supports, background))
        xml = xml.replace("</asset>", "  " + _cone_mesh_xml() + "\n  </asset>")
        s = mujoco.MjSpec.from_string(xml)
    else:
        s = mujoco.MjSpec()
        s.option.timestep = 0.001
        s.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        s.visual.global_.offwidth = 1920
        s.visual.global_.offheight = 1080
        if kind == "plant":
            s.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.1],
                                 contype=B_GROUND, conaffinity=B_FOOT | B_LOAD, friction=[0.8, 0.02, 0.0001])
            s.worldbody.add_light(type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[0, 0, 10],
                                  dir=[-0.4, 0.3, -0.85], castshadow=True)
    if kind == "ctrl":
        s.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    layout = layout or Layout(n)
    if robot_xy_yaw is None:
        robot_xy_yaw = [layout.base_xy_yaw(i, beam_pos[:2]) for i in range(n)]
    for i, (x, y, yaw) in enumerate(robot_xy_yaw):
        c = robot_spec(kind, mu_foot=mu_foot, mu_palm=mu_palm)
        fr = s.worldbody.add_frame(pos=[x, y, 0.0], quat=yaw_quat(yaw))
        fr.attach_body(c.body("base_link"), f"r{i}_", "")
    add_payload_ibeam(s, list(beam_pos), profile=profile, L=beam_len, mass=mass,
                      visual=(kind == "plant" and with_env))
    if kind == "plant":
        for i in range(n):
            for jn in JOINTS:
                j = s.joint(f"r{i}_{jn}")
                lim = j.actfrcrange[1]
                k = 0.0 if rate_gains is None else rate_gains[jn]
                a = s.add_actuator(name=f"r{i}_{jn}", target=f"r{i}_{jn}",
                                   trntype=mujoco.mjtTrn.mjTRN_JOINT)
                a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                a.gainprm[0] = 1.0
                a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                a.biasprm[2] = -k
                a.forcelimited = 1
                a.forcerange = [-lim, lim]
                a.ctrllimited = 0
    return s


def build(n, kind="plant", **kw):
    s = build_spec(n, kind, **kw)
    m = s.compile()
    return s, m


def robot_qpos_stand(m, i, xy_yaw, z=1.016):
    """qpos slice values for robot i standing at (x, y, yaw)."""
    out = {}
    x, y, yaw = xy_yaw
    out["base"] = np.r_[x, y, z, yaw_quat(yaw)]
    out["joints"] = np.array([STAND[j] for j in JOINTS])
    return out
