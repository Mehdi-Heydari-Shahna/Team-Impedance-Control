"""SO(3)/SE(3) helpers."""
import numpy as np
import mujoco


def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def exp(phi):
    th = np.linalg.norm(phi)
    K = hat(phi)
    if th < 1e-9:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def log(R):
    """Principal logarithm, returned as a vector (angle < pi)."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-7:
        return 0.5 * vee(R - R.T)
    if np.pi - th < 1e-5:
        # near pi: use the symmetric part
        A = (R + np.eye(3)) / 2
        k = np.argmax(np.diag(A))
        ax = A[:, k] / np.sqrt(max(A[k, k], 1e-12))
        return th * ax / np.linalg.norm(ax)
    return th / (2 * np.sin(th)) * vee(R - R.T)


def jl_inv(phi):
    """Inverse of the left Jacobian of SO(3): d/dt log(R) = J_l^{-1}(phi) zeta for dR/dt = hat(zeta) R."""
    th = np.linalg.norm(phi)
    K = hat(phi)
    if th < 1e-6:
        return np.eye(3) - 0.5 * K + K @ K / 12.0
    return np.eye(3) - 0.5 * K + (1.0 / th ** 2 - (1 + np.cos(th)) / (2 * th * np.sin(th))) * K @ K


def quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def minjerk(s):
    """Quintic blend on [0,1]: value, first and second derivative w.r.t. s."""
    s = np.clip(s, 0.0, 1.0)
    return (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5, 30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4,
            60 * s - 180 * s ** 2 + 120 * s ** 3)
