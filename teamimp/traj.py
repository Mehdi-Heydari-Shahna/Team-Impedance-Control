"""Reference trajectory primitives (world-frame poses with velocities and accelerations).

A pose sample is (p, R, v, w, a, wd): position, rotation, linear velocity, angular velocity (world),
and their time derivatives.
"""
import numpy as np

from .so3 import exp, log, hat, minjerk


def _hold(p, R):
    z = np.zeros(3)
    return p.copy(), R.copy(), z, z.copy(), z.copy(), z.copy()


class _HoldSeg:
    """Constant pose segment (picklable)."""

    def __init__(self, p, R):
        self.p, self.R = p, R

    def __call__(self, t):
        return _hold(self.p, self.R)


class _MoveSeg:
    """Min-jerk pose segment with an optional vertical bump (picklable)."""

    def __init__(self, t0, T, p0, R0, D, phi, bump):
        self.t0, self.T, self.p0, self.R0, self.D, self.phi, self.bump = t0, T, p0, R0, D, phi, bump

    def __call__(self, t):
        t0, T, p0, R0, D, phi, bump = self.t0, self.T, self.p0, self.R0, self.D, self.phi, self.bump
        s = (t - t0) / T
        s = min(max(s, 0.0), 1.0)
        m, dm, ddm = minjerk(s)
        p = p0 + m * D
        v = dm / T * D
        a = ddm / T ** 2 * D
        if bump:
            b = 16 * s * s * (1 - s) ** 2
            db = 32 * s * (1 - s) * (1 - 2 * s)
            ddb = 32 * (1 - 6 * s + 6 * s * s)
            p = p + bump * b * np.array([0, 0, 1.0])
            v = v + bump * db / T * np.array([0, 0, 1.0])
            a = a + bump * ddb / T ** 2 * np.array([0, 0, 1.0])
        R = exp(m * phi) @ R0
        w = dm / T * phi
        wd = ddm / T ** 2 * phi
        return p, R, v, w, a, wd


class PoseSchedule:
    """Concatenation of time-ordered segments; each maps t -> pose sample on [t0, t1]. Holds between/after."""

    def __init__(self, p, R):
        self.segs = []
        self.p0 = np.asarray(p, float).copy()
        self.R0 = np.asarray(R, float).copy()
        self.p_end, self.R_end = self.p0.copy(), self.R0.copy()
        self.t_end = -np.inf

    def __call__(self, t):
        for (t0, t1, f) in reversed(self.segs):
            if t >= t0:
                if t <= t1:
                    return f(t)
                p, R = f(t1)[:2]
                return _hold(p, R)
        return _hold(self.p0, self.R0)

    def end_pose(self):
        return self.p_end.copy(), self.R_end.copy()

    def reset(self, p, R, t):
        """Restart from the given pose at time t (drops later segments)."""
        self.segs = [s for s in self.segs if s[1] <= t]
        p, R = np.asarray(p, float).copy(), np.asarray(R, float).copy()
        self.segs.append((t, t, _HoldSeg(p, R)))
        self.p_end, self.R_end, self.t_end = p.copy(), R.copy(), t

    def move(self, t0, T, p1=None, R1=None, bump=0.0, dp=None):
        """Min-jerk move from the current end pose to (p1, R1) over [t0, t0+T] (optionally a vertical bump)."""
        p0, R0 = self.p_end.copy(), self.R_end.copy()
        if p1 is None:
            p1 = p0 + (np.zeros(3) if dp is None else np.asarray(dp, float))
        p1 = np.asarray(p1, float).copy()
        R1 = R0.copy() if R1 is None else np.asarray(R1, float).copy()
        phi = log(R1 @ R0.T)
        D = p1 - p0

        f = _MoveSeg(t0, T, p0, R0, D, phi, bump)
        self.segs.append((t0, t0 + T, f))
        self.p_end, self.R_end, self.t_end = p1, R1, t0 + T
        return t0 + T


class RelPose:
    """Pose rigidly attached to a moving frame (e.g., a free hand attached to its pelvis reference)."""

    def __init__(self, base, r, Rr):
        self.base = base
        self.r = np.asarray(r, float)
        self.Rr = np.asarray(Rr, float)

    def __call__(self, t):
        p, R, v, w, a, wd = self.base(t)
        rr = R @ self.r
        return (p + rr, R @ self.Rr, v + np.cross(w, rr), w.copy(),
                a + np.cross(wd, rr) + np.cross(w, np.cross(w, rr)), wd.copy())


class VecSchedule:
    def __init__(self, x):
        self.segs = []
        self.x_end = np.asarray(x, float).copy()
        self.x0 = self.x_end.copy()

    def __call__(self, t):
        z = np.zeros_like(self.x_end)
        for (t0, t1, x0, x1) in reversed(self.segs):
            if t >= t0:
                if t <= t1:
                    m, dm, ddm = minjerk((t - t0) / (t1 - t0))
                    return x0 + m * (x1 - x0), dm / (t1 - t0) * (x1 - x0), ddm / (t1 - t0) ** 2 * (x1 - x0)
                return x1.copy(), z, z.copy()
        return self.x0.copy(), z, z.copy()

    def move(self, t0, T, x1):
        x1 = np.asarray(x1, float).copy()
        self.segs.append((t0, t0 + T, self.x_end.copy(), x1))
        self.x_end = x1
        return t0 + T
