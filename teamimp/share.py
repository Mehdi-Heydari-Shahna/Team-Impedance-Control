"""Load sharing through the internal channel: a QP over the internal wrench lambda_d (in N(G)) and the
foot distribution f_F (in N(L^T) cap N(Psi~)), with the feasibility margins of Corollary 1 as constraints
(contact wrench cones of soles and palms) plus actuator torque limits.

With lambda* = lambda_r - mu(Psi^+ lambda_d) - mu(f_F) (Corollary 1, nominal motion), the connection wrenches
and the efforts are affine in the decision variables:
    lambda_g = lambda_r,g + V_I z_d                      (grasp rows; f_F does not load the grasps)
    lambda_f = lambda_r,f - Mmu_f (Psi^+ V_I z_d + N_F c)
    f        = f_0 + Psi^+ V_I z_d + N_F c
"""
import numpy as np
import daqp

SOLE = dict(X=(-0.100, 0.100), Y=(-0.0425, 0.0425), mu=0.8)
PALM = dict(X=(-0.041, 0.041), Y=(-0.083, 0.070), mu=0.7)
L_SCALE = 0.1           # moments are measured as forces at 0.1 m in the margin metric


def cwc(X, Y, mu, cop_frac=1.0, fr_frac=1.0, fmin=0.0, tors=None):
    """Polyhedral contact wrench cone G w <= h for a rectangular contact (z = normal into the body that
    receives the wrench w = (f, m), moment about the frame origin on the contact plane)."""
    xlo, xhi = X
    ylo, yhi = Y
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    ax, ay = 0.5 * (xhi - xlo) * cop_frac, 0.5 * (yhi - ylo) * cop_frac
    xlo, xhi, ylo, yhi = cx - ax, cx + ax, cy - ay, cy + ay
    me = mu * fr_frac / np.sqrt(2.0)
    mt = (tors if tors is not None else 0.3 * mu * min(ax, ay)) * fr_frac
    G = np.array([
        [0, 0, -1, 0, 0, 0],
        [1, 0, -me, 0, 0, 0], [-1, 0, -me, 0, 0, 0], [0, 1, -me, 0, 0, 0], [0, -1, -me, 0, 0, 0],
        [0, 0, -xhi, 0, -1, 0], [0, 0, xlo, 0, 1, 0],
        [0, 0, -yhi, 1, 0, 0], [0, 0, ylo, -1, 0, 0],
        [0, 0, -mt, 0, 0, 1], [0, 0, -mt, 0, 0, -1]], float)
    h = np.zeros(G.shape[0])
    h[0] = -fmin
    return G, h


def margin(w, spec):
    """Corollary 1 margin rho_k: Euclidean distance of w from the boundary of the (full) cone in the scaled
    wrench space (f, m / L_SCALE); negative if outside. Also CoP margin (m), friction utilization."""
    G, h = cwc(spec["X"], spec["Y"], spec["mu"])
    S = np.r_[np.ones(3), L_SCALE * np.ones(3)]
    Gs = G * S[None, :]
    nrm = np.linalg.norm(Gs, axis=1)
    rho = np.min((h - G @ w) / nrm)
    fz = w[2]
    if fz > 1e-6:
        cop = np.array([-w[4] / fz, w[3] / fz])
        cm = min(cop[0] - spec["X"][0], spec["X"][1] - cop[0], cop[1] - spec["Y"][0], spec["Y"][1] - cop[1])
        fu = np.hypot(w[0], w[1]) / (spec["mu"] * fz)
    else:
        cm, fu = -np.inf, np.inf
    return rho, cm, fu


class LoadShare:
    def __init__(self, cop_frac=0.7, fr_frac=0.7, fmin_foot=30.0, fmin_palm=25.0, tau_frac=0.85,
                 w_int=1e-3, w_foot=1e-4, w_soft=1.0, w_hard=1e6, w_tau=1e5, every=10, tau_filter=0.08,
                 w_tan=30.0, tau_ff_only=True, w_soft_palm=10.0):
        self.cop_frac, self.fr_frac = cop_frac, fr_frac
        # margin slacks of the palms weigh more than those of the soles: a light payload loads the palms little, and
        # the grasps must not be traded for sole margins (the soles keep their full cones as hard limits)
        self.w_soft_palm = w_soft if w_soft_palm is None else w_soft_palm
        self.fmin_foot, self.fmin_palm = fmin_foot, fmin_palm
        self.tau_frac = tau_frac
        self.w_int, self.w_foot = w_int, w_foot
        self.w_soft, self.w_hard, self.w_tau = w_soft, w_hard, w_tau
        self.w_tan = w_tan
        self.tau_ff_only = tau_ff_only
        self.every = every
        self.tau_filter = tau_filter
        self.caps = {}            # (i, s) -> max normal force of foot (unloading before lift-off)
        self.use_internal = True
        self.use_feet = True
        self.k = 0
        self.lam_d = None
        self.fF = None
        self.lam_d_raw = None
        self.fF_raw = None
        self.info = {}

    def reset(self):
        self.k = 0
        self.lam_d = None
        self.fF = None
        self.lam_d_raw = None
        self.fF_raw = None

    def solve(self, tm, last, frc_lim):
        """One QP solve at the current state; returns (lam_d, fF, info).
        Slack: per connection a cheap one for the margin cone and an expensive one for the physical cone
        (unilaterality, full friction cone and contact area); an expensive one for the torque limits."""
        nI = tm.nI if (self.use_internal and tm.ng) else 0
        NF = tm.NF if self.use_feet else np.zeros((tm.mact, 0))
        nF = NF.shape[1]
        PV = tm.Psip @ tm.VI[:, :nI] if nI else np.zeros((tm.mact, 0))
        lam0 = last["lam_r"]
        f0 = last["fr"] if self.tau_ff_only else last["f"] - last["f_int"] - last["fF"]
        if self.tau_ff_only and tm.ng and "psi" in last:
            # include the effort of the internal-impedance state (psi - lambda_d), computed by the controller
            f0 = f0 + tm.Psip @ (last["psi"] - last["lam_d"])
        conns = []
        for k, (kind, i, s) in enumerate(tm.rows):
            sl = slice(6 * k, 6 * k + 6)
            spec = SOLE if kind == "foot" else PALM
            fmin = self.fmin_foot if kind == "foot" else self.fmin_palm
            cap = self.caps.get((i, s), None) if kind == "foot" else None
            if cap is not None:
                fmin = min(fmin, 0.5 * cap)
            Gs, hs = cwc(spec["X"], spec["Y"], spec["mu"], self.cop_frac, self.fr_frac, fmin)
            Gh, hh = cwc(spec["X"], spec["Y"], spec["mu"], 1.0, 1.0, 0.0)
            if cap is not None:
                Gh = np.vstack([Gh, [0, 0, 1, 0, 0, 0]])
                hh = np.r_[hh, cap]
            conns.append((k, kind, i, s, sl, Gs, hs, Gh, hh))
        nC = len(conns)
        nS = 2 * nC + 1
        nx = nI + nF + nS
        Arows, ub = [], []
        Mf = tm.Mmu
        dLz = -Mf @ PV if nI else np.zeros((tm.r, 0))
        dLc = -Mf @ NF if nF else np.zeros((tm.r, 0))
        S = np.r_[np.ones(3), L_SCALE * np.ones(3)]
        for j, (k, kind, i, s, sl, Gs, hs, Gh, hh) in enumerate(conns):
            for (G, h, js) in ((Gs, hs, nI + nF + j), (Gh, hh, nI + nF + nC + j)):
                nrm = np.linalg.norm(G * S[None, :], axis=1)
                Gn = G / nrm[:, None]
                hn = h / nrm
                row = np.zeros((G.shape[0], nx))
                if nI:
                    row[:, :nI] = Gn @ dLz[sl]
                if nF:
                    row[:, nI:nI + nF] = Gn @ dLc[sl]
                row[:, js] = -1.0
                Arows.append(row)
                ub.append(hn - Gn @ lam0[sl])
        lim = self.tau_frac * frc_lim
        Tz = np.zeros((tm.mact, nx))
        if nI:
            Tz[:, :nI] = PV
        if nF:
            Tz[:, nI:nI + nF] = NF
        Tn = Tz / lim[:, None]
        sl_t = np.zeros((tm.mact, nx))
        sl_t[:, nx - 1] = -1.0
        Arows += [Tn + sl_t, -Tn + sl_t]
        ub += [1.0 - f0 / lim, 1.0 + f0 / lim]
        A = np.vstack(Arows)
        bu = np.concatenate(ub)
        bl = -1e30 * np.ones_like(bu)
        H = np.zeros((nx, nx))
        if nI:
            # tangential internal forces (friction) are expensive, normal-force and moment redistribution cheap
            Wg = np.tile(np.r_[self.w_tan, self.w_tan, 1.0, np.ones(3) / 0.1 ** 2], tm.ng)
            H[:nI, :nI] = self.w_int * (tm.VI[:, :nI].T * Wg) @ tm.VI[:, :nI]
        if nF:
            H[nI:nI + nF, nI:nI + nF] = self.w_foot * NF.T @ NF + 1e-9 * np.eye(nF)
        o = nI + nF
        H[o:o + nC, o:o + nC] = np.diag([self.w_soft if c[1] == "foot" else self.w_soft_palm for c in conns])
        H[o + nC:o + 2 * nC, o + nC:o + 2 * nC] = self.w_hard * np.eye(nC)
        H[nx - 1, nx - 1] = self.w_tau
        H += 1e-10 * np.eye(nx)
        fq = np.zeros(nx)
        bu_s = np.r_[1e30 * np.ones(nI + nF), 1e30 * np.ones(nS)]
        bl_s = np.r_[-1e30 * np.ones(nI + nF), np.zeros(nS)]
        x, fval, flag, inf = daqp.solve(H, fq, A, np.r_[bu_s, bu], np.r_[bl_s, bl])
        if flag < 0 or x is None or not np.all(np.isfinite(x)):
            self.info = dict(flag=flag, slack_soft=np.zeros(nC), slack_hard=np.zeros(nC), slack_tau=0.0, n=nx,
                             m=A.shape[0], failed=True)
            return None, None, self.info
        zd = x[:nI]
        c = x[nI:nI + nF]
        lam_d = tm.VI[:, :nI] @ zd if nI else np.zeros(6 * tm.ng)
        fF = NF @ c if nF else np.zeros(tm.mact)
        self.info = dict(flag=flag, slack_soft=x[o:o + nC].copy(), slack_hard=x[o + nC:o + 2 * nC].copy(),
                         slack_tau=float(x[nx - 1]), n=nx, m=A.shape[0])
        return lam_d, fF, self.info

    def update(self, tm, last, frc_lim, dt):
        """Call once per control step after the controller computed its effort."""
        if self.k % self.every == 0 or self.lam_d is None:
            lam_d, fF, _ = self.solve(tm, last, frc_lim)
            if lam_d is not None:
                self.lam_d_raw, self.fF_raw = lam_d, fF
            elif self.lam_d_raw is None or self.lam_d_raw.shape[0] != 6 * tm.ng \
                    or self.fF_raw is None or self.fF_raw.shape[0] != tm.mact:
                self.lam_d_raw, self.fF_raw = np.zeros(6 * tm.ng), np.zeros(tm.mact)
        self.k += 1
        a = 1.0 - np.exp(-dt / self.tau_filter)
        if self.lam_d is None or self.lam_d.shape != self.lam_d_raw.shape:
            self.lam_d = self.lam_d_raw.copy()
        else:
            self.lam_d += a * (self.lam_d_raw - self.lam_d)
        if self.fF is None or self.fF.shape != self.fF_raw.shape:
            self.fF = self.fF_raw.copy()
        else:
            self.fF += a * (self.fF_raw - self.fF)
        return self.lam_d, self.fF


def min_max_utilization(tm, last, frc_lim, use_internal=True, use_feet=True, caps=None, cop_frac=1.0, fr_frac=1.0,
                        fmin=(5.0, 5.0)):
    """Smallest achievable maximum actuator utilization max_k |f_k| / lim_k over the internal channel
    (lambda_d in N(G), f_F in N(L^T) cap N(Psi~)) such that every connection wrench of the nominal motion
    (Corollary 1, s = 0, z = 0) lies in its contact wrench cone. Returns (u*, lam_d, fF, flag); u* > 1 means
    that no load sharing keeps the actuators within their limits (failure predicted)."""
    nI = tm.nI if (use_internal and tm.ng) else 0
    NF = tm.NF if use_feet else np.zeros((tm.mact, 0))
    nF = NF.shape[1]
    PV = tm.Psip @ tm.VI[:, :nI] if nI else np.zeros((tm.mact, 0))
    lam0 = last["lam_r"]
    f0 = last["fr"]
    nx = nI + nF + 1
    rows, ub = [], []
    dLz = -tm.Mmu @ PV if nI else np.zeros((tm.r, 0))
    dLc = -tm.Mmu @ NF if nF else np.zeros((tm.r, 0))
    S = np.r_[np.ones(3), L_SCALE * np.ones(3)]
    for k, (kind, i, s) in enumerate(tm.rows):
        sl = slice(6 * k, 6 * k + 6)
        spec = SOLE if kind == "foot" else PALM
        G, h = cwc(spec["X"], spec["Y"], spec["mu"], cop_frac, fr_frac, fmin[0] if kind == "foot" else fmin[1])
        if caps and (i, s) in caps:
            G = np.vstack([G, [0, 0, 1, 0, 0, 0]])
            h = np.r_[h, caps[(i, s)]]
        nrm = np.linalg.norm(G * S[None, :], axis=1)
        Gn, hn = G / nrm[:, None], h / nrm
        row = np.zeros((G.shape[0], nx))
        if nI:
            row[:, :nI] = Gn @ dLz[sl]
        if nF:
            row[:, nI:nI + nF] = Gn @ dLc[sl]
        rows.append(row)
        ub.append(hn - Gn @ lam0[sl])
    T = np.zeros((tm.mact, nx))
    if nI:
        T[:, :nI] = PV
    if nF:
        T[:, nI:nI + nF] = NF
    Tn = T / frc_lim[:, None]
    ucol = np.zeros((tm.mact, nx))
    ucol[:, -1] = -1.0
    rows += [Tn + ucol, -Tn + ucol]
    ub += [-f0 / frc_lim, f0 / frc_lim]
    A = np.vstack(rows)
    bu = np.concatenate(ub)
    H = 1e-6 * np.eye(nx)
    H[-1, -1] = 1e-4
    fq = np.zeros(nx)
    fq[-1] = 1.0
    x, fval, flag, info = daqp.solve(H, fq, A, bu, -1e30 * np.ones_like(bu))
    if x is None or flag < 0:
        return np.inf, None, None, flag
    lam_d = tm.VI[:, :nI] @ x[:nI] if nI else np.zeros(6 * tm.ng)
    fF = NF @ x[nI:nI + nF] if nF else np.zeros(tm.mact)
    return float(x[-1]), lam_d, fF, flag
