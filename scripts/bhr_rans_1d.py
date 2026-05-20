#!/usr/bin/env python3
"""1D spherical-radial compressible Euler + BHR k-eps-a-b RANS model.

Models the chamber blast in spherical symmetry. The host hydro (HLLC + MUSCL,
spherical finite volume) evolves the compressible mean flow incl. the shock;
the BHR closure adds turbulent pressure feedback and transports k, eps, a_r, b.

Non-dimensional units matching the DNS: gamma=1.4, R=1, smooth_sphere IC
(rho_B=10, T_B=100, r0=0.1; ambient rho=1, T=1). Reflecting wall at R_wall.

Usage:
    python3 bhr_rans_1d.py [--n 400] [--rwall 0.5] [--tend 0.5]
                            [--bhr 1|0] [--out out_bhr_rans_1d.npz]
"""

import argparse
import numpy as np

GAMMA = 1.4
RGAS = 1.0


# ----------------------------------------------------------------------------
# Thermodynamics + reconstruction
# ----------------------------------------------------------------------------
def primitive(U):
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    p = (GAMMA - 1.0) * rho * (E - 0.5 * u * u)
    p = np.maximum(p, 1e-12)
    return rho, u, p


def conservative(rho, u, p):
    E = p / ((GAMMA - 1.0) * rho) + 0.5 * u * u
    return np.array([rho, rho * u, rho * E])


def minmod(a, b):
    s = np.sign(a)
    return s * np.maximum(0.0, np.minimum(np.abs(a), s * b))


def muscl_face_states(q):
    """Given cell-centered q[0..N-1], return left/right reconstructed states
    at interior faces 1..N-1: qL at face i = right edge of cell i-1,
    qR at face i = left edge of cell i. Minmod-limited."""
    dq = np.zeros_like(q)
    dq[1:-1] = minmod(q[1:-1] - q[:-2], q[2:] - q[1:-1])
    qL = q[:-1] + 0.5 * dq[:-1]   # state to the left of face f=1..N-1
    qR = q[1:] - 0.5 * dq[1:]     # state to the right
    return qL, qR


# ----------------------------------------------------------------------------
# HLLC flux (1D Euler), with an extra passively-advected mass-specific scalar
# ----------------------------------------------------------------------------
def hllc_flux(rhoL, uL, pL, rhoR, uR, pR):
    """Return (F_rho, F_mom, F_ene, S_star, mass_flux) at each face."""
    EL = pL / ((GAMMA - 1.0) * rhoL) + 0.5 * uL * uL
    ER = pR / ((GAMMA - 1.0) * rhoR) + 0.5 * uR * uR
    cL = np.sqrt(GAMMA * pL / rhoL)
    cR = np.sqrt(GAMMA * pR / rhoR)

    # Wave-speed estimates (Davis / Einfeldt).
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)

    # Ensure SL is genuinely left-going and SR right-going so the star-state
    # denominators (S-u) stay bounded away from zero (avoids the classic
    # HLLC divide-by-zero when a wave speed coincides with the local velocity).
    SL = np.minimum(SL, uL - 1e-6 * cL)
    SR = np.maximum(SR, uR + 1e-6 * cR)

    # Contact speed.
    num = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR))
    den = (rhoL * (SL - uL) - rhoR * (SR - uR))
    Sstar = num / np.where(np.abs(den) < 1e-30, 1e-30, den)

    # Left/right physical fluxes.
    FL = np.array([rhoL * uL,
                   rhoL * uL * uL + pL,
                   uL * (rhoL * EL + pL)])
    FR = np.array([rhoR * uR,
                   rhoR * uR * uR + pR,
                   uR * (rhoR * ER + pR)])

    # Star-region conserved states (guarded denominators).
    def star(rho, u, p, E, S):
        smu = S - u
        smu = np.where(np.abs(smu) < 1e-30, np.sign(smu) * 1e-30 + 1e-30, smu)
        sms = S - Sstar
        sms = np.where(np.abs(sms) < 1e-30, np.sign(sms) * 1e-30 + 1e-30, sms)
        coef = rho * smu / sms
        Us0 = coef
        Us1 = coef * Sstar
        Us2 = coef * (E + (Sstar - u) * (Sstar + p / (rho * smu)))
        return np.array([Us0, Us1, Us2])

    UL = np.array([rhoL, rhoL * uL, rhoL * EL])
    UR = np.array([rhoR, rhoR * uR, rhoR * ER])
    UsL = star(rhoL, uL, pL, EL, SL)
    UsR = star(rhoR, uR, pR, ER, SR)

    FsL = FL + SL * (UsL - UL)
    FsR = FR + SR * (UsR - UR)

    F = np.empty_like(FL)
    for comp in range(3):
        F[comp] = np.select(
            [SL >= 0, (SL < 0) & (Sstar >= 0),
             (Sstar < 0) & (SR > 0), SR <= 0],
            [FL[comp], FsL[comp], FsR[comp], FR[comp]])
    mass_flux = F[0]
    return F[0], F[1], F[2], Sstar, mass_flux


# ----------------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------------
class Grid:
    def __init__(self, n, rwall):
        self.n = n
        self.rf = np.linspace(0.0, rwall, n + 1)   # faces
        self.rc = 0.5 * (self.rf[:-1] + self.rf[1:])
        self.Af = 4.0 * np.pi * self.rf ** 2        # face areas
        self.V = (4.0 * np.pi / 3.0) * (self.rf[1:] ** 3 - self.rf[:-1] ** 3)
        self.dr = self.rf[1:] - self.rf[:-1]


# ----------------------------------------------------------------------------
# Initial condition (smooth sphere, matching DNS)
# ----------------------------------------------------------------------------
def init_state(g, rho_B=10.0, T_B=100.0, rho0=1.0, T0=1.0, r0=0.1,
               thickness=0.012):
    r = g.rc
    ramp = 0.5 * (1.0 - np.tanh((r - r0) / thickness))
    rho = rho0 + (rho_B - rho0) * ramp
    T = T0 + (T_B - T0) * ramp
    p = rho * RGAS * T
    u = np.zeros_like(r)
    U = conservative(rho, u, p)
    return U


# ----------------------------------------------------------------------------
# BHR constants
# ----------------------------------------------------------------------------
class BHR:
    C_mu = 0.09
    C_e1 = 1.44
    C_e2 = 1.92
    C_e3 = 1.0
    C_a = 1.0
    C_b = 1.0
    sigma_k = 1.0
    sigma_e = 1.3
    sigma_a = 1.0
    sigma_b = 1.0
    k_floor = 1e-10
    e_floor = 1e-12
    b_max = 2.0
    mu_phys = 5.0e-4
    prandtl = 0.71
    prod_limit = 10.0      # Menter production limiter: P_k <= prod_limit * rho * eps
    L_max = 0.5            # mixing-length ceiling (= R_wall) -> caps mu_t
    pt_frac_max = 0.5      # turbulent pressure <= pt_frac_max * thermal pressure


# ----------------------------------------------------------------------------
# Spherical divergence helper for a face flux array Ff[0..N] -> cell update
# ----------------------------------------------------------------------------
def div_flux(g, Ff):
    return (g.Af[1:] * Ff[1:] - g.Af[:-1] * Ff[:-1]) / g.V


def grad_centered(g, q):
    """d q / d r at cell centers, central difference, one-sided at ends."""
    dq = np.zeros_like(q)
    dq[1:-1] = (q[2:] - q[:-2]) / (g.rc[2:] - g.rc[:-2])
    dq[0] = (q[1] - q[0]) / (g.rc[1] - g.rc[0])
    dq[-1] = (q[-1] - q[-2]) / (g.rc[-1] - g.rc[-2])
    return dq


# ----------------------------------------------------------------------------
# Main solver
# ----------------------------------------------------------------------------
def run(n=400, rwall=0.5, tend=0.5, use_bhr=True, cfl=0.3, feedback=False,
        c_mu=None, c_a=None, c_b=None, prod_limit=None, seed_scale=1e-3,
        b_seed=0.0,
        save_times=(0.02, 0.05, 0.1, 0.25, 0.5), out="out_bhr_rans_1d.npz"):
    # Calibration overrides (fall back to BHR class defaults).
    C_mu = BHR.C_mu if c_mu is None else c_mu
    C_a = BHR.C_a if c_a is None else c_a
    C_b = BHR.C_b if c_b is None else c_b
    PRODLIM = BHR.prod_limit if prod_limit is None else prod_limit
    g = Grid(n, rwall)
    U = init_state(g)
    rho, u, p = primitive(U)

    # Turbulence primitives.
    e_int = p / ((GAMMA - 1.0) * rho)
    k = np.full(n, BHR.k_floor)
    if use_bhr:
        # Seed k at the contact layer (large drho/dr).
        drho = np.abs(grad_centered(g, rho))
        seed = seed_scale * e_int * (drho / (drho.max() + 1e-30))
        k = np.maximum(seed, BHR.k_floor)
    len0 = 0.1 / 4.0
    eps = np.maximum(C_mu * k ** 1.5 / len0, BHR.e_floor)
    a = np.zeros(n)   # radial mass flux a_r
    b = np.zeros(n)   # density-spec-vol covariance
    if use_bhr and b_seed > 0.0:
        # Seed density-spec-vol covariance at the contact layer. In 1D
        # spherical symmetry grad(rho) || grad(p), so baroclinic vorticity
        # is zero and the turbulence cannot self-generate from the mean
        # flow; b_seed represents the 3D (Y4,2 + ensemble) perturbation
        # that drives the real instability. It bootstraps a_r via b*dp/dr.
        drho = np.abs(grad_centered(g, rho))
        b = b_seed * (drho / (drho.max() + 1e-30))

    saves = []
    save_set = sorted(save_times)
    next_save = 0
    t = 0.0
    step = 0
    ke_hist = []

    while t < tend - 1e-12:
        rho, u, p = primitive(U)
        c = np.sqrt(np.maximum(GAMMA * p / rho, 1e-12))
        # Turbulent pressure feeds the hydro only when feedback is on.
        pt = np.minimum((2.0 / 3.0) * rho * k, BHR.pt_frac_max * p) \
            if (use_bhr and feedback) else np.zeros(n)
        c_eff = np.sqrt(c * c + (5.0 / 3.0) * pt / rho) if feedback else c
        dt = cfl * np.min(g.dr / (np.abs(u) + c_eff + 1e-30))
        dt = min(dt, tend - t)

        # ---- Hydro substep (operator split, explicit Euler in time) ----
        # Reconstruct primitives (rho, u, p+pt) to faces.
        p_tot = p + pt
        rhoL, rhoR = muscl_face_states(rho)
        uL, uR = muscl_face_states(u)
        pL, pR = muscl_face_states(p_tot)
        rhoL = np.maximum(rhoL, 1e-10); rhoR = np.maximum(rhoR, 1e-10)
        pL = np.maximum(pL, 1e-12); pR = np.maximum(pR, 1e-12)

        Fr, Fm, Fe, Sstar, mflux = hllc_flux(rhoL, uL, pL, rhoR, uR, pR)

        # Assemble interior face fluxes (faces 1..N-1); walls handled below.
        Frho = np.zeros(n + 1)
        Fmom = np.zeros(n + 1)
        Fene = np.zeros(n + 1)
        Mflux = np.zeros(n + 1)
        Frho[1:-1] = Fr
        Fmom[1:-1] = Fm
        Fene[1:-1] = Fe
        Mflux[1:-1] = mflux

        # Reflecting walls: zero mass/energy flux; momentum flux = wall pressure.
        # r=0 center.
        Fmom[0] = p_tot[0]
        # r=R_wall.
        Fmom[-1] = p_tot[-1]

        dU = np.zeros_like(U)
        dU[0] = -div_flux(g, Frho)
        # Momentum: area-weighted (rho u^2 + p) flux divergence already
        # contains the d(r^2 p)/dr part; add geometric +2 p_tot / r source.
        dU[1] = -div_flux(g, Fmom) + 2.0 * p_tot / g.rc
        dU[2] = -div_flux(g, Fene)

        Unew = U + dt * dU

        # ---- Advect turbulence scalars with the same mass flux (upwind) ----
        if use_bhr:
            def advect(phi):
                # upwind phi at interior faces by sign of Mflux
                phL = phi[:-1]
                phR = phi[1:]
                phi_face = np.where(Mflux[1:-1] >= 0, phL, phR)
                Fphi = np.zeros(n + 1)
                Fphi[1:-1] = Mflux[1:-1] * phi_face
                # walls: zero mass flux -> zero scalar flux
                return -div_flux(g, Fphi)
            rho_k = rho * k + dt * advect(k)
            rho_e = rho * eps + dt * advect(eps)
            rho_a = rho * a + dt * advect(a)
            rho_b = rho * b + dt * advect(b)
            rho_new = np.maximum(Unew[0], 1e-10)
            k = rho_k / rho_new
            eps = rho_e / rho_new
            a = rho_a / rho_new
            b = rho_b / rho_new

        U = Unew

        # ---- BHR source substep (point-implicit destruction) ----
        if use_bhr:
            rho, u, p = primitive(U)
            # Mixing-length-bounded eps floor caps mu_t = C_mu rho k^2/eps at
            # rho sqrt(k) L_max (eddy size cannot exceed the domain).
            eps = np.maximum(eps, C_mu * k ** 1.5 / BHR.L_max)
            mu_t = C_mu * rho * k * k / np.maximum(eps, BHR.e_floor)

            dudr = grad_centered(g, u)
            dpdr = grad_centered(g, p)
            drhodr = grad_centered(g, rho)

            # Deviatoric Reynolds stress (radial), R_rr = 2 mu_t dudr - 2/3 rho k.
            R_rr = 2.0 * mu_t * dudr - (2.0 / 3.0) * rho * k

            P_k = R_rr * dudr                # shear production (shock-prone)
            P_b = a * dpdr                   # variable-density production
            # Menter limiter on the SHEAR production only (it spikes at the
            # captured shock). The variable-density production P_b is the
            # physical blast-turbulence driver and is left unclamped (it is
            # already bounded via the realizability cap on a).
            P_k = np.clip(P_k, -PRODLIM * rho * eps, PRODLIM * rho * eps)
            prod = P_k + P_b

            # Diffusion (turbulent) of k, eps, a, b.
            def diff(phi, sigma):
                # d/dr[(mu+mu_t/sigma) dphi/dr] in spherical FV
                coef_f = np.zeros(n + 1)
                coef_c = BHR.mu_phys + mu_t / sigma
                coef_f[1:-1] = 0.5 * (coef_c[:-1] + coef_c[1:])
                dphidr_f = np.zeros(n + 1)
                dphidr_f[1:-1] = (phi[1:] - phi[:-1]) / (g.rc[1:] - g.rc[:-1])
                flux = coef_f * dphidr_f
                return div_flux(g, flux)

            Dk = diff(k, BHR.sigma_k)
            De = diff(eps, BHR.sigma_e)
            Da = diff(a, BHR.sigma_a)
            Db = diff(b, BHR.sigma_b)

            inv_k = 1.0 / np.maximum(k, BHR.k_floor)

            # Explicit production + diffusion; implicit linear destruction.
            # k: d(rho k)/dt = prod - rho eps + Dk
            rho_k = rho * k + dt * (prod + Dk)
            # implicit eps sink in k: rho eps  -> treat eps as known here.
            rho_k = rho_k - dt * rho * eps
            k = np.maximum(rho_k / rho, BHR.k_floor)

            # eps: production (eps/k) Ce1 * prod (limited) ; destruction -Ce2 rho eps^2/k
            eps_prod = (eps * inv_k) * (BHR.C_e1 * prod) + De
            # point-implicit for the -Ce2 rho eps^2/k term:
            #   rho eps^{n+1} = rho eps + dt eps_prod - dt Ce2 rho (eps/k) eps^{n+1}
            denom = 1.0 + dt * BHR.C_e2 * eps * inv_k
            eps = np.maximum((eps + dt * eps_prod / np.maximum(rho, 1e-12)) / denom,
                             BHR.e_floor)

            # a_r: prod = b dp/dr - (R_rr/rho) drho/dr - rho a dudr ; sink -Ca rho (eps/k) a
            a_prod = (b * dpdr - (R_rr / rho) * drhodr - rho * a * dudr + Da)
            denom_a = 1.0 + dt * C_a * eps * inv_k
            a = (a + dt * a_prod / np.maximum(rho, 1e-12)) / denom_a

            # b: prod = -2 a (b+1) drho/dr / rho ; sink -Cb rho (eps/k) b
            b_prod = -2.0 * a * (b + 1.0) * drhodr / np.maximum(rho, 1e-12) + Db / np.maximum(rho, 1e-12)
            denom_b = 1.0 + dt * C_b * eps * inv_k
            b = (b + dt * b_prod) / denom_b

            # ---- Reynolds-stress feedback on mean momentum/energy ----
            # Optional: one-way coupling (feedback=False) keeps the mean flow
            # identical to the pure-hydro blast and tests whether BHR predicts
            # the right turbulence given that mean flow. Two-way (feedback=True)
            # adds the deviatoric stress + dissipative heating.
            if feedback:
                dev = 2.0 * mu_t * dudr
                stressflux = np.zeros(n + 1)
                stressflux[1:-1] = 0.5 * (dev[:-1] + dev[1:])
                dmom = div_flux(g, stressflux)
                U[1] = U[1] + dt * dmom
                U[2] = U[2] + dt * (dmom * u + rho * eps)

            # ---- Realizability clips (scrub NaN/inf first; clip passes NaN) ----
            k_max = 50.0
            k = np.nan_to_num(k, nan=BHR.k_floor, posinf=k_max, neginf=BHR.k_floor)
            eps = np.nan_to_num(eps, nan=BHR.e_floor, posinf=1e6, neginf=BHR.e_floor)
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            b = np.nan_to_num(b, nan=0.0, posinf=BHR.b_max, neginf=0.0)
            k = np.clip(k, BHR.k_floor, k_max)
            eps = np.maximum(eps, BHR.e_floor)
            b = np.clip(b, 0.0, BHR.b_max)
            a_bound = np.sqrt(2.0 * k * np.maximum(b, 0.0)) + 1e-12
            a = np.clip(a, -a_bound, a_bound)

        # Diagnostics.
        rho, u, p = primitive(U)
        KE = np.sum(0.5 * rho * u * u * g.V)
        TKE = np.sum(rho * k * g.V) if use_bhr else 0.0
        ke_hist.append((t, KE, TKE))

        # Save snapshots.
        while next_save < len(save_set) and t + dt >= save_set[next_save] - 1e-9:
            ts = save_set[next_save]
            saves.append(dict(t=t + dt, r=g.rc.copy(),
                              rho=rho.copy(), u=u.copy(), p=p.copy(),
                              k=k.copy(), eps=eps.copy(), a=a.copy(), b=b.copy()))
            next_save += 1

        t += dt
        step += 1
        if step % 2000 == 0:
            print(f"step {step}  t={t:.4f}  dt={dt:.2e}  KE={KE:.4e}  TKE={TKE:.4e}")

    ke_hist = np.array(ke_hist)
    np.savez(out, saves=np.array(saves, dtype=object), ke_hist=ke_hist,
             use_bhr=use_bhr)
    print(f"Done. {step} steps. Wrote {out} with {len(saves)} snapshots.")
    return saves, ke_hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--rwall", type=float, default=0.5)
    ap.add_argument("--tend", type=float, default=0.5)
    ap.add_argument("--bhr", type=int, default=1)
    ap.add_argument("--out", default="out_bhr_rans_1d.npz")
    args = ap.parse_args()
    run(n=args.n, rwall=args.rwall, tend=args.tend,
        use_bhr=bool(args.bhr), out=args.out)
