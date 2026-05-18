#include "physics/Forcing.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>

namespace blast {

SpectralForcing::SpectralForcing(const Grid& g, const Params& p)
    : p_(p), rng_(static_cast<unsigned int>(p.seed)) {
    const Real two_pi_Lx = 2.0 * M_PI / g.lx;
    const Real two_pi_Ly = 2.0 * M_PI / g.ly;
    const Real two_pi_Lz = 2.0 * M_PI / g.lz;

    std::normal_distribution<Real> nd(0.0, 1.0);
    // OU stationary variance = T_corr/2 per real component, so |a|^2 averaged
    // over time equals T_corr (real^2 + imag^2). Initialize at stationary.
    const Real sigma0 = std::sqrt(p.T_corr / 2.0);

    // Enumerate the Hermitian half of the integer wavenumber lattice within
    // the spherical shell [k_lo, k_hi]. Canonical half-space (so each
    // (k, -k) pair is counted exactly once): mz > 0, or
    // (mz == 0 and my > 0), or (mz == 0 and my == 0 and mx > 0).
    for (int mz = 0; mz <= p.k_hi; ++mz) {
        const int my_lo = (mz == 0) ? 0 : -p.k_hi;
        for (int my = my_lo; my <= p.k_hi; ++my) {
            const int mx_lo = (mz == 0 && my == 0) ? 1 : -p.k_hi;
            for (int mx = mx_lo; mx <= p.k_hi; ++mx) {
                const Real ki = std::sqrt(
                    static_cast<Real>(mx * mx + my * my + mz * mz));
                if (ki < p.k_lo - 0.5 || ki > p.k_hi + 0.5) continue;

                Mode m;
                m.kx = mx * two_pi_Lx;
                m.ky = my * two_pi_Ly;
                m.kz = mz * two_pi_Lz;
                m.mx = mx;  m.my = my;  m.mz = mz;
                m.kmag = std::sqrt(m.kx * m.kx + m.ky * m.ky + m.kz * m.kz);

                const Real khx = m.kx / m.kmag;
                const Real khy = m.ky / m.kmag;
                const Real khz = m.kz / m.kmag;
                // Build e1 perpendicular to k; e2 = khat x e1.
                if (std::fabs(khz) < 0.9) {
                    const Real s = 1.0 / std::sqrt(khx * khx + khy * khy);
                    m.e1[0] = -khy * s; m.e1[1] = khx * s; m.e1[2] = 0.0;
                } else {
                    const Real s = 1.0 / std::sqrt(khy * khy + khz * khz);
                    m.e1[0] = 0.0; m.e1[1] = -khz * s; m.e1[2] = khy * s;
                }
                m.e2[0] = khy * m.e1[2] - khz * m.e1[1];
                m.e2[1] = khz * m.e1[0] - khx * m.e1[2];
                m.e2[2] = khx * m.e1[1] - khy * m.e1[0];

                m.a1 = std::complex<Real>(sigma0 * nd(rng_), sigma0 * nd(rng_));
                m.a2 = std::complex<Real>(sigma0 * nd(rng_), sigma0 * nd(rng_));
                modes_.push_back(m);
            }
        }
    }
}

void SpectralForcing::ensure_trig_tables(const Grid& local) const {
    // Rebuild tables if the local grid (rank, dx, x0) changed since last call.
    // Same physical extent + same modes -> bit-exact reuse on every call.
    if (tab_nx_ == local.nx && tab_ny_ == local.ny && tab_nz_ == local.nz
        && tab_x0_ == local.x0 && tab_y0_ == local.y0 && tab_z0_ == local.z0
        && tab_dx_ == local.dx() && tab_dy_ == local.dy() && tab_dz_ == local.dz())
        return;

    tab_nx_ = local.nx; tab_ny_ = local.ny; tab_nz_ = local.nz;
    tab_x0_ = local.x0; tab_y0_ = local.y0; tab_z0_ = local.z0;
    tab_dx_ = local.dx(); tab_dy_ = local.dy(); tab_dz_ = local.dz();

    const int M = p_.k_hi;
    const int span = 2 * M + 1;

    auto build = [&](Real x0, Real dx, int n,
                     std::vector<Real>& cos_t, std::vector<Real>& sin_t) {
        cos_t.assign(static_cast<std::size_t>(span) * n, 0.0);
        sin_t.assign(static_cast<std::size_t>(span) * n, 0.0);
        // For each integer m in [-M, +M] (table index m_idx = m + M),
        // store cos(m * 2pi/L * x_i) and sin(...) for every cell index i.
        // L = (n_global) * dx, but cos(m * 2pi/L * x) for x on the cell-center
        // grid uses the physical-space relation: phase = m * two_pi_per_L * x.
        // We compute phase directly from the cell center coordinate so the
        // table is grid-aligned by construction.
        for (int idx = 0; idx < span; ++idx) {
            const int m = idx - M;
            const Real freq = static_cast<Real>(m) * 2.0 * M_PI;
            // freq * x / L, where L is the GLOBAL extent. We stored 2*pi*m/L
            // directly via modes_; reuse here by querying the first mode that
            // happens to have this m along the axis. Cleaner: compute
            // (2 pi / L) from one of the modes.
            // For axis x: m.kx = m * two_pi_Lx. So two_pi_Lx = m.kx / m for m != 0,
            // or alternatively L_x = global span, but we don't have it directly.
            // Simpler: pass two_pi_per_L into build via capture.
            (void)freq;
        }
    };
    (void)build;  // suppress unused-lambda warning if needed below

    // Recover two_pi_per_L_axis from the modes_ list (constructor stored them).
    // Find a mode with m_axis != 0 and infer two_pi_per_L = k_axis / m_axis.
    Real two_pi_Lx = 0, two_pi_Ly = 0, two_pi_Lz = 0;
    for (const auto& m : modes_) {
        if (two_pi_Lx == 0 && m.mx != 0) two_pi_Lx = m.kx / m.mx;
        if (two_pi_Ly == 0 && m.my != 0) two_pi_Ly = m.ky / m.my;
        if (two_pi_Lz == 0 && m.mz != 0) two_pi_Lz = m.kz / m.mz;
    }
    // Axes that have no nonzero-m mode in the half-space we enumerated:
    // fall back to one of the global frequency components via a different
    // mode; or, if still 0, leave the table at the m=0 row only.
    // (k_lo >= 1 guarantees at least one nonzero component per mode.)
    if (two_pi_Lx == 0) two_pi_Lx = 1.0;
    if (two_pi_Ly == 0) two_pi_Ly = 1.0;
    if (two_pi_Lz == 0) two_pi_Lz = 1.0;

    auto fill = [&](Real x0, Real dx, int n, Real two_pi_L,
                    std::vector<Real>& cos_t, std::vector<Real>& sin_t) {
        cos_t.assign(static_cast<std::size_t>(span) * n, 0.0);
        sin_t.assign(static_cast<std::size_t>(span) * n, 0.0);
        for (int idx = 0; idx < span; ++idx) {
            const int m = idx - M;
            const Real freq = static_cast<Real>(m) * two_pi_L;
            for (int i = 0; i < n; ++i) {
                const Real x = x0 + (i + 0.5) * dx;
                cos_t[static_cast<std::size_t>(idx) * n + i] = std::cos(freq * x);
                sin_t[static_cast<std::size_t>(idx) * n + i] = std::sin(freq * x);
            }
        }
    };
    fill(tab_x0_, tab_dx_, tab_nx_, two_pi_Lx, cos_table_x_, sin_table_x_);
    fill(tab_y0_, tab_dy_, tab_ny_, two_pi_Ly, cos_table_y_, sin_table_y_);
    fill(tab_z0_, tab_dz_, tab_nz_, two_pi_Lz, cos_table_z_, sin_table_z_);
}

void SpectralForcing::evolve_ou(Real dt) {
    // Exact OU update (Mannella 1989):
    //   a(t+dt) = a(t) * exp(-dt/T) + sigma_eq * sqrt(1 - exp(-2 dt/T)) * xi
    // with xi ~ N(0,1) per real component and sigma_eq^2 = T_corr/2.
    const Real T = p_.T_corr;
    const Real decay = std::exp(-dt / T);
    const Real diffusion = std::sqrt(p_.T_corr * (1.0 - decay * decay) / 2.0);

    std::normal_distribution<Real> nd(0.0, 1.0);
    for (auto& m : modes_) {
        const Real r1 = nd(rng_), i1 = nd(rng_);
        const Real r2 = nd(rng_), i2 = nd(rng_);
        m.a1 = decay * m.a1 + diffusion * std::complex<Real>(r1, i1);
        m.a2 = decay * m.a2 + diffusion * std::complex<Real>(r2, i2);
    }
}

void SpectralForcing::apply(State& U, const Grid& local, Real dt
#ifdef BLAST_MPI
                            , MPI_Comm comm
#endif
                            ) {
    const int nx = local.nx, ny = local.ny, nz = local.nz;
    const std::size_t N = static_cast<std::size_t>(nx) * ny * nz;

    std::vector<Real> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);

    ensure_trig_tables(local);
    const int M = p_.k_hi;
    const Real* __restrict__ cx = cos_table_x_.data();
    const Real* __restrict__ sx = sin_table_x_.data();
    const Real* __restrict__ cy = cos_table_y_.data();
    const Real* __restrict__ sy = sin_table_y_.data();
    const Real* __restrict__ cz = cos_table_z_.data();
    const Real* __restrict__ sz = sin_table_z_.data();

    // 1. Evaluate raw force field at each local cell.
    //    f(x) = sum_m 2 Re[ (a1 e1 + a2 e2) * exp(i k_m . x) ]
    //         = sum_m 2 [ Re(a1) cos(k.x) - Im(a1) sin(k.x) ] e1
    //             + 2 [ Re(a2) cos(k.x) - Im(a2) sin(k.x) ] e2
    //
    // Factorize the phase via cos/sin-sum identities, reusing per-axis trig
    // tables built once at first call. Per cell+mode this replaces 2 trig
    // calls (~60 cycles) with 4 muls and 2 adds (~10 cycles).
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                Real fxc = 0, fyc = 0, fzc = 0;
                for (const auto& m : modes_) {
                    const int idx_x = (m.mx + M) * nx + i;
                    const int idx_y = (m.my + M) * ny + j;
                    const int idx_z = (m.mz + M) * nz + k;
                    const Real cX = cx[idx_x], sX = sx[idx_x];
                    const Real cY = cy[idx_y], sY = sy[idx_y];
                    const Real cZ = cz[idx_z], sZ = sz[idx_z];
                    // cos(A+B) and sin(A+B) for A=ky*y, B=kz*z
                    const Real cYZ = cY * cZ - sY * sZ;
                    const Real sYZ = sY * cZ + cY * sZ;
                    // cos(A+B+C) and sin(A+B+C) for the full phase
                    const Real cp = cX * cYZ - sX * sYZ;
                    const Real sp = sX * cYZ + cX * sYZ;
                    const Real c1 = 2.0 * (m.a1.real() * cp - m.a1.imag() * sp);
                    const Real c2 = 2.0 * (m.a2.real() * cp - m.a2.imag() * sp);
                    fxc += c1 * m.e1[0] + c2 * m.e2[0];
                    fyc += c1 * m.e1[1] + c2 * m.e2[1];
                    fzc += c1 * m.e1[2] + c2 * m.e2[2];
                }
                const std::size_t cell = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                fx[cell] = fxc; fy[cell] = fyc; fz[cell] = fzc;
            }
        }

    auto& rho = U[RHO];
    auto& mx  = U[RHOU];
    auto& my  = U[RHOV];
    auto& mz  = U[RHOW];
    auto& en  = U[RHOE];

    // 2. Measure local <rho u . f> and <rho>.
    Real local_power = 0.0, local_mass = 0.0;
#pragma omp parallel for collapse(2) schedule(static) \
        reduction(+:local_power, local_mass)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                const Real r = rho(i, j, k);
                const Real u = mx(i, j, k) / r;
                const Real v = my(i, j, k) / r;
                const Real w = mz(i, j, k) / r;
                local_power += r * (u * fx[idx] + v * fy[idx] + w * fz[idx]);
                local_mass  += r;
            }

    Real global_power = local_power, global_mass = local_mass;
#ifdef BLAST_MPI
    if (comm != MPI_COMM_NULL) {
        Real buf[2] = {local_power, local_mass};
        Real out[2] = {0.0, 0.0};
        MPI_Allreduce(buf, out, 2, MPI_DOUBLE, MPI_SUM, comm);
        global_power = out[0];
        global_mass  = out[1];
    }
#endif

    // 3. Rescale to hit eps_target (= mean injection power per unit mass).
    Real raw_eps = global_power / std::max(global_mass, Real(1e-300));
    Real scale = 1.0;
    if (std::fabs(raw_eps) > 1e-30) scale = p_.eps_target / raw_eps;
    // Clamp on a transient zero-velocity start; once u settles, scale ~ 1.
    const Real max_scale = 1e3;
    if (!std::isfinite(scale)) scale = 0.0;
    if (std::fabs(scale) > max_scale) scale = max_scale * (scale > 0 ? 1 : -1);
    last_eps_ = raw_eps * scale;

    // 4. Apply momentum + total-energy injection.
    //    d(rho u)/dt = rho f  =>  delta(rho u) = dt * rho * f * scale
    //    d(rho E)/dt = rho u . f
#pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t idx = static_cast<std::size_t>(i)
                    + nx * (static_cast<std::size_t>(j) + ny * k);
                const Real r = rho(i, j, k);
                const Real u = mx(i, j, k) / r;
                const Real v = my(i, j, k) / r;
                const Real w = mz(i, j, k) / r;
                const Real fxs = fx[idx] * scale;
                const Real fys = fy[idx] * scale;
                const Real fzs = fz[idx] * scale;
                mx(i, j, k) += dt * r * fxs;
                my(i, j, k) += dt * r * fys;
                mz(i, j, k) += dt * r * fzs;
                en(i, j, k) += dt * r * (u * fxs + v * fys + w * fzs);
            }
}

}  // namespace blast
