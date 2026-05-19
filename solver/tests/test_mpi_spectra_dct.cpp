// DCT-based velocity spectrum test for slip-wall domains.
//
// Test setup:
//   * Domain [0, L]^3 with slip walls on every face.
//   * Density rho = rho_0 (uniform).
//   * Velocity = a single cosine/sine triple matching the symmetry the slip
//     wall implies for each component:
//
//       u(x,y,z) = U0 * sin(n_u_x * pi*x/L) * cos(n_u_y * pi*y/L) * cos(n_u_z * pi*z/L)
//       v(x,y,z) = V0 * cos(n_v_x * pi*x/L) * sin(n_v_y * pi*y/L) * cos(n_v_z * pi*z/L)
//       w(x,y,z) = W0 * cos(n_w_x * pi*x/L) * cos(n_w_y * pi*y/L) * sin(n_w_z * pi*z/L)
//
//     The sin-axis (the DST axis) is the velocity component's own normal
//     direction, so the field satisfies u_n = 0 on the matching slip walls.
//
//   * For each component we know the analytic mode wavenumber:
//       |k_u|^2 = (pi/L)^2 (n_u_x^2 + n_u_y^2 + n_u_z^2)
//     and the analytic spectral peak energy = (1/8) * 0.5 * U0^2 per mode,
//     because each component is a single basis function with unit
//     orthonormalized energy <0.5 * f^2>_volume = 0.5 * U0^2 / 8 (the 8 is
//     the cosine-orthogonality normalization 1/2 per axis when none of
//     n is zero -- and ALL our nonzero indices give 1/2 from the cos*cos
//     integral over [0,L]).
//
// Checks:
//   (1) Peak bin: round(|k_u|/(pi/L)) should hold the bulk of u's energy,
//       similarly for v, w. Spectrum sum over all 3 modes = expected.
//   (2) Parseval: sum_b E_dct(b) == <(1/2) u_i u_i>_volume computed in
//       physical space, to a few ULP * N^3.

#include "bc/BC.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "diagnostics/FFT.hpp"
#include "diagnostics/Spectra.hpp"
#include "parallel/Domain.hpp"

#include <mpi.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace blast;

namespace {

struct Mode { int n_x, n_y, n_z; double amp; };

// Pick small mode integers so the modes live within a 16^3 transform.
const Mode kU = {2, 1, 1, 1.0};   // u: sin(2 pi x/L) cos(pi y/L) cos(pi z/L)
const Mode kV = {1, 2, 1, 0.7};   // v: cos(pi x/L) sin(2 pi y/L) cos(pi z/L)
const Mode kW = {1, 1, 2, 0.4};   // w: cos(pi x/L) cos(pi y/L) sin(2 pi z/L)

void fill_field(State& U, const Grid& global,
                const std::array<long long, 3>& off) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    auto& rho  = U[RHO];
    auto& rhou = U[RHOU];
    auto& rhov = U[RHOV];
    auto& rhow = U[RHOW];
    auto& rhoE = U[RHOE];

    const double rho0 = 1.0;
    const double p0   = 1.0;
    const double gam  = 1.4;

    const double Lx = global.lx, Ly = global.ly, Lz = global.lz;
    const double pi_Lx = M_PI / Lx;
    const double pi_Ly = M_PI / Ly;
    const double pi_Lz = M_PI / Lz;

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const long long i_g = off[0] + i;
                const long long j_g = off[1] + j;
                const long long k_g = off[2] + k;
                const double x = global.x0 + (i_g + 0.5) * global.dx();
                const double y = global.y0 + (j_g + 0.5) * global.dy();
                const double z = global.z0 + (k_g + 0.5) * global.dz();

                const double u = kU.amp
                    * std::sin(kU.n_x * pi_Lx * x)
                    * std::cos(kU.n_y * pi_Ly * y)
                    * std::cos(kU.n_z * pi_Lz * z);
                const double v = kV.amp
                    * std::cos(kV.n_x * pi_Lx * x)
                    * std::sin(kV.n_y * pi_Ly * y)
                    * std::cos(kV.n_z * pi_Lz * z);
                const double w = kW.amp
                    * std::cos(kW.n_x * pi_Lx * x)
                    * std::cos(kW.n_y * pi_Ly * y)
                    * std::sin(kW.n_z * pi_Lz * z);

                rho(i, j, k)  = rho0;
                rhou(i, j, k) = rho0 * u;
                rhov(i, j, k) = rho0 * v;
                rhow(i, j, k) = rho0 * w;
                const double ke = 0.5 * rho0 * (u*u + v*v + w*w);
                rhoE(i, j, k) = p0 / (gam - 1.0) + ke;
            }
}

State make_state(const Grid& global, const Domain& d, Grid& local_out) {
    local_out = d.local_grid(global);
    State U(local_out.nx, local_out.ny, local_out.nz);
    auto off = d.global_offset(global);
    fill_field(U, global, off);
    BCSet bc; bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall; bc.zlo = bc.zhi = BCType::SlipWall;
    apply_bcs(U, bc, d);
    return U;
}

double compute_local_KE(const State& U) {
    const int nx = U.nx(), ny = U.ny(), nz = U.nz();
    const auto& rho  = U[RHO];
    const auto& rhou = U[RHOU];
    const auto& rhov = U[RHOV];
    const auto& rhow = U[RHOW];
    double s = 0.0;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double r = rho(i, j, k);
                const double uu = rhou(i, j, k) / r;
                const double vv = rhov(i, j, k) / r;
                const double ww = rhow(i, j, k) / r;
                s += 0.5 * (uu * uu + vv * vv + ww * ww);
            }
    return s;
}

}  // namespace

static int test_dct_spectrum() {
    const int N = 16;
    Grid global;
    global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = 1.0;
    global.x0 = global.y0 = global.z0 = 0.0;

    BCSet bc; bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall; bc.zlo = bc.zhi = BCType::SlipWall;
    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local;
    State U = make_state(global, d, local);

    // Build three R2R plans, one per velocity component. The DST axis is the
    // component's own (i.e. its normal direction), DCT on the others.
    R2R3DPlanMPI plan_u(N, N, N, d.comm(),
                        r2r::DCT_II, r2r::DCT_II, r2r::DST_II);  // z, y, x
    R2R3DPlanMPI plan_v(N, N, N, d.comm(),
                        r2r::DCT_II, r2r::DST_II, r2r::DCT_II);
    R2R3DPlanMPI plan_w(N, N, N, d.comm(),
                        r2r::DST_II, r2r::DCT_II, r2r::DCT_II);

    ShellSpectrum sp =
        velocity_spectrum_dct_mpi(U, global, plan_u, plan_v, plan_w, d);

    // Physical-space mean (1/2) u_i u_i for the Parseval check.
    double KE_local = compute_local_KE(U);
    double KE_global = 0.0;
    MPI_Allreduce(&KE_local, &KE_global, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    const double total_cells =
        static_cast<double>(global.nx) * global.ny * global.nz;
    const double KE_mean = KE_global / total_cells;

    // Sum the spectral bins.
    double E_sum = 0.0;
    for (std::size_t b = 0; b < sp.E.size(); ++b) E_sum += sp.E[b];

    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;

    // Expected peak bin for each mode.
    const double pi_L = M_PI / global.lx;
    auto bin_for = [&](const Mode& m) {
        const double kmag2 = pi_L * pi_L *
            (m.n_x * m.n_x + m.n_y * m.n_y + m.n_z * m.n_z);
        return static_cast<int>(std::round(std::sqrt(kmag2) / pi_L));
    };
    const int b_u = bin_for(kU);
    const int b_v = bin_for(kV);
    const int b_w = bin_for(kW);

    // Analytic per-mode energy contribution: for a basis function
    // (sin or cos) of integer index n != 0 over [0, L], <f^2>_volume = 1/2.
    // Energy per mode = 0.5 * (amp^2) * (1/2)^3 (one factor per axis).
    auto mode_energy = [](const Mode& m) {
        return 0.5 * m.amp * m.amp * (1.0 / 8.0);
    };
    const double E_u = mode_energy(kU);
    const double E_v = mode_energy(kV);
    const double E_w = mode_energy(kW);
    const double E_expected = E_u + E_v + E_w;

    if (rank == 0) {
        std::printf("[rank0] DCT spectrum: KE_mean=%.6e  Sum E_b=%.6e  "
                    "expected=%.6e  rel=%.3e\n",
                    KE_mean, E_sum, E_expected,
                    std::fabs(E_sum - KE_mean) / std::max(KE_mean, 1e-30));
        // (1) Parseval against KE in physical space.
        const double rel = std::fabs(E_sum - KE_mean) /
                           std::max(KE_mean, 1e-30);
        if (rel > 1e-10) {
            std::fprintf(stderr,
                         "Parseval failure: rel=%g > 1e-10\n", rel);
            rc = 1;
        }
        // (2) Spectrum sum matches analytic per-mode total.
        const double rel_an = std::fabs(E_sum - E_expected) /
                              std::max(E_expected, 1e-30);
        if (rel_an > 1e-10) {
            std::fprintf(stderr,
                         "analytic E sum failure: rel=%g > 1e-10\n", rel_an);
            rc = 1;
        }
        // (3) Each peak bin holds essentially all of that component's energy.
        std::printf("[rank0] peaks: b_u=%d E=%.6e (expected ~%.6e) | "
                    "b_v=%d E=%.6e (expected ~%.6e) | "
                    "b_w=%d E=%.6e (expected ~%.6e)\n",
                    b_u, (b_u < static_cast<int>(sp.E.size()) ? sp.E[b_u] : 0.0), E_u,
                    b_v, (b_v < static_cast<int>(sp.E.size()) ? sp.E[b_v] : 0.0), E_v,
                    b_w, (b_w < static_cast<int>(sp.E.size()) ? sp.E[b_w] : 0.0), E_w);
        // Each component's mode contributes >= 99.99% of its expected energy
        // into bins {b_u-1, b_u, b_u+1} (allow for rounding into adjacent bin).
        auto bin_window = [&](int b) {
            double s = 0;
            for (int bb = std::max(0, b - 1);
                 bb <= std::min(static_cast<int>(sp.E.size()) - 1, b + 1); ++bb)
                s += sp.E[bb];
            return s;
        };
        if (bin_window(b_u) < 0.9999 * E_u) rc = 1;
        if (bin_window(b_v) < 0.9999 * E_v) rc = 1;
        if (bin_window(b_w) < 0.9999 * E_w) rc = 1;
        if (rc == 0) std::printf("[OK] DCT spectrum matches analytic\n");
    }

    int rc_all = 0;
    MPI_Allreduce(&rc, &rc_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return rc_all;
}

// Verify the Helmholtz decomposition obeys energy conservation
//   K_sol + K_dil = E_total (sum of velocity_spectrum_dct_mpi bins)
// across the same mixed-mode synthetic field. Each bin's contribution is
// energy-conserving for the orthogonal decomposition only in aggregate
// (per-bin equality doesn't hold because the same |k| can come from
// different basis-index triples, so we compare TOTAL energies here).
static int test_dct_helmholtz() {
    const int N = 16;
    Grid global;
    global.nx = global.ny = global.nz = N;
    global.lx = global.ly = global.lz = 1.0;
    global.x0 = global.y0 = global.z0 = 0.0;

    BCSet bc; bc.xlo = bc.xhi = BCType::SlipWall;
    bc.ylo = bc.yhi = BCType::SlipWall; bc.zlo = bc.zhi = BCType::SlipWall;
    Domain d(MPI_COMM_WORLD, global, bc);
    Grid local;
    State U = make_state(global, d, local);

    R2R3DPlanMPI plan_u(N, N, N, d.comm(),
                        r2r::DCT_II, r2r::DCT_II, r2r::DST_II);
    R2R3DPlanMPI plan_v(N, N, N, d.comm(),
                        r2r::DCT_II, r2r::DST_II, r2r::DCT_II);
    R2R3DPlanMPI plan_w(N, N, N, d.comm(),
                        r2r::DST_II, r2r::DCT_II, r2r::DCT_II);

    ShellSpectrum sp =
        velocity_spectrum_dct_mpi(U, global, plan_u, plan_v, plan_w, d);

    // Helmholtz requires fresh plans because velocity_spectrum_dct_mpi above
    // overwrote the spectral buffers. Re-pack and re-transform inside.
    R2R3DPlanMPI plan_u2(N, N, N, d.comm(),
                         r2r::DCT_II, r2r::DCT_II, r2r::DST_II);
    R2R3DPlanMPI plan_v2(N, N, N, d.comm(),
                         r2r::DCT_II, r2r::DST_II, r2r::DCT_II);
    R2R3DPlanMPI plan_w2(N, N, N, d.comm(),
                         r2r::DST_II, r2r::DCT_II, r2r::DCT_II);
    HelmholtzResult h = helmholtz_decompose_dct_mpi(U, global, plan_u2, plan_v2,
                                                    plan_w2, d);

    double E_total = 0.0;
    for (double e : sp.E) E_total += e;

    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rc = 0;
    if (rank == 0) {
        const double K_sum = h.K_sol + h.K_dil;
        const double rel = std::fabs(K_sum - E_total) /
                           std::max(E_total, 1e-30);
        std::printf("[rank0] DCT Helmholtz: K_sol=%.6e K_dil=%.6e K_sum=%.6e  "
                    "E_total=%.6e  rel=%.3e\n",
                    h.K_sol, h.K_dil, K_sum, E_total, rel);
        if (rel > 1e-10) {
            std::fprintf(stderr,
                         "DCT Helmholtz energy conservation failure: rel=%g > 1e-10\n",
                         rel);
            rc = 1;
        }
        // Sanity: both K_sol and K_dil should be non-negative.
        if (h.K_sol < -1e-12 || h.K_dil < -1e-12) {
            std::fprintf(stderr, "negative energy: K_sol=%g K_dil=%g\n",
                         h.K_sol, h.K_dil);
            rc = 1;
        }
        if (rc == 0)
            std::printf("[OK] DCT Helmholtz conserves total energy\n");
    }

    int rc_all = 0;
    MPI_Allreduce(&rc, &rc_all, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return rc_all;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rc = 0;
    rc |= test_dct_spectrum();
    rc |= test_dct_helmholtz();
    MPI_Finalize();
    return rc;
}
