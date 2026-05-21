#pragma once

#include "core/Config.hpp"     // BCSet
#include "core/Field3D.hpp"
#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"

#include <array>

namespace blast {

// BHR (Besnard-Harlow-Rauenzahn) variable-density k-eps-a-b model, 3D Cartesian,
// run as an operator-split sub-step alongside the compressible (LES) hydro.
// Ported from scripts/bhr_rans_1d.py (spherical) to 3D. The turbulent mass flux
// a_i and density-specific-volume covariance b are the variable-density closure
// variables; the dominant TKE source is the variable-density production
// P_VD = -a_i d p/dx_i (validated dominant in the LES a/b budget).
struct BHRParams {
    bool enabled  = false;
    bool feedback = false;     // two-way: add Reynolds stress + dissipative heat to U
    Real C_mu = 0.09, C_e1 = 1.44, C_e2 = 1.92;
    Real C_a  = 1.0,  C_b  = 1.0;
    Real sigma_k = 1.0, sigma_e = 1.3, sigma_a = 1.0, sigma_b = 1.0;
    Real mu_phys = 5.0e-4;     // molecular viscosity (match physics.mu)
    Real k_floor = 1e-10, e_floor = 1e-12, b_max = 2.0, k_max = 50.0;
    Real L_max = 0.5, prod_limit = 10.0, pt_frac_max = 0.5;
    Real seed_scale = 1e-3, b_seed = 0.0;   // IC seeding at the contact layer
};

// Six turbulence primitives on the same grid + ghost layer as State:
// k, eps, a_x, a_y, a_z, b.
class TurbState {
public:
    enum { TK = 0, TE = 1, TAX = 2, TAY = 3, TAZ = 4, TB = 5, NT = 6 };
    TurbState() = default;
    TurbState(int nx, int ny, int nz, int ng = NGHOST) { allocate(nx, ny, nz, ng); }
    void allocate(int nx, int ny, int nz, int ng = NGHOST) {
        static constexpr const char* names[NT] =
            {"tke", "teps", "a_x", "a_y", "a_z", "bvar"};
        for (int v = 0; v < NT; ++v) {
            f_[v].resize(nx, ny, nz, ng);
            f_[v].set_name(names[v]);
        }
    }
    Field3D&       operator[](int v)       { return f_[v]; }
    const Field3D& operator[](int v) const { return f_[v]; }
    int nx() const { return f_[0].nx(); }
    int ny() const { return f_[0].ny(); }
    int nz() const { return f_[0].nz(); }
    int ng() const { return f_[0].ng(); }
private:
    std::array<Field3D, TurbState::NT> f_;
};

// Seed k (and optionally b) at the contact layer (large |grad rho|); eps from k;
// a = 0. Mirrors bhr_rans_1d.py init.
void bhr_init(TurbState& T, const State& U, const Grid& g,
              const IdealGas& eos, const BHRParams& bp);

// Ghost fill: slip-wall -> Neumann (zero-gradient) for k,eps,b; the normal
// component of a_i is mirrored with sign flip, tangential components zero-grad.
// Periodic -> wrap. Outflow -> zero-order extrapolation.
void bhr_apply_bcs(TurbState& T, const BCSet& bc);

// One operator-split BHR sub-step: advect (upwind) + diffuse + BHR sources
// (point-implicit destruction) on T, using mean gradients from U. If
// bp.feedback, also adds the Reynolds-stress divergence and dissipative heating
// to U. Caller must have filled U's ghosts (apply_bcs) before this call.
void bhr_substep(State& U, TurbState& T, const Grid& g, const BCSet& bc,
                 const IdealGas& eos, const BHRParams& bp, Real dt);

// Volume-integrated turbulent kinetic energy <rho k> for diagnostics.
Real bhr_tke_integral(const State& U, const TurbState& T, const Grid& g);

// Max turbulent kinematic viscosity nu_t = C_mu k^2/eps / ... over the interior,
// for the turbulent-diffusion CFL guard when feedback is on.
Real bhr_max_nu_t(const State& U, const TurbState& T, const BHRParams& bp);

// Interior peak magnitudes (k, |a|, b) for diagnostics.
void bhr_peaks(const TurbState& T, Real& kmax, Real& amax, Real& bmax);

// Write spherically-binned radial profiles k(r), a_r(r)=a.rhat, b(r) to a CSV
// (columns: r,k,a_r,b) for validation against dns_radial_profiles.npz.
void bhr_write_radial_profiles(const State& U, const TurbState& T, const Grid& g,
                               const std::string& path, Real t);

// Resolution-adequacy blend f_k = k_sgs/(k_res + k_sgs), the LES/RANS sensor:
//   k_res = 1/2 |u - test_filter(u)|^2  (resolved energy in the [dx,2dx] octave)
//   k_sgs = the BHR modeled k.
// f_k -> 1 where modeled k dominates (under-resolved -> RANS); f_k -> 0 where
// resolved energy dominates (-> LES). Writes radial profile CSV
// (columns: r,f_k,k_res,k_sgs) and returns the volume-mean f_k.
Real bhr_write_sensor_profiles(const State& U, const TurbState& T, const Grid& g,
                               const std::string& path, Real t);

}  // namespace blast
