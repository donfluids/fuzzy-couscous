#include "numerics/RK3.hpp"

#include "bc/BC.hpp"
#include "numerics/HyperdissipationSpectral.hpp"
#include "numerics/RHS.hpp"
#include "physics/Multifluid.hpp"

namespace blast {

RK3::RK3(int nx, int ny, int nz, int ng)
    : U1_(nx, ny, nz, ng), k_(nx, ny, nz, ng) {
    scratch_.allocate(nx, ny, nz, ng);
}

void RK3::step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
               Real dt) {
    ViscousParams vp{};
    vp.mu = 0.0;
    step(U, g, bc, eos, vp, dt);
}

void RK3::step(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
               const ViscousParams& vp, Real dt, const Field3D* gfn,
               const MixtureEOS* mix) {
    scratch_.disable_weno = vp.abv_disable_weno;
    scratch_.use_compact  = vp.use_compact10;
    scratch_.mf_conservative = vp.mf_conservative;
    auto eval_rhs = [&](const State& Uin) {
        compute_rhs_inviscid(Uin, g, eos, scratch_, k_, gfn, mix);
        if (vp.mu > 0.0 || vp.hyper_coeff > 0.0 || vp.hyper6_coeff > 0.0
            || vp.abv_enabled)
            add_rhs_viscous(Uin, g, eos, vp, scratch_, k_);
    };

    // Positivity floor only when running multifluid (gfn != nullptr): a strong
    // variable-gamma contact / rarefaction can overshoot to negative rho or p.
    // Single-gas runs (gfn == nullptr) skip this and stay bit-identical.
    // Floors come from vp (defaults match the historical two-gamma constants;
    // the driver scales them to the ambient state for strong-contrast JWL).
    const bool floor = (gfn != nullptr);
    const Real kRhoFloor = vp.rho_floor, kEintFloor = vp.eint_floor;

    apply_bcs(U, bc);
    eval_rhs(U);
    state_axpby(U1_, 1.0, U, dt, k_);
    if (floor) enforce_positivity(U1_, kRhoFloor, kEintFloor);

    apply_bcs(U1_, bc);
    eval_rhs(U1_);
    state_axpbypcz(U1_, 3.0 / 4.0, U, 1.0 / 4.0, U1_, dt / 4.0, k_);
    if (floor) enforce_positivity(U1_, kRhoFloor, kEintFloor);

    apply_bcs(U1_, bc);
    eval_rhs(U1_);
    state_axpbypcz(U, 1.0 / 3.0, U, 2.0 / 3.0, U1_, 2.0 / 3.0 * dt, k_);
    if (floor) enforce_positivity(U, kRhoFloor, kEintFloor);
}

void RK3::step_5eq(State& U, FiveEqAux& aux, const Grid& g, const BCSet& bc,
                   const IdealGas& eos, const ViscousParams& vp, Real dt,
                   const MixtureEOS& mix) {
    if (!aux_allocated_) {
        aux1_.allocate(U.nx(), U.ny(), U.nz(), U.ng());
        kaux_.allocate(U.nx(), U.ny(), U.nz(), U.ng());
        aux_allocated_ = true;
    }
    scratch_.disable_weno = vp.abv_disable_weno;

    // The volume fraction doubles as the gfn marker (the marker-jump WENO sensor
    // is skipped for five-equation; gfn is only used for any gfn-gated logic).
    const Field3D* gfn = &aux.a1;
    auto eval_rhs = [&](const State& Uin, const FiveEqAux& auxIn) {
        compute_rhs_inviscid(Uin, g, eos, scratch_, k_, gfn, &mix, &auxIn, &kaux_);
        if (vp.mu > 0.0 || vp.hyper_coeff > 0.0 || vp.hyper6_coeff > 0.0
            || vp.abv_enabled)
            add_rhs_viscous(Uin, g, eos, vp, scratch_, k_);
    };

    const Real rf = vp.rho_floor, ef = vp.eint_floor;
    const Real af = mix.a_floor;
    const Real zf = 1e-12;

    apply_bcs(U, bc);
    mf_fill_aux_bcs(aux, bc);
    eval_rhs(U, aux);
    state_axpby(U1_, 1.0, U, dt, k_);
    aux_axpby(aux1_, 1.0, aux, dt, kaux_);
    enforce_5eq_bounds(U1_, aux1_, af, zf);
    enforce_positivity(U1_, rf, ef);

    apply_bcs(U1_, bc);
    mf_fill_aux_bcs(aux1_, bc);
    eval_rhs(U1_, aux1_);
    state_axpbypcz(U1_, 3.0 / 4.0, U, 1.0 / 4.0, U1_, dt / 4.0, k_);
    aux_axpbypcz(aux1_, 3.0 / 4.0, aux, 1.0 / 4.0, aux1_, dt / 4.0, kaux_);
    enforce_5eq_bounds(U1_, aux1_, af, zf);
    enforce_positivity(U1_, rf, ef);

    apply_bcs(U1_, bc);
    mf_fill_aux_bcs(aux1_, bc);
    eval_rhs(U1_, aux1_);
    state_axpbypcz(U, 1.0 / 3.0, U, 2.0 / 3.0, U1_, 2.0 / 3.0 * dt, k_);
    aux_axpbypcz(aux, 1.0 / 3.0, aux, 2.0 / 3.0, aux1_, 2.0 / 3.0 * dt, kaux_);
    enforce_5eq_bounds(U, aux, af, zf);
    enforce_positivity(U, rf, ef);
}

void RK3::step_with_source(State& U, const Grid& g, const BCSet& bc,
                           const IdealGas& eos, const ViscousParams& vp,
                           Real dt, Real t_current, const SourceCallback& src) {
    scratch_.disable_weno = vp.abv_disable_weno;
    scratch_.use_compact  = vp.use_compact10;
    scratch_.mf_conservative = vp.mf_conservative;
    auto eval_rhs = [&](const State& Uin, Real t_stage) {
        compute_rhs_inviscid(Uin, g, eos, scratch_, k_);
        if (vp.mu > 0.0 || vp.hyper_coeff > 0.0 || vp.hyper6_coeff > 0.0
            || vp.abv_enabled)
            add_rhs_viscous(Uin, g, eos, vp, scratch_, k_);
        if (src) src(k_, g, t_stage);
    };

    apply_bcs(U, bc);
    eval_rhs(U, t_current);
    state_axpby(U1_, 1.0, U, dt, k_);

    apply_bcs(U1_, bc);
    eval_rhs(U1_, t_current + dt);
    state_axpbypcz(U1_, 3.0 / 4.0, U, 1.0 / 4.0, U1_, dt / 4.0, k_);

    apply_bcs(U1_, bc);
    eval_rhs(U1_, t_current + 0.5 * dt);
    state_axpbypcz(U, 1.0 / 3.0, U, 2.0 / 3.0, U1_, 2.0 / 3.0 * dt, k_);
}

#ifdef BLAST_MPI
void RK3::init_spectral_hyper_mpi(const Grid& global_grid, const Domain& d,
                                  SpectralBCMode mode) {
    scratch_.spectral_hyper =
        std::make_unique<HyperdissipationSpectralMpi>(global_grid, d, mode);
}

void RK3::step_mpi(State& U, const Grid& g, const BCSet& bc, const IdealGas& eos,
                   const ViscousParams& vp, Real dt,
                   const Domain& d, Halo& halo, const Field3D* gfn,
                   const MixtureEOS* mix) {
    scratch_.disable_weno = vp.abv_disable_weno;
    scratch_.use_compact  = vp.use_compact10;
    scratch_.mf_conservative = vp.mf_conservative;
    auto eval_rhs = [&](State& Uin) {
        halo.exchange(Uin);
        apply_bcs(Uin, bc, d);
        compute_rhs_inviscid(Uin, g, eos, scratch_, k_, gfn, mix);
        if (vp.mu > 0.0 || vp.hyper_coeff > 0.0 || vp.hyper6_coeff > 0.0
            || vp.abv_enabled)
            add_rhs_viscous(Uin, g, eos, vp, scratch_, k_);
    };

    // Positivity floor only for multifluid (gfn != nullptr): a strong
    // variable-gamma contact can overshoot to negative rho or p. Single-gas
    // runs stay bit-identical (no floor). Mirrors the serial step().
    const Real kRhoFloor = vp.rho_floor, kEintFloor = vp.eint_floor;
    const bool floor = (gfn != nullptr);

    eval_rhs(U);
    state_axpby(U1_, 1.0, U, dt, k_);
    if (floor) enforce_positivity(U1_, kRhoFloor, kEintFloor);

    eval_rhs(U1_);
    state_axpbypcz(U1_, 3.0 / 4.0, U, 1.0 / 4.0, U1_, dt / 4.0, k_);
    if (floor) enforce_positivity(U1_, kRhoFloor, kEintFloor);

    eval_rhs(U1_);
    state_axpbypcz(U, 1.0 / 3.0, U, 2.0 / 3.0, U1_, 2.0 / 3.0 * dt, k_);
    if (floor) enforce_positivity(U, kRhoFloor, kEintFloor);
}
#endif

}  // namespace blast
