#include "numerics/RK3.hpp"

#include "bc/BC.hpp"
#include "numerics/HyperdissipationSpectral.hpp"
#include "numerics/RHS.hpp"

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
               const ViscousParams& vp, Real dt) {
    auto eval_rhs = [&](const State& Uin) {
        compute_rhs_inviscid(Uin, g, eos, scratch_, k_);
        if (vp.mu > 0.0 || vp.hyper_coeff > 0.0 || vp.hyper6_coeff > 0.0)
            add_rhs_viscous(Uin, g, eos, vp, scratch_, k_);
    };

    apply_bcs(U, bc);
    eval_rhs(U);
    state_axpby(U1_, 1.0, U, dt, k_);

    apply_bcs(U1_, bc);
    eval_rhs(U1_);
    state_axpbypcz(U1_, 3.0 / 4.0, U, 1.0 / 4.0, U1_, dt / 4.0, k_);

    apply_bcs(U1_, bc);
    eval_rhs(U1_);
    state_axpbypcz(U, 1.0 / 3.0, U, 2.0 / 3.0, U1_, 2.0 / 3.0 * dt, k_);
}

void RK3::step_with_source(State& U, const Grid& g, const BCSet& bc,
                           const IdealGas& eos, const ViscousParams& vp,
                           Real dt, Real t_current, const SourceCallback& src) {
    auto eval_rhs = [&](const State& Uin, Real t_stage) {
        compute_rhs_inviscid(Uin, g, eos, scratch_, k_);
        if (vp.mu > 0.0 || vp.hyper_coeff > 0.0 || vp.hyper6_coeff > 0.0)
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
                   const Domain& d, Halo& halo) {
    auto eval_rhs = [&](State& Uin) {
        halo.exchange(Uin);
        apply_bcs(Uin, bc, d);
        compute_rhs_inviscid(Uin, g, eos, scratch_, k_);
        if (vp.mu > 0.0 || vp.hyper_coeff > 0.0 || vp.hyper6_coeff > 0.0)
            add_rhs_viscous(Uin, g, eos, vp, scratch_, k_);
    };

    eval_rhs(U);
    state_axpby(U1_, 1.0, U, dt, k_);

    eval_rhs(U1_);
    state_axpbypcz(U1_, 3.0 / 4.0, U, 1.0 / 4.0, U1_, dt / 4.0, k_);

    eval_rhs(U1_);
    state_axpbypcz(U, 1.0 / 3.0, U, 2.0 / 3.0, U1_, 2.0 / 3.0 * dt, k_);
}
#endif

}  // namespace blast
