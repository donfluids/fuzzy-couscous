#pragma once

#include "core/Grid.hpp"
#include "core/State.hpp"
#include "physics/EOS.hpp"
#include "physics/ViscousFlux.hpp"

#ifdef BLAST_MPI
#include <mpi.h>
#endif

namespace blast {

struct VelocityStats {
    Real u_mean[3];      // volume-averaged velocity
    Real mom[3];         // < rho u_i >  (mean momentum density). Total momentum
                         // = mom * volume is conserved exactly for periodic BCs
                         // (no boundary force); for slip walls it tracks the net
                         // wall pressure impulse (~0 by symmetry for a centered
                         // blast), so its drift is a symmetry / impulse monitor.
    Real u_rms;          // sqrt( <(u_i - <u_i>)^2> )
    Real ke_total;       // < (1/2) rho |u|^2 >        (mean kinetic energy density)
    Real tke;            // < (1/2) rho |u'|^2 >  with mean removed (Reynolds)
    Real M_t;            // u_rms / <c>
    Real c_mean;         // < sound speed >
    Real T_mean;
    Real rho_mean;
    Real p_mean;
    // Total energy conservation monitor. e_total = < rho E > is the mean total
    // energy density; with a fixed domain volume it is conserved (constant in
    // time) for a closed chamber with no forcing -- viscous / WENO / LAD
    // dissipation only converts kinetic to internal, leaving e_total fixed.
    Real e_total;        // < rho E >                  (kinetic + internal)
    Real e_int;          // < rho e_int > = e_total - ke_total
};

VelocityStats velocity_stats(const State& U, const IdealGas& eos);

struct DissipationBudget {
    // All quantities are volume-averaged dissipation rates per unit mass
    // (i.e. units of velocity^2 / time).
    Real eps_total;  // mu/rho * < tau_ij d u_i/d x_j > (full tensor form)
    Real eps_sol;    // nu * < |omega|^2 >   (solenoidal / enstrophy-based)
    Real eps_dil;    // (4/3) * nu * < (div u)^2 >
    Real omega2_mean;
    Real div2_mean;
};

// Computes resolved dissipation budget from velocity gradients via 6th-order
// stencils. With constant mu (Stokes form), eps_total = eps_sol + eps_dil
// exactly on smooth incompressible parts and tracks the full dissipation
// on the rest. Requires NGHOST >= 6 (we use 6 throughout).
DissipationBudget dissipation_budget(const State& U, const Grid& g,
                                     const IdealGas& eos, const ViscousParams& vp);

#ifdef BLAST_MPI
// MPI variants: take the GLOBAL grid extent so means are divided by global
// cell count; reduce per-component sums with MPI_Allreduce on `comm`.
VelocityStats velocity_stats(const State& U, const IdealGas& eos,
                             long long N_global, MPI_Comm comm);
DissipationBudget dissipation_budget(const State& U, const Grid& g,
                                     const IdealGas& eos,
                                     const ViscousParams& vp,
                                     long long N_global, MPI_Comm comm);
#endif

}  // namespace blast
