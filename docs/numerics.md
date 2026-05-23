# Numerical method

The full equations are in
[`equations/governing-equations.pdf`](equations/governing-equations.pdf); this
page is the implementation-level reference.

| Component | Implementation |
|---|---|
| Spatial discretization (inviscid) | 6th-order central / 5th-order WENO5 hybrid, flux-difference form, Lax–Friedrichs splitting (`numerics/WENO5.hpp`, `physics/EulerFlux.hpp`) |
| Shock sensor | Ducros `θ = (∇·u)²/((∇·u)² + ω² + ε)` + pressure-jump tanh ramp + half-width-2 dilation along the face direction (`numerics/Ducros.cpp`) |
| Localized artificial diffusivity | Cook/Kawai–Lele LAD (canonical `r=4`), sensor-localized μ\*/β\*/D\* for shocks and contacts (`numerics/ArtificialDiffusivity.cpp`) |
| Viscous fluxes | Stokes tensor τ_ij with optional bulk viscosity; energy flux includes τ·u + Fourier heat flux (`physics/ViscousFlux.hpp`) |
| Gradient operator | composed 6th-order central first derivatives, NGHOST = 6 (`numerics/Gradients.cpp`) |
| LES sink | `ν_h ∇⁴U` hyperdissipation on every conserved variable, `physics.hyper_coeff` knob (`numerics/HyperdissipationSpectral.cpp`) |
| Time integration | SSP-RK3 (Gottlieb–Shu), optional MMS source callback (`numerics/RK3.cpp`) |
| Boundary conditions | periodic, slip-wall (adiabatic), characteristic outflow (`bc/BC.cpp`) |
| Equation of state | ideal gas; marker-selected mixture (two-γ or JWL) (`physics/MixtureEOS.hpp`) |

## Conservation by construction

Every flux term is a difference of face fluxes,
`R_i -= (F_{i+1/2} − F_{i−1/2})/Δx` (`numerics/RHS.cpp`). A face value computed
from identical inputs cancels bit-for-bit when summed over the domain (discrete
divergence theorem), so totals are conserved independent of reconstruction order;
only physical boundary faces survive, and at a slip wall `u·n = 0` zeroes the
mass and energy fluxes. The 6th-order truncation error lives in the *solution*,
not the totals — see the conservation-vs-truncation discussion in the top-level
`README.md` and [`diagnostics.md`](diagnostics.md#conservation-monitors).

## TGV at Re=1600 on 64³ — Kolmogorov recovery

`solver/examples/tgv_re1600_64_hyper2.toml` runs the Taylor–Green vortex to
t = 12 with `ν = 1/1600`, `ν_h = 2.5e-5`. The shell-averaged spectrum develops a
clean `k^(−5/3)` inertial range in `k ∈ [4,10]` for `t ≥ 8` (local slope −1.5 to
−1.7); tke decays 0.125 → 0.043, peak dissipation at t ≈ 9, with
`ε_dil/ε_sol < 10⁻⁵` (purely solenoidal, as the TGV symmetry class requires).

The 64³ hyperdissipation coefficient (`2.5e-5`) is `Δx²`-scaled from the 32³
value (`1e-4`), **not** `Δx⁴`: empirically the stability invariant is fixed
`ν_eff = ν_h k_max²`, while matching the dissipation rate at Nyquist lets the
cascade outpace the sink.

## Test suite

44 serial + 4 MPI binaries (× 3 rank counts), all passing. Highlights:

| Suite | Verifies |
|---|---|
| AdvectSmooth, Sod1D, Sedov3D, ViscousMMS | RK3 3rd-order convergence; exact Riemann; analytic shock radius; 6th-order viscous Laplacian |
| MMS3D (3) | end-to-end Navier–Stokes accuracy (entropy wave, YSD vortex, viscous-NS source) on 32³→48³→64³ |
| Hyperdissipation (3) | discrete `∇⁴` matches analytic `k⁴`; SSP-RK3 preserves `exp(−λt)`; zero-coeff bit-exact |
| Spectrum, Helmholtz, Dissipation | Parseval; single-mode binning; pure-solenoidal/irrotational projection; M_t / ε_sol / ε_dil identities |
| SlipWall (2), ShuOsher1D (2), CJDetonation | mass+energy conservation & acoustic round-trip; post-shock oscillations & refinement; exact CJ relations |
| MPI (4 binaries) | halo exchange; serial↔MPI bit-exactness (max |Δ| = 0); collective restart round-trip; distributed FFTW3-MPI spectra cross-check |

Run: `ctest --test-dir solver/build --output-on-failure` (serial) and
`solver/run_mpi.sh ctest --test-dir solver/build_mpi --output-on-failure` (MPI).
On the 9965 the suite runs in ~107 s; on a 4-core sandbox ~95 min (dominated by
MMS3D + SlipWall).
