# Diagnostics

The diagnostics are the core paper-revision deliverable. They are logged every
`stats_every` steps to a stats CSV; the spectra-HDF5 dump optionally appends
shell-averaged `E(k)` and the Helmholtz split `E_sol(k)` / `E_dil(k)` per step.
Implementation: `diagnostics/Statistics.cpp`, `diagnostics/Spectra.cpp`.

## Logged quantities

- **Velocity / Reynolds stats:** `u_rms`, `tke`, `ke_total`,
  `M_t = u_rms/⟨c⟩`, `c_mean`, `T_mean`, `rho_mean`, `p_mean` (with mean removal,
  so tke vs. EKE are both reported).
- **Dissipation budget:**
  `ε_total = (μ/ρ)⟨τ_ij ∂u_i/∂x_j⟩`, `ε_sol = ν⟨|ω|²⟩`,
  `ε_dil = (4/3)ν⟨(∇·u)²⟩`; enstrophy `⟨|ω|²⟩` and `⟨(∇·u)²⟩` independently.
- **Helmholtz split:** solenoidal/dilatational kinetic-energy split plus per-shell
  spectra. The dilatational field is further separable into pseudo-sound
  (slaved to vortical turbulence) and true sound (free/standing acoustics) via a
  pressure-Poisson split — see `postprocessing/scripts/pseudosound_split.py`.
- **Conservation monitors:** total energy `e_total = ⟨ρE⟩` and `E_ratio = E/E0`;
  total mass via `rho_mean` and `M_ratio = M/M0`; total momentum `mom_x/y/z`
  reported as `|p|/(ρc)`. `e_int` is accumulated directly from the field
  (`Σ(ρE − ½ρ|u|²)`), so `KE + e_int = e_total` is a real consistency check.

## Conservation monitors: what they do and do not measure

For a closed chamber (slip walls, no forcing) `E_ratio` and `M_ratio` sit at
`1.0` to round-off (`~1e-15`). This is **not** an accuracy statement — it is a
discrete *conservation* identity, a different axis from *truncation error*:

- Fluxes are differences of face fluxes, so a face value cancels bit-for-bit when
  summed over the domain (discrete divergence theorem); only boundary faces
  survive, and at a slip wall `u·n = 0` zeroes the mass/energy flux. Totals are
  conserved independent of reconstruction order.
- The 6th-order truncation error lives in the *solution* (pointwise ρ, velocity,
  shock position, spectrum), not the totals. To see it, use the MMS convergence
  tests, which measure the L2 solution error and its 6th-order decay.
- Momentum is the diagnostic contrast: at a slip wall the wall flux carries the
  pressure term `p·n`, which the telescope does not cancel, so total momentum is
  conserved only when wall-pressure impulses cancel by symmetry. An off-center
  blast pins `E_ratio`/`M_ratio` at 1 while `|p|/ρc` climbs.

So the monitors are a **structural / regression check**, not an accuracy metric:
round-off drift is positive evidence the scheme is in conservative flux form and
the slip-wall BC is flux-consistent; drift above the floor flags a real defect
(non-conservative source, broken operator split, boundary leakage, instability).

## Reviewer-concern mapping

| Concern | Code path |
|---|---|
| M1 effective Reynolds | `dissipation_budget` separates resolved viscous ε; numerical sink via `hyper_coeff` |
| M2 grid convergence | re-run any example at varying `nx` |
| M3 decay exponent + CI | `postprocessing/tools/fit_decay.py` (2000-sample bootstrap) |
| M4 solenoidal/dilatational | `helmholtz_decompose` + per-step `K_sol`/`K_dil`/`ε_sol`/`ε_dil` |
| M5 / M8 tke vs EKE | `velocity_stats` reports both with mean removal |
| M6 ensembles | `ensemble_seed` + multi-mode random IC + `postprocessing/tools/ensemble_average.py` |
| M7 early-time sanity | configurable `snapshot_every`; pre/post first-reflection capture |
| M9 CJ initial condition | `ic_cj_detonation_3d` (Williams exact relations, validated to 1e-6) |
