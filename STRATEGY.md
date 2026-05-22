---
name: fuzzy-couscous
last_updated: 2026-05-21
---

# fuzzy-couscous Strategy

## Target problem

The blast-turbulence claims in *Daniel (2025), "Blast-induced turbulence in closed
chambers,"* have to be defended against peer review (M1–M9) — but the original code
no longer exists, so the supporting physics and numerics must be rebuilt clean-room.
The claims are only as strong as the least-accurate component, and the work now
reaches beyond the original paper into multi-fluid and BHR-modeled regimes.

## Our approach

Be as accurate as possible in every aspect: a clean-room solver carrying all the
essential physics and numerics to support the claims, treating no component as
secondary — verification, the physics/numerics dissipation split, multi-fluid, and
BHR modeling each have to stand on their own. The code is the evidence, so fidelity
everywhere — not a single headline feature — is the bet.

## Who it's for

**Primary:** Researchers in the broader blast/shock-turbulence community — hiring
fuzzy-couscous to get a verifiable, reproducible solver and validation-grade evidence
they can trust and build their own work on.

**Secondary:** Industry and small businesses doing blast/shock modeling — hiring it
for defensible, ready-to-apply modeling capability and results without standing up
their own clean-room solver.

## Key metrics

- **Verification convergence fidelity** — observed order of accuracy vs. theoretical
  across the suite (RK3 3rd-order, viscous 6th-order, `∇⁴`↔`k⁴`), plus bit-exact MPI
  (max |Δ| = 0). Source: ctest.
- **Validation error vs. references** — L1/L2 against canonical cases: Sod, Sedov
  radius, TGV `k^−5/3` inertial slope, Shu–Osher self-convergence. Source: per-case
  output / tools.
- **Physics/numerics attribution quality** — how cleanly the numerical LES sink is
  quantified vs. physical dissipation (effective-Re error, `ε_sol`/`ε_dil`
  identities). Source: `dissipation_budget`.
- **Decay-exponent uncertainty** — width of the bootstrap CI on the turbulence decay
  exponent (M3). Source: `tools/fit_decay.py`.
- **Regime coverage with tests** — fraction of target regimes (single/multi-fluid,
  BHR, CJ detonation, closed-chamber) backed by a passing verification and validation
  case. Source: ctest + examples.
- **Experimental-data comparison error** — discrepancy between simulated diagnostics
  and the manuscript's reference experimental measurements; built up and refined as
  the comparison dataset matures. Source: manuscript experimental data (TBD).

## Tracks

### Core solver fidelity

The numerics + base physics engine: hybrid 6th-order central / WENO5, Ducros shock
capture, viscous fluxes, SSP-RK3, and the `ν_h ∇⁴ U` LES sink.

_Why it serves the approach:_ every claim rides on the base solver being right
component-by-component.

### Physics scope extension (multi-fluid → BHR)

Multi-fluid capability to generate the variable-density physics data, with BHR
turbulence modeling built on top of it — both beyond the original paper.

_Why it serves the approach:_ the work reaches regimes the manuscript never covered,
and BHR is only as good as the multi-fluid data it relies on.

### Verification & validation

The test battery (MMS, bit-exact MPI, canonical cases) plus the experimental
cross-check.

_Why it serves the approach:_ "no weak component" is only true if it's continuously
proven; this is where most of the metrics live.

### Diagnostics & community delivery

The reviewer-concern diagnostics (dissipation budget, Helmholtz split, decay fits,
ensembles) plus reproducible, documented, scalable delivery so the research community
and industry can run it.

_Why it serves the approach:_ the code is the evidence only if it produces defensible
diagnostics others can reproduce.
