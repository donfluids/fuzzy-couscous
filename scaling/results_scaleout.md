# Strong scaling — 256³ paper Case 1

Single-node AMD EPYC 9965 (192 cores Zen5c, 12 NUMA × 16 cores), OpenMPI 4.1.6. Layout policy: **8 MPI ranks × T OpenMP threads**, where T = (total cores) / 8. The 8-rank Cartesian decomposition is the per-budget optimum identified in `results_c96.md` / `results_c192.md` / `results_shapes.md`. 30 step paper-Case-1 IC with all snapshot / spectra I/O disabled; s/step measured from the spdlog step-5 → step-30 timestamp delta on rank 0.

| Cores | Layout | s/step | Speedup vs 8c | Efficiency |
|---|---|---|---|---|
|   8 | 8×1   |  3.805 s |  1.00× | 100.0 % |
|  16 | 8×2   |  3.613 s |  1.05× |  52.7 % |
|  24 | 8×3   |  2.518 s |  1.51× |  50.4 % |
|  32 | 8×4   |  1.974 s |  1.93× |  48.2 % |
|  48 | 8×6   |  1.560 s |  2.44× |  40.7 % |
|  64 | 8×8   |  1.248 s |  3.05× |  38.1 % |
|  96 | 8×12  |  0.948 s |  4.01× |  33.5 % |
| 128 | 8×16  |  0.794 s |  4.79× |  29.9 % |
| 192 | 8×24  |  0.703 s |  5.41× |  22.5 % |


**96 → 192 cores (production regime):** 0.948 → 0.703 s/step = **1.35× speedup for 2× cores = 67 % strong-scaling efficiency**. This is the relevant number for paper runs — both points use 8 ranks with near-cubic decomposition and threads-per-rank that fill the available CCDs.

The efficiency vs the 8-core baseline is pessimistic by design: at 8×1 each rank occupies a single core in a 16-core CCD, so the rank already has the full CCD memory channel to itself. Adding threads inside the same CCD (8×2 → 8×16) buys execution units but no extra bandwidth, so the curve only steepens once T exceeds 16 and ranks start spanning CCDs.
