# Strong scaling — paper Case 1, 256³, fixed 192 cores

Single-node AMD EPYC 9965, cores 0..191 (= 12 of 12 NUMA nodes, 16 cores per node), OpenMPI 4.1.6, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`. Each row holds **R × T = 192 total physical cores** — partitioning study (best split of a fixed core budget between MPI ranks and OpenMP threads). 30-step paper-Case-1 IC (`scaling_case1_256.toml`) with all snapshot / spectra I/O disabled. s/step measured from the spdlog step-5 → step-30 timestamp delta on rank 0.

| Ranks × Threads | Wall (30 steps) | s/step | vs 1×192 | vs best |
|---|---|---|---|---|
|   1 × 192   |   41.49 s |  1.342 s |  1.00× |  53.4 % |
|   2 × 96    |   31.69 s |  0.977 s |  1.37× |  73.4 % |
|   4 × 48    |   24.91 s |  0.790 s |  1.70× |  90.7 % |
|   6 × 32    |   24.29 s |  0.765 s |  1.75× |  93.6 % |
|   8 × 24  ★ |   22.73 s |  0.717 s |  1.87× | 100.0 % |
|  12 × 16    |   23.32 s |  0.728 s |  1.84× |  98.4 % |
|  16 × 12    |   23.58 s |  0.732 s |  1.83× |  98.0 % |
|  24 × 8     |   23.77 s |  0.737 s |  1.82× |  97.2 % |
|  48 × 4     |   25.03 s |  0.765 s |  1.75× |  93.7 % |
|  96 × 2     |   28.07 s |  0.835 s |  1.61× |  85.8 % |
| 192 × 1     |   35.19 s |  1.011 s |  1.33× |  70.9 % |

**Optimum: 8 ranks × 24 threads** at 0.717 s/step.
