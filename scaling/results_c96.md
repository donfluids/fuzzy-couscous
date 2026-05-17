# Strong scaling — paper Case 1, 256³, fixed 96 cores

Single-node AMD EPYC 9965, cores 0..95 (= 6 of 12 NUMA nodes, 16 cores per node), OpenMPI 4.1.6, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`. Each row holds **R × T = 96 total physical cores** — partitioning study (best split of a fixed core budget between MPI ranks and OpenMP threads). 30-step paper-Case-1 IC (`scaling_case1_256.toml`) with all snapshot / spectra I/O disabled. s/step measured from the spdlog step-5 → step-30 timestamp delta on rank 0.

| Ranks × Threads | Wall (30 steps) | s/step | vs 1×96 | vs best |
|---|---|---|---|---|
|   1 × 96    |   43.05 s |  1.395 s |  1.00× |  67.0 % |
|   2 × 48    |   36.54 s |  1.140 s |  1.22× |  81.9 % |
|   4 × 24    |   31.73 s |  1.018 s |  1.37× |  91.7 % |
|   6 × 16    |   30.44 s |  0.975 s |  1.43× |  95.8 % |
|   8 × 12  ★ |   29.22 s |  0.934 s |  1.49× | 100.0 % |
|  12 × 8     |   30.21 s |  0.961 s |  1.45× |  97.2 % |
|  16 × 6     |   29.97 s |  0.949 s |  1.47× |  98.5 % |
|  24 × 4     |   30.65 s |  0.968 s |  1.44× |  96.5 % |
|  48 × 2     |   31.97 s |  0.992 s |  1.41× |  94.2 % |
|  96 × 1     |   35.15 s |  1.070 s |  1.30× |  87.3 % |

**Optimum: 8 ranks × 12 threads** at 0.934 s/step.
