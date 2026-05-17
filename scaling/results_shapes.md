# Decomposition-shape sweep — 192-core paper Case 1, 256³

Same conditions as the partitioning sweeps. Within each rank tier we override `MPI_Dims_create` via `BLAST_DIMS=PX,PY,PZ`. **Overall best: R=8, T=24, dims=1,4,2 (squat_yz) at 0.708 s/step.**

### R = 8, T = 24  (default = 2,2,2)

| Decomposition | tag | s/step | vs default | vs best |
|---|---|---|---|---|
| 1,4,2 ★ | squat_yz | 0.708 s | -0.8 % | 100.0 % |
| 2,2,2 | default | 0.714 s | +0.0 % | 99.2 % |
| 4,2,1 | squat | 0.766 s | +7.3 % | 92.5 % |
| 1,1,8 | slab_z | 0.789 s | +10.5 % | 89.8 % |
| 8,1,1 | slab_x | 0.904 s | +26.6 % | 78.4 % |

### R = 12, T = 16  (default = 3,2,2)

| Decomposition | tag | s/step | vs default | vs best |
|---|---|---|---|---|
| 3,2,2 ★ | default | 0.726 s | +0.0 % | 100.0 % |
| 4,3,1 | squat | 0.732 s | +0.9 % | 99.1 % |
| 6,2,1 | squat2 | 0.762 s | +5.0 % | 95.2 % |
| 12,1,1 | slab_x | 1.069 s | +47.3 % | 67.9 % |

### R = 24, T = 8  (default = 4,3,2)

| Decomposition | tag | s/step | vs default | vs best |
|---|---|---|---|---|
| 4,3,2 ★ | default | 0.737 s | +0.0 % | 100.0 % |
| 6,2,2 | squat | 0.756 s | +2.5 % | 97.5 % |
| 8,3,1 | pencil | 0.861 s | +16.8 % | 85.6 % |
| 24,1,1 | slab | 1.247 s | +69.2 % | 59.1 % |
