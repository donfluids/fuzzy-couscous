# Runs migration manifest

- Generated: 2026-05-23T06:37:07
- Status: APPLIED
- Entries: 81
- By category: blast=52, chamber=3, misc=1, scaling=11, tgv=12, tnt=2

Rollback with: `python3 postprocessing/scripts/migrate_runs.py --rollback`

## Mappings

| category | old path | new path |
|---|---|---|
| blast | `runs/out_blast_128_bhr` | `runs/blast/out_blast_128_bhr` |
| blast | `runs/out_blast_128_budget_seed1` | `runs/blast/out_blast_128_budget_seed1` |
| blast | `runs/out_blast_128_budget_seed2` | `runs/blast/out_blast_128_budget_seed2` |
| blast | `runs/out_blast_128_budget_seed3` | `runs/blast/out_blast_128_budget_seed3` |
| blast | `runs/out_blast_128_budget_seed4` | `runs/blast/out_blast_128_budget_seed4` |
| blast | `runs/out_blast_128_budget_seed5` | `runs/blast/out_blast_128_budget_seed5` |
| blast | `runs/out_blast_128_c6_ab` | `runs/blast/out_blast_128_c6_ab` |
| blast | `runs/out_blast_128_cp10_ab` | `runs/blast/out_blast_128_cp10_ab` |
| blast | `runs/out_blast_128_early_seed1` | `runs/blast/out_blast_128_early_seed1` |
| blast | `runs/out_blast_128_early_seed2` | `runs/blast/out_blast_128_early_seed2` |
| blast | `runs/out_blast_128_early_seed3` | `runs/blast/out_blast_128_early_seed3` |
| blast | `runs/out_blast_128_early_seed4` | `runs/blast/out_blast_128_early_seed4` |
| blast | `runs/out_blast_128_early_seed5` | `runs/blast/out_blast_128_early_seed5` |
| blast | `runs/out_blast_128_fineLES` | `runs/blast/out_blast_128_fineLES` |
| blast | `runs/out_blast_128_first05_seed1` | `runs/blast/out_blast_128_first05_seed1` |
| blast | `runs/out_blast_128_gauss_hyper6` | `runs/blast/out_blast_128_gauss_hyper6` |
| blast | `runs/out_blast_128_gauss_mt16` | `runs/blast/out_blast_128_gauss_mt16` |
| blast | `runs/out_blast_128_mid_seed1` | `runs/blast/out_blast_128_mid_seed1` |
| blast | `runs/out_blast_128_slip_hyper6_fd` | `runs/blast/out_blast_128_slip_hyper6_fd` |
| blast | `runs/out_blast_128_slip_hyper6_fd_long` | `runs/blast/out_blast_128_slip_hyper6_fd_long` |
| blast | `runs/out_blast_128_slip_hyper6_fd_seed1` | `runs/blast/out_blast_128_slip_hyper6_fd_seed1` |
| blast | `runs/out_blast_128_slip_hyper6_fd_seed2` | `runs/blast/out_blast_128_slip_hyper6_fd_seed2` |
| blast | `runs/out_blast_128_slip_hyper6_fd_seed3` | `runs/blast/out_blast_128_slip_hyper6_fd_seed3` |
| blast | `runs/out_blast_128_slip_hyper6_fd_seed4` | `runs/blast/out_blast_128_slip_hyper6_fd_seed4` |
| blast | `runs/out_blast_128_slip_nohyper` | `runs/blast/out_blast_128_slip_nohyper` |
| blast | `runs/out_blast_256_2fluid` | `runs/blast/out_blast_256_2fluid` |
| blast | `runs/out_blast_256_match128` | `runs/blast/out_blast_256_match128` |
| blast | `runs/out_blast_256_slip_hyper6_fd` | `runs/blast/out_blast_256_slip_hyper6_fd` |
| blast | `runs/out_blast_512_thick` | `runs/blast/out_blast_512_thick` |
| blast | `runs/out_blast_64_2fluid` | `runs/blast/out_blast_64_2fluid` |
| blast | `runs/out_blast_64_2fluid_contact` | `runs/blast/out_blast_64_2fluid_contact` |
| blast | `runs/out_blast_64_bhr` | `runs/blast/out_blast_64_bhr` |
| blast | `runs/out_blast_64_bhr_noseed` | `runs/blast/out_blast_64_bhr_noseed` |
| blast | `runs/out_blast_64_cj_stale` | `runs/blast/out_blast_64_cj_stale` |
| blast | `runs/out_blast_64_coarseLES` | `runs/blast/out_blast_64_coarseLES` |
| blast | `runs/out_blast_64_hybrid` | `runs/blast/out_blast_64_hybrid` |
| blast | `runs/out_blast_64_hybridNH` | `runs/blast/out_blast_64_hybridNH` |
| blast | `runs/out_blast_64_hybridV` | `runs/blast/out_blast_64_hybridV` |
| blast | `runs/out_blast_768_thick` | `runs/blast/out_blast_768_thick` |
| misc | `runs/out_mpi_smoke` | `runs/misc/out_mpi_smoke` |
| tgv | `runs/out_tgv128_hyper2` | `runs/tgv/out_tgv128_hyper2` |
| tgv | `runs/out_tgv128_hyper2_calib` | `runs/tgv/out_tgv128_hyper2_calib` |
| tgv | `runs/out_tgv128_hyper6_fd` | `runs/tgv/out_tgv128_hyper6_fd` |
| tgv | `runs/out_tgv128_hyper6_spectral` | `runs/tgv/out_tgv128_hyper6_spectral` |
| tgv | `runs/out_tgv256_hyper2` | `runs/tgv/out_tgv256_hyper2` |
| tgv | `runs/out_tgv256_hyper2_calib` | `runs/tgv/out_tgv256_hyper2_calib` |
| tgv | `runs/out_tgv256_hyper2_nohyper_calib` | `runs/tgv/out_tgv256_hyper2_nohyper_calib` |
| tgv | `runs/out_tgv256_hyper2_perf` | `runs/tgv/out_tgv256_hyper2_perf` |
| tgv | `runs/out_tgv64_hyper2` | `runs/tgv/out_tgv64_hyper2` |
| tgv | `runs/out_tgv64_hyper2.bak.20260517_162132` | `runs/tgv/out_tgv64_hyper2.bak.20260517_162132` |
| tgv | `runs/out_tgv64_hyper6_fd` | `runs/tgv/out_tgv64_hyper6_fd` |
| tgv | `runs/out_tgv64_hyper6_spectral` | `runs/tgv/out_tgv64_hyper6_spectral` |
| blast | `solver/out_blast_128_cj_t5` | `runs/blast/out_blast_128_cj_t5` |
| blast | `solver/out_blast_128_sg_t5` | `runs/blast/out_blast_128_sg_t5` |
| blast | `solver/out_blast_64_cj` | `runs/blast/out_blast_64_cj` |
| blast | `solver/out_blast_64_cj_inv` | `runs/blast/out_blast_64_cj_inv` |
| blast | `solver/out_blast_64_cj_long` | `runs/blast/out_blast_64_cj_long` |
| blast | `solver/out_blast_64_cj_t5` | `runs/blast/out_blast_64_cj_t5` |
| blast | `solver/out_blast_64_cj_timing` | `runs/blast/out_blast_64_cj_timing` |
| blast | `solver/out_blast_64_sg_long` | `runs/blast/out_blast_64_sg_long` |
| blast | `solver/out_blast_64_sg_t5` | `runs/blast/out_blast_64_sg_t5` |
| blast | `solver/out_blast_64_strong_sg` | `runs/blast/out_blast_64_strong_sg` |
| chamber | `solver/runs/out_sf_chamber_128` | `runs/chamber/out_sf_chamber_128` |
| chamber | `solver/runs/out_tg_chamber_128` | `runs/chamber/out_tg_chamber_128` |
| chamber | `solver/runs/out_tnt_chamber_128` | `runs/chamber/out_tnt_chamber_128` |
| tnt | `solver/runs/out_tnt_freeair_128` | `runs/tnt/out_tnt_freeair_128` |
| tnt | `solver/runs/out_tnt_smoke` | `runs/tnt/out_tnt_smoke` |
| blast | `runs/blast_256_2fluid.run.log` | `runs/blast/blast_256_2fluid.run.log` |
| blast | `runs/blast_512_thick.run.log` | `runs/blast/blast_512_thick.run.log` |
| blast | `runs/blast_768_thick.run.log` | `runs/blast/blast_768_thick.run.log` |
| scaling | `scaling/runs_c192` | `runs/scaling/runs_c192` |
| scaling | `scaling/runs_c96` | `runs/scaling/runs_c96` |
| scaling | `scaling/runs_scaleout` | `runs/scaling/runs_scaleout` |
| scaling | `scaling/runs_shapes` | `runs/scaling/runs_shapes` |
| scaling | `scaling/logs_c192` | `runs/scaling/logs_c192` |
| scaling | `scaling/logs_c96` | `runs/scaling/logs_c96` |
| scaling | `scaling/logs_scaleout` | `runs/scaling/logs_scaleout` |
| scaling | `scaling/logs_shapes` | `runs/scaling/logs_shapes` |
| scaling | `scaling/pgo_profile` | `runs/scaling/pgo_profile` |
| scaling | `scaling/pgo_train_case1_out` | `runs/scaling/pgo_train_case1_out` |
| scaling | `scaling/pgo_train_tgv64_out` | `runs/scaling/pgo_train_tgv64_out` |

## Mappings (machine-readable -- do not edit; used by --rollback)

```tsv
runs/out_blast_128_bhr	runs/blast/out_blast_128_bhr
runs/out_blast_128_budget_seed1	runs/blast/out_blast_128_budget_seed1
runs/out_blast_128_budget_seed2	runs/blast/out_blast_128_budget_seed2
runs/out_blast_128_budget_seed3	runs/blast/out_blast_128_budget_seed3
runs/out_blast_128_budget_seed4	runs/blast/out_blast_128_budget_seed4
runs/out_blast_128_budget_seed5	runs/blast/out_blast_128_budget_seed5
runs/out_blast_128_c6_ab	runs/blast/out_blast_128_c6_ab
runs/out_blast_128_cp10_ab	runs/blast/out_blast_128_cp10_ab
runs/out_blast_128_early_seed1	runs/blast/out_blast_128_early_seed1
runs/out_blast_128_early_seed2	runs/blast/out_blast_128_early_seed2
runs/out_blast_128_early_seed3	runs/blast/out_blast_128_early_seed3
runs/out_blast_128_early_seed4	runs/blast/out_blast_128_early_seed4
runs/out_blast_128_early_seed5	runs/blast/out_blast_128_early_seed5
runs/out_blast_128_fineLES	runs/blast/out_blast_128_fineLES
runs/out_blast_128_first05_seed1	runs/blast/out_blast_128_first05_seed1
runs/out_blast_128_gauss_hyper6	runs/blast/out_blast_128_gauss_hyper6
runs/out_blast_128_gauss_mt16	runs/blast/out_blast_128_gauss_mt16
runs/out_blast_128_mid_seed1	runs/blast/out_blast_128_mid_seed1
runs/out_blast_128_slip_hyper6_fd	runs/blast/out_blast_128_slip_hyper6_fd
runs/out_blast_128_slip_hyper6_fd_long	runs/blast/out_blast_128_slip_hyper6_fd_long
runs/out_blast_128_slip_hyper6_fd_seed1	runs/blast/out_blast_128_slip_hyper6_fd_seed1
runs/out_blast_128_slip_hyper6_fd_seed2	runs/blast/out_blast_128_slip_hyper6_fd_seed2
runs/out_blast_128_slip_hyper6_fd_seed3	runs/blast/out_blast_128_slip_hyper6_fd_seed3
runs/out_blast_128_slip_hyper6_fd_seed4	runs/blast/out_blast_128_slip_hyper6_fd_seed4
runs/out_blast_128_slip_nohyper	runs/blast/out_blast_128_slip_nohyper
runs/out_blast_256_2fluid	runs/blast/out_blast_256_2fluid
runs/out_blast_256_match128	runs/blast/out_blast_256_match128
runs/out_blast_256_slip_hyper6_fd	runs/blast/out_blast_256_slip_hyper6_fd
runs/out_blast_512_thick	runs/blast/out_blast_512_thick
runs/out_blast_64_2fluid	runs/blast/out_blast_64_2fluid
runs/out_blast_64_2fluid_contact	runs/blast/out_blast_64_2fluid_contact
runs/out_blast_64_bhr	runs/blast/out_blast_64_bhr
runs/out_blast_64_bhr_noseed	runs/blast/out_blast_64_bhr_noseed
runs/out_blast_64_cj_stale	runs/blast/out_blast_64_cj_stale
runs/out_blast_64_coarseLES	runs/blast/out_blast_64_coarseLES
runs/out_blast_64_hybrid	runs/blast/out_blast_64_hybrid
runs/out_blast_64_hybridNH	runs/blast/out_blast_64_hybridNH
runs/out_blast_64_hybridV	runs/blast/out_blast_64_hybridV
runs/out_blast_768_thick	runs/blast/out_blast_768_thick
runs/out_mpi_smoke	runs/misc/out_mpi_smoke
runs/out_tgv128_hyper2	runs/tgv/out_tgv128_hyper2
runs/out_tgv128_hyper2_calib	runs/tgv/out_tgv128_hyper2_calib
runs/out_tgv128_hyper6_fd	runs/tgv/out_tgv128_hyper6_fd
runs/out_tgv128_hyper6_spectral	runs/tgv/out_tgv128_hyper6_spectral
runs/out_tgv256_hyper2	runs/tgv/out_tgv256_hyper2
runs/out_tgv256_hyper2_calib	runs/tgv/out_tgv256_hyper2_calib
runs/out_tgv256_hyper2_nohyper_calib	runs/tgv/out_tgv256_hyper2_nohyper_calib
runs/out_tgv256_hyper2_perf	runs/tgv/out_tgv256_hyper2_perf
runs/out_tgv64_hyper2	runs/tgv/out_tgv64_hyper2
runs/out_tgv64_hyper2.bak.20260517_162132	runs/tgv/out_tgv64_hyper2.bak.20260517_162132
runs/out_tgv64_hyper6_fd	runs/tgv/out_tgv64_hyper6_fd
runs/out_tgv64_hyper6_spectral	runs/tgv/out_tgv64_hyper6_spectral
solver/out_blast_128_cj_t5	runs/blast/out_blast_128_cj_t5
solver/out_blast_128_sg_t5	runs/blast/out_blast_128_sg_t5
solver/out_blast_64_cj	runs/blast/out_blast_64_cj
solver/out_blast_64_cj_inv	runs/blast/out_blast_64_cj_inv
solver/out_blast_64_cj_long	runs/blast/out_blast_64_cj_long
solver/out_blast_64_cj_t5	runs/blast/out_blast_64_cj_t5
solver/out_blast_64_cj_timing	runs/blast/out_blast_64_cj_timing
solver/out_blast_64_sg_long	runs/blast/out_blast_64_sg_long
solver/out_blast_64_sg_t5	runs/blast/out_blast_64_sg_t5
solver/out_blast_64_strong_sg	runs/blast/out_blast_64_strong_sg
solver/runs/out_sf_chamber_128	runs/chamber/out_sf_chamber_128
solver/runs/out_tg_chamber_128	runs/chamber/out_tg_chamber_128
solver/runs/out_tnt_chamber_128	runs/chamber/out_tnt_chamber_128
solver/runs/out_tnt_freeair_128	runs/tnt/out_tnt_freeair_128
solver/runs/out_tnt_smoke	runs/tnt/out_tnt_smoke
runs/blast_256_2fluid.run.log	runs/blast/blast_256_2fluid.run.log
runs/blast_512_thick.run.log	runs/blast/blast_512_thick.run.log
runs/blast_768_thick.run.log	runs/blast/blast_768_thick.run.log
scaling/runs_c192	runs/scaling/runs_c192
scaling/runs_c96	runs/scaling/runs_c96
scaling/runs_scaleout	runs/scaling/runs_scaleout
scaling/runs_shapes	runs/scaling/runs_shapes
scaling/logs_c192	runs/scaling/logs_c192
scaling/logs_c96	runs/scaling/logs_c96
scaling/logs_scaleout	runs/scaling/logs_scaleout
scaling/logs_shapes	runs/scaling/logs_shapes
scaling/pgo_profile	runs/scaling/pgo_profile
scaling/pgo_train_case1_out	runs/scaling/pgo_train_case1_out
scaling/pgo_train_tgv64_out	runs/scaling/pgo_train_tgv64_out
```
