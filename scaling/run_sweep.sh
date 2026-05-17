#!/usr/bin/env bash
# Strong-scaling sweep for blast_les_mpi on the 192-core EPYC 9965.
# Fixed total core budget = 96 (cores 0..95, NUMA nodes 0..5). 256^3 paper
# Case 1, 30 steps, all snapshot/spectra I/O disabled. The 10 (R, T) points
# always satisfy R*T = 96.
#
# Pinning: --map-by slot:PE=$T --bind-to core assigns T contiguous physical
# cores per rank starting at core 0; OMP_PROC_BIND=close + OMP_PLACES=cores
# keeps the OpenMP threads on those same cores.

set -euo pipefail

TOTAL_CORES="${1:-96}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$REPO_DIR/solver/build_mpi/blast_les_mpi"
CFG="$REPO_DIR/solver/examples/scaling_case1_256.toml"
RUN_ROOT="$SCRIPT_DIR/runs_c${TOTAL_CORES}"
LOG_ROOT="$SCRIPT_DIR/logs_c${TOTAL_CORES}"
SUMMARY="$SCRIPT_DIR/sweep_summary_c${TOTAL_CORES}.tsv"

# The MPI binary links libmpi_cxx (OpenMPI 4 only) via libhdf5_openmpi, so we
# must use the system OpenMPI 4 runtime, not the OpenMPI 5 in /opt. /usr/local
# stays on the search path so the multi-threaded FFTW3 (fftw_taint et al.) is
# still found.
MPIRUN="/usr/bin/mpirun.openmpi"
RUN_LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/local/lib"

if [[ ! -x "$BIN" ]]; then
    echo "binary not found: $BIN" >&2
    exit 1
fi
if [[ ! -f "$CFG" ]]; then
    echo "config not found: $CFG" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

# (ranks, threads) pairs with ranks*threads = TOTAL_CORES. NUMA topology is
# 12 nodes × 16 cores; the rank counts below give clean divisibility and
# include both NUMA-aligned and CCD-split layouts.
case "$TOTAL_CORES" in
    96)
        CONFIGS=("1 96" "2 48" "4 24" "6 16" "8 12" "12 8" "16 6" "24 4" "48 2" "96 1")
        ;;
    192)
        CONFIGS=("1 192" "2 96" "4 48" "6 32" "8 24" "12 16" "16 12" "24 8" "48 4" "96 2" "192 1")
        ;;
    *)
        echo "no CONFIGS table for TOTAL_CORES=$TOTAL_CORES" >&2
        exit 2
        ;;
esac

printf 'ranks\tthreads\twall_total_s\twall_per_step_s\tsteps_logged\n' > "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
    read -r R T <<<"$cfg"
    tag="r${R}_t${T}"
    out_dir="$RUN_ROOT/$tag"
    log_file="$LOG_ROOT/${tag}.log"

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    # Per-run TOML clone so out_dir is unique.
    run_cfg="$out_dir/scaling_case1_256.toml"
    sed "s|out_dir          = .*|out_dir          = \"$out_dir\"|" "$CFG" > "$run_cfg"

    echo "=== $tag :: ranks=$R threads=$T ==="

    t0=$(date +%s.%N)
    LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" "$MPIRUN" -n "$R" \
        --map-by "slot:PE=$T" \
        --bind-to core \
        -x OMP_NUM_THREADS="$T" \
        -x OMP_PROC_BIND=close \
        -x OMP_PLACES=cores \
        -x LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" \
        "$BIN" "$run_cfg" > "$log_file" 2>&1
    rc=$?
    t1=$(date +%s.%N)

    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')

    if [[ $rc -ne 0 ]]; then
        echo "  FAILED (rc=$rc); see $log_file" >&2
        printf '%d\t%d\t%s\tFAIL\tFAIL\n' "$R" "$T" "$wall" >> "$SUMMARY"
        continue
    fi

    # Parse last step number + step-rate from log timestamps.
    # spdlog pattern is [HH:MM:SS.mmm]; we time step 5 -> last step.
    # sed strips ANSI color codes from %^/%$ in the spdlog pattern.
    perstep=$(python3 "$SCRIPT_DIR/parse_steps.py" --window 5 "$log_file")
    nsteps=$(sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$log_file" \
             | grep -E '^\[[0-9:.]+\] \[info\] step ' \
             | tail -1 | awk '{print $4}')

    echo "  wall_total=${wall}s  wall/step=${perstep}s  last_step=${nsteps}"
    printf '%d\t%d\t%s\t%s\t%s\n' "$R" "$T" "$wall" "$perstep" "$nsteps" >> "$SUMMARY"
done

echo
echo "Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
