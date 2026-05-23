#!/usr/bin/env bash
# True strong-scaling sweep: fixed 256^3 paper Case 1, layout policy is
# always 8 MPI ranks × T OpenMP threads where T = cores/8. The 8-rank
# decomposition is the per-budget best from the partitioning + shape
# studies, so this curve isolates how throughput scales with total cores.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$REPO_DIR/solver/build_mpi/blast_les_mpi"
CFG="$REPO_DIR/solver/examples/scaling_case1_256.toml"
RUN_ROOT="$REPO_DIR/runs/scaling/runs_scaleout"
LOG_ROOT="$REPO_DIR/runs/scaling/logs_scaleout"
SUMMARY="$SCRIPT_DIR/scaleout.tsv"

MPIRUN="/usr/bin/mpirun.openmpi"
RUN_LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/local/lib"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

# Total cores per row. R = 8 throughout; T = cores / 8.
BUDGETS=(8 16 24 32 48 64 96 128 192)

printf 'cores\tranks\tthreads\twall_total_s\twall_per_step_s\tsteps\n' > "$SUMMARY"

for cores in "${BUDGETS[@]}"; do
    R=8
    T=$(( cores / R ))
    tag="c${cores}_r${R}_t${T}"
    out_dir="$RUN_ROOT/$tag"
    log_file="$LOG_ROOT/${tag}.log"

    rm -rf "$out_dir"; mkdir -p "$out_dir"
    run_cfg="$out_dir/scaling_case1_256.toml"
    sed "s|out_dir          = .*|out_dir          = \"$out_dir\"|" "$CFG" > "$run_cfg"

    echo "=== $tag :: cores=$cores R=$R T=$T ==="

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
        printf '%d\t%d\t%d\t%s\tFAIL\tFAIL\n' "$cores" "$R" "$T" "$wall" >> "$SUMMARY"
        continue
    fi

    perstep=$(python3 "$REPO_DIR/postprocessing/scripts/scaling/parse_steps.py" --window 5 "$log_file")
    nsteps=$(sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$log_file" \
             | grep -E '^\[[0-9:.]+\] \[info\] step ' \
             | tail -1 | awk '{print $4}')

    echo "  wall_total=${wall}s  wall/step=${perstep}s  last_step=${nsteps}"
    printf '%d\t%d\t%d\t%s\t%s\t%s\n' "$cores" "$R" "$T" "$wall" "$perstep" "$nsteps" >> "$SUMMARY"
done

echo
echo "Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
