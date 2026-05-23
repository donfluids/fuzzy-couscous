#!/usr/bin/env bash
# Decomposition-shape sweep: fix total cores = 192, vary the 3D Cartesian
# decomposition shape via BLAST_DIMS within each (R, T) tier. Goal is to see
# whether forcing slab / pencil layouts beats MPI_Dims_create's near-cubic
# default at 256^3.
#
# Configs: R T DIMS  (DIMS = PX,PY,PZ such that PX*PY*PZ = R)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$REPO_DIR/solver/build_mpi/blast_les_mpi"
CFG="$REPO_DIR/solver/examples/scaling_case1_256.toml"
RUN_ROOT="$REPO_DIR/runs/scaling/runs_shapes"
LOG_ROOT="$REPO_DIR/runs/scaling/logs_shapes"
SUMMARY="$SCRIPT_DIR/sweep_shapes.tsv"

MPIRUN="/usr/bin/mpirun.openmpi"
RUN_LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/local/lib"

if [[ ! -x "$BIN" || ! -f "$CFG" ]]; then
    echo "binary or config missing" >&2; exit 1
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

# (R, T, DIMS, tag). T = 192/R throughout.
CONFIGS=(
    # R=8 -- the rank-count optimum from the 192-core partitioning sweep.
    "8  24  2,2,2  default"
    "8  24  4,2,1  squat"
    "8  24  1,4,2  squat_yz"
    "8  24  8,1,1  slab_x"
    "8  24  1,1,8  slab_z"

    # R=12 -- 1-NUMA-per-rank alternative.
    "12 16  3,2,2  default"
    "12 16  4,3,1  squat"
    "12 16  6,2,1  squat2"
    "12 16  12,1,1 slab_x"

    # R=24 -- a higher rank tier to test how shape matters when the subgrid
    # per rank gets smaller.
    "24 8   4,3,2  default"
    "24 8   6,2,2  squat"
    "24 8   8,3,1  pencil"
    "24 8   24,1,1 slab"
)

printf 'ranks\tthreads\tdims\ttag\twall_total_s\twall_per_step_s\tsteps\n' > "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
    read -r R T DIMS TAG <<<"$cfg"
    safe_dims="${DIMS//,/x}"
    name="r${R}_t${T}_${safe_dims}_${TAG}"
    out_dir="$RUN_ROOT/$name"
    log_file="$LOG_ROOT/${name}.log"

    rm -rf "$out_dir"; mkdir -p "$out_dir"
    run_cfg="$out_dir/scaling_case1_256.toml"
    sed "s|out_dir          = .*|out_dir          = \"$out_dir\"|" "$CFG" > "$run_cfg"

    echo "=== $name :: R=$R T=$T BLAST_DIMS=$DIMS ==="

    t0=$(date +%s.%N)
    LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" "$MPIRUN" -n "$R" \
        --map-by "slot:PE=$T" \
        --bind-to core \
        -x OMP_NUM_THREADS="$T" \
        -x OMP_PROC_BIND=close \
        -x OMP_PLACES=cores \
        -x LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" \
        -x BLAST_DIMS="$DIMS" \
        "$BIN" "$run_cfg" > "$log_file" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')

    if [[ $rc -ne 0 ]]; then
        echo "  FAILED (rc=$rc); see $log_file" >&2
        printf '%d\t%d\t%s\t%s\t%s\tFAIL\tFAIL\n' "$R" "$T" "$DIMS" "$TAG" "$wall" >> "$SUMMARY"
        continue
    fi

    perstep=$(python3 "$REPO_DIR/postprocessing/scripts/scaling/parse_steps.py" --window 5 "$log_file")
    nsteps=$(sed 's/\x1b\[[0-9;]*[A-Za-z]//g' "$log_file" \
             | grep -E '^\[[0-9:.]+\] \[info\] step ' \
             | tail -1 | awk '{print $4}')

    echo "  wall_total=${wall}s  wall/step=${perstep}s  last_step=${nsteps}"
    printf '%d\t%d\t%s\t%s\t%s\t%s\t%s\n' "$R" "$T" "$DIMS" "$TAG" "$wall" "$perstep" "$nsteps" >> "$SUMMARY"
done

echo
echo "Summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
