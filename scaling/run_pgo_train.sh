#!/usr/bin/env bash
# Three-phase PGO build for blast_les_mpi on the EPYC 9965:
#   1. Instrumented build into solver/build_pgo (-fprofile-generate, non-atomic
#      counters).
#   2. Training: 30 steps scaling_case1_64_pgo_train (shocked branch) + 30 steps
#      TGV 64^3 (smooth WENO5 branch). Run as 1 MPI rank × 8 OpenMP threads
#      (1 Zen5c CCD, shared L3) on the 64^3 grids. Two reasons for "small grid,
#      few threads":
#        - The 256^3 grid at 192 threads thrashes L1/L2 with gcov counter
#          writes, giving ~250x slowdown over baseline -- 30 training steps
#          would take ~3 hours.
#        - PGO consumes branch-frequency ratios, not loop iteration counts.
#          The 64^3 grid exercises the same WENO5/sensor/RHS branches as
#          256^3 production runs at a fraction of the step cost.
#      Halo / Allreduce code paths are not profiled in single-rank training
#      (under 5% of step time at 8x24 production, so PGO loss is small).
#   3. Re-configure same build dir with -fprofile-use and rebuild from clean.
#
# Apt OpenMPI 4.1.6 + apt FFTW3 build env, matching solver/cmake/configure-mpi.sh
# and solver/run_mpi.sh. See
# docs/solutions/build-errors/mpi-build-fftw3-and-mpi-abi-mismatch-2026-05-17.md.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd -- "$REPO_DIR"

BUILD_DIR="$REPO_DIR/solver/build_pgo"
PROFILE_DIR="$REPO_DIR/runs/scaling/pgo_profile"
BIN="$BUILD_DIR/blast_les_mpi"

MPIRUN="/usr/bin/mpirun.openmpi"
RUN_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu"

CONFIGURE_CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DBLAST_MPI=ON
    -DMPI_CXX_COMPILER=/usr/bin/mpicxx
    -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun
    -DFFTW3_INCLUDE_DIR=/usr/include
    -DFFTW3_LIB=/usr/lib/x86_64-linux-gnu/libfftw3.so
    -DFFTW3_OMP_LIB=/usr/lib/x86_64-linux-gnu/libfftw3_omp.a
    -DFFTW3_MPI_LIB=/usr/lib/x86_64-linux-gnu/libfftw3_mpi.a
)

# === Phase 1: instrumented build ============================================
echo "=== [1/3] instrumented build ==="
rm -rf "$BUILD_DIR" "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"
PATH=/usr/bin:/bin cmake -S solver -B "$BUILD_DIR" \
    "${CONFIGURE_CMAKE_FLAGS[@]}" \
    -DPGO_GENERATE=ON \
    -DPGO_PROFILE_DIR="$PROFILE_DIR"
cmake --build "$BUILD_DIR" -j 32 --target blast_les_mpi

# === Phase 2: training runs =================================================
run_train () {
    local cfg="$1"; local tag="$2"
    local out_dir="$REPO_DIR/runs/scaling/pgo_train_${tag}_out"
    rm -rf "$out_dir"; mkdir -p "$out_dir"
    local run_cfg="$out_dir/$(basename -- "$cfg")"
    sed "s|out_dir          = .*|out_dir          = \"$out_dir\"|" "$cfg" > "$run_cfg"
    echo "=== [2/3] training: $tag ==="
    PATH=/usr/bin:/bin LD_LIBRARY_PATH="$RUN_LD_LIBRARY_PATH" \
        "$MPIRUN" -n 1 \
        --map-by "slot:PE=8" --bind-to core \
        -x OMP_NUM_THREADS=8 \
        -x OMP_PROC_BIND=close \
        -x OMP_PLACES=cores \
        -x LD_LIBRARY_PATH \
        "$BIN" "$run_cfg"
}
run_train "$REPO_DIR/solver/examples/scaling_case1_64_pgo_train.toml"    "case1"
run_train "$REPO_DIR/solver/examples/tgv_re1600_64_hyper2_pgo_train.toml" "tgv64"

# Sanity check: .gcda files should now exist under $PROFILE_DIR.
gcda_count=$(find "$PROFILE_DIR" -name '*.gcda' | wc -l)
echo "=== profile populated: $gcda_count .gcda files under $PROFILE_DIR ==="
if [[ "$gcda_count" -eq 0 ]]; then
    echo "ERROR: no .gcda files produced; training did not capture a profile" >&2
    exit 3
fi

# === Phase 3: PGO-optimized rebuild =========================================
echo "=== [3/3] final PGO build (-fprofile-use) ==="
PATH=/usr/bin:/bin cmake -S solver -B "$BUILD_DIR" \
    "${CONFIGURE_CMAKE_FLAGS[@]}" \
    -DPGO_GENERATE=OFF \
    -DPGO_USE=ON \
    -DPGO_PROFILE_DIR="$PROFILE_DIR"
cmake --build "$BUILD_DIR" --clean-first -j 32 --target blast_les_mpi

echo
echo "PGO binary:        $BIN"
echo "Non-PGO baseline:  $REPO_DIR/solver/build_mpi/blast_les_mpi"
echo
echo "To compare at the production 8×24 / 192c layout:"
echo "  scaling/run_sweep.sh 192               # against the existing build_mpi binary"
echo "  BIN=$BIN scaling/run_sweep.sh 192      # override to time the PGO binary"
echo "(run_sweep.sh currently hard-codes the binary path; edit BIN= at the top, or"
echo " temporarily symlink solver/build_pgo/blast_les_mpi over solver/build_mpi/blast_les_mpi.)"
