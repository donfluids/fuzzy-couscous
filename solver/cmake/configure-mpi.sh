#!/usr/bin/env bash
# Configure the MPI build (solver/build_mpi/) using apt OpenMPI 4.1.6 + apt FFTW3,
# overriding any sideloaded /usr/local FFTW or /opt/openmpi-5 that cmake's
# default search would otherwise pick up.
#
# Usage (from the repo root):
#   ./solver/cmake/configure-mpi.sh                  # configures solver/build_mpi/
#   ./solver/cmake/configure-mpi.sh -DEXTRA_FLAG=... # extra args forwarded to cmake
#
# After configure:
#   cmake --build solver/build_mpi -j
#   ./solver/run_mpi.sh ctest --test-dir solver/build_mpi --output-on-failure
#
# See docs/solutions/build-errors/mpi-build-fftw3-and-mpi-abi-mismatch-2026-05-17.md
set -euo pipefail

# Run from the repo root regardless of where the user invoked us.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
cd -- "${repo_root}"

PATH=/usr/bin:/bin cmake -S solver -B solver/build_mpi \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLAST_MPI=ON \
    -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
    -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun \
    -DFFTW3_INCLUDE_DIR=/usr/include \
    -DFFTW3_LIB=/usr/lib/x86_64-linux-gnu/libfftw3.so \
    -DFFTW3_OMP_LIB=/usr/lib/x86_64-linux-gnu/libfftw3_omp.a \
    -DFFTW3_MPI_LIB=/usr/lib/x86_64-linux-gnu/libfftw3_mpi.a \
    "$@"
