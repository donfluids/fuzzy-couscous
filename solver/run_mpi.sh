#!/usr/bin/env bash
# Run a command under the apt OpenMPI 4.1.6 stack, hiding the /opt/openmpi-5
# install and /usr/local FFTW that the interactive shell preloads for
# OpenFOAM/OpenRadioss. Required on boxes where ldd shows libmpi.so.40
# resolving outside /usr/lib/x86_64-linux-gnu/.
#
# Usage:
#   ./solver/run_mpi.sh ctest --test-dir solver/build_mpi --output-on-failure
#   ./solver/run_mpi.sh mpirun -n 8 ./solver/build_mpi/blast_les_mpi case.toml
#
# See docs/solutions/build-errors/mpi-build-fftw3-and-mpi-abi-mismatch-2026-05-17.md
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 64
fi

export PATH=/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu

exec "$@"
