---
title: "MPI build on a mixed OpenMPI 4.1.6 / 5.0.10 environment"
date: 2026-05-17
category: docs/solutions/build-errors
module: solver/build-mpi
problem_type: build_error
component: tooling
symptoms:
  - "linker error: undefined reference to `fftw_taint` and `fftw_join_taint` when building blast_les_mpi"
  - "cmake's find_library picked /usr/local/lib/libfftw3.so (sideloaded) instead of apt's /usr/lib/x86_64-linux-gnu/libfftw3.so.3"
  - "ldd blast_les_mpi resolved libmpi.so.40 to /opt/openmpi-5_0304 (OpenMPI 5.0.10) while libmpi_cxx.so.40 and parallel HDF5 came from apt OpenMPI 4.1.6"
  - "two OpenMPI runtimes loaded into one process because libfftw3_mpi from both vendors shares SONAME libmpi.so.40"
  - "MPI regression tests only pass under a scrubbed PATH and LD_LIBRARY_PATH pointing at /usr/lib/x86_64-linux-gnu/openmpi"
root_cause: config_error
resolution_type: environment_setup
severity: high
related_components:
  - development_workflow
  - testing_framework
tags:
  - mpi
  - openmpi
  - fftw3
  - cmake
  - ld-library-path
  - abi-mismatch
  - hdf5-parallel
  - blast-les
---

# MPI build on a mixed OpenMPI 4.1.6 / 5.0.10 environment

## Problem

After merging an upstream MPI port, `blast_les_mpi` failed to link with undefined references to `fftw_taint`/`fftw_join_taint`, and once linked it pulled two incompatible OpenMPI runtimes (4.1.6 and 5.0.10) into the same process via `libmpi.so.40`. Both failures were caused by a sideloaded `/usr/local` FFTW + `/opt/openmpi-5_0304` OpenMPI 5 stack masking the apt OpenMPI 4.1.6 + apt FFTW3 stack that the MPI port was actually built against.

## Symptoms

- Link step failed with `/usr/bin/ld: ... undefined reference to fftw_taint` and `fftw_join_taint`, repeated across objects in `libfftw3_mpi.a(dft-problem.o)` and siblings.
- CMake emitted `Cannot generate a safe runtime search path... runtime library [libfftw3_mpi.so.3] in /usr/lib/x86_64-linux-gnu may be hidden by files in: /usr/local/lib` and silently dropped the requested rpath.
- `ldd build_mpi/blast_les_mpi` showed a split runtime: `libmpi.so.40 => /opt/openmpi-5_0304/lib/libmpi.so.40` (5.0.10) alongside `libmpi_cxx.so.40 => /usr/lib/x86_64-linux-gnu/libmpi_cxx.so.40` (4.1.6) and `libhdf5_openmpi.so.103` built against 4.1.6.
- `libfftw3.so.3` resolved to `/usr/local/lib/libfftw3.so.3` (sideloaded), which `nm -D` confirms does not export `fftw_taint`.
- `which mpirun` returned `/opt/openmpi-5_0304/bin/mpirun` (OpenMPI 5.0.10) because the user's shell preloads `/opt/openmpi-5_0304/bin` in PATH for OpenFOAM/OpenRadioss.

## What Didn't Work

- `-DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/usr/lib/x86_64-linux-gnu"` (and the matching shared-linker flag) — CMake intentionally strips rpath entries that resolve to implicit system directories to keep binaries portable, so the rpath never landed in the ELF and `/usr/local/lib` still won at load time.
- Linking the apt dynamic `-DFFTW3_MPI_LIB=/usr/lib/x86_64-linux-gnu/libfftw3_mpi.so` — link succeeds against apt 4.1.6 ABI, but `ldconfig -p` lists `/usr/local/lib/libfftw3_mpi.so.3` first, so the loader picks the 5.0.10 build at runtime and you get a mixed-MPI process.
- Relying on the user's interactive shell env (PATH/LD_LIBRARY_PATH tuned for OpenFOAM/OpenRadioss with `/opt/openmpi-5_0304` first) — guarantees `libmpi.so.40` resolves to 5.0.10 even when every other MPI symbol came from apt 4.1.6.
- Trusting OpenMPI's `libmpi.so.40` SONAME as an ABI promise — the SONAME is held stable across 4.x/5.x as a compatibility window, but the ABI is not, so mixing majors crashes at `MPI_Init` or the first collective.

## Solution

Two-part fix: pin the build to apt FFTW3 (static for the MPI/OMP variants), and scrub the runtime env at every invocation so only apt OpenMPI 4.1.6 is loaded.

### 1. Install the apt FFTW3-MPI package

```
sudo apt install -y libfftw3-mpi-dev
```

This ships `libfftw3_mpi.a` and `libfftw3_mpi.so` built against system OpenMPI 4.1.6, alongside `libhdf5-openmpi-dev` which is already installed.

### 2. Configure with explicit apt-FFTW3 hints

```
cmake -S solver -B solver/build_mpi \
  -DCMAKE_BUILD_TYPE=Release -DBLAST_MPI=ON \
  -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
  -DMPIEXEC_EXECUTABLE=/usr/bin/mpirun \
  -DFFTW3_INCLUDE_DIR=/usr/include \
  -DFFTW3_LIB=/usr/lib/x86_64-linux-gnu/libfftw3.so \
  -DFFTW3_OMP_LIB=/usr/lib/x86_64-linux-gnu/libfftw3_omp.a \
  -DFFTW3_MPI_LIB=/usr/lib/x86_64-linux-gnu/libfftw3_mpi.a

cmake --build solver/build_mpi -j
```

Static `.a` for `fftw3_omp` and `fftw3_mpi` is deliberate — apt only ships these as static anyway, and static linkage sidesteps the `/usr/local` vs `/usr/lib/x86_64-linux-gnu` runtime-resolution race for the MPI variant.

### 3. Scrub PATH/LD_LIBRARY_PATH at runtime

```
PATH=/usr/bin:/bin \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu \
ctest --test-dir solver/build_mpi --output-on-failure
```

Same prefix for production runs:

```
PATH=/usr/bin:/bin \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu \
mpirun -n <N> ./solver/build_mpi/blast_les_mpi <config.toml>
```

Verify the resolution before trusting it:

```
PATH=/usr/bin:/bin \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu \
ldd solver/build_mpi/blast_les_mpi | grep -E 'mpi|fftw|hdf5'
```

Every `libmpi*` and `libhdf5_openmpi*` line should now point at `/usr/lib/x86_64-linux-gnu/...`. Expected result: `ctest` 16/16 green (unit_tests ~236.6 s + 5 MPI tests × 3 rank counts, total ~249.4 s).

### 4. Bake it into a wrapper script

Create `solver/run_mpi.sh` so nobody has to remember the env scrub:

```bash
#!/usr/bin/env bash
# Force the apt OpenMPI 4.1.6 stack, hiding the /opt/openmpi-5 install
# that the user's interactive shell preloads for OpenFOAM/OpenRadioss.
export PATH=/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu/hdf5/openmpi:/usr/lib/x86_64-linux-gnu
exec "$@"
```

Then `./solver/run_mpi.sh ctest --test-dir solver/build_mpi --output-on-failure` or `./solver/run_mpi.sh mpirun -n 8 ./solver/build_mpi/blast_les_mpi case.toml`. A companion `solver/cmake/configure-mpi.sh` that wraps the configure command from step 2 makes the build side equally one-shot.

## Why This Works

- **Apt FFTW3 exports the missing symbols.** `nm -D /usr/lib/x86_64-linux-gnu/libfftw3.so.3 | grep taint` lists `fftw_taint` and `fftw_join_taint` as `T` (defined/exported), while `/usr/local/lib/libfftw3.so` does not. The apt-built `libfftw3_mpi.a` references these symbols externally, so the linker needs the apt `libfftw3` in the same link line — passing `-DFFTW3_LIB=/usr/lib/x86_64-linux-gnu/libfftw3.so` forces exactly that.
- **Static `libfftw3_mpi.a` sidesteps ld.so.** Even with apt installed, `ldconfig -p` returns `libfftw3_mpi.so.3 => /usr/local/lib/libfftw3_mpi.so.3` (the 5.0.10 build) ahead of the apt copy, and CMake refuses to bake a rpath into `/usr/lib/x86_64-linux-gnu` because that's an implicit system dir. Static linkage embeds the apt-built object code into `blast_les_mpi` at link time, eliminating the runtime lookup entirely.
- **LD_LIBRARY_PATH scrubbing is required because OpenMPI's SONAME lies about ABI.** Both OpenMPI 4.1.6 and 5.0.10 expose `libmpi.so.40` (a deliberate compatibility-window policy), but the ABI is not stable across majors. With `/opt/openmpi-5_0304/lib` first in `LD_LIBRARY_PATH`, `libmpi.so.40` resolves to 5.0.10 while `libmpi_cxx.so.40` and `libhdf5_openmpi.so.103` stay on the apt 4.1.6 build — guaranteed crash at `MPI_Init` or the first collective. Putting `/usr/lib/x86_64-linux-gnu/openmpi/lib` first forces every MPI symbol through the 4.1.6 copy and gives a consistent stack.
- **CMake's `-Wl,-rpath` workaround can't help here.** CMake strips rpath entries pointing at implicit system directories (`/usr/lib/x86_64-linux-gnu` qualifies) to keep binaries portable, which is why the "may be hidden by files in: /usr/local/lib" warning fires but the rpath never appears in the ELF. The only knobs that actually win are (a) static linkage and (b) an env scrub the loader honors before consulting ld.so.cache.

## Prevention

- Land `solver/run_mpi.sh` (env scrub) and `solver/cmake/configure-mpi.sh` (cmake hints) and make them the documented entry points for MPI build/test/run on this box.
- Add a README section under `solver/` explaining the `/opt/openmpi-5_0304` vs apt OpenMPI 4.1.6 conflict, the required cmake flags, and the runtime env requirement — future-you and any collaborator will hit this within a week of touching the MPI build.
- House rule: when `libfftw3-mpi-dev` (apt) coexists with a `/usr/local` FFTW build, prefer the apt static `.a` for the MPI/OMP variants. The apt dynamic `.so` link will succeed and then silently lose to `/usr/local` at load time.
- Consider a `solver/build_mpi.Dockerfile` or a `direnv`/`env -i` build target so MPI configure/build/test is isolated from the user's interactive OpenFOAM/OpenRadioss environment. A CI-style fresh-shell target would have caught both failures on the first attempt.
- Add a `ctest` smoke step that runs `ldd build_mpi/blast_les_mpi | grep -E 'mpi|fftw|hdf5'` and fails if any line resolves outside `/usr/lib/x86_64-linux-gnu` — turns "two-MPI-loaded" into a build-time error instead of a runtime crash.

## Related Issues

- `README.md` "MPI (OpenMPI)" section (lines 31–47) documents the happy-path apt-only build (`libopenmpi-dev`, `libhdf5-openmpi-dev`, `libfftw3-mpi-dev` + `cmake -DBLAST_MPI=ON`). It does not cover sideloaded-MPI conflicts; consider linking here once this doc lands.
- `solver/CMakeLists.txt` (lines 82–132) — the MPI block that emits `FATAL_ERROR` when parallel HDF5 or `libfftw3_mpi` is missing. Hard-codes apt parallel-HDF5 paths.
- GitHub issues on `donfluids/fuzzy-couscous` were not enumerated (sandbox blocked `gh issue list`); re-run manually before treating this as exhaustive.
