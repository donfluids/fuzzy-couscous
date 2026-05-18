cmake_minimum_required(VERSION 3.20)
project(blast_les CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NATIVE_OPT "Enable -march=native and aggressive vectorization" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BLAST_MPI   "Build MPI variant alongside serial (libfftw3_mpi + parallel HDF5)" OFF)

if(NATIVE_OPT)
    add_compile_options(-march=native -mtune=native -ffast-math -fno-finite-math-only)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

find_package(OpenMP REQUIRED)
find_package(spdlog REQUIRED)

# Serial HDF5 search. Ubuntu installs the header under hdf5/serial/, which
# isn't on the default include path. Add it explicitly.
find_package(HDF5 REQUIRED COMPONENTS C)
set(SERIAL_HDF5_INCLUDE_DIR /usr/include/hdf5/serial)

# FFTW3 (serial + OpenMP).
find_path(FFTW3_INCLUDE_DIR fftw3.h)
find_library(FFTW3_LIB fftw3)
find_library(FFTW3_OMP_LIB fftw3_omp)
if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIB OR NOT FFTW3_OMP_LIB)
    message(FATAL_ERROR "FFTW3 (+ fftw3_omp) not found")
endif()

find_path(TOMLPP_INCLUDE_DIR toml++/toml.hpp PATHS /usr/include /usr/local/include)
if(NOT TOMLPP_INCLUDE_DIR)
    message(FATAL_ERROR "toml++ headers not found")
endif()

set(BLAST_CORE_SOURCES
    src/core/Field3D.cpp
    src/core/Config.cpp
    src/io/Log.cpp
    src/bc/BC.cpp
    src/numerics/RHS.cpp
    src/numerics/RhsScratch.cpp
    src/numerics/RK3.cpp
    src/numerics/Ducros.cpp
    src/numerics/Filter.cpp
    src/numerics/Gradients.cpp
    src/ic/Canonical.cpp
    src/physics/RiemannExact.cpp
    src/physics/Forcing.cpp
    src/diagnostics/FFT.cpp
    src/diagnostics/Spectra.cpp
    src/diagnostics/Statistics.cpp
    src/io/HDF5Writer.cpp
    src/io/Restart.cpp
)

# === Serial library + executable ===
add_library(blast_core STATIC ${BLAST_CORE_SOURCES})
target_include_directories(blast_core PUBLIC
    src
    ${HDF5_INCLUDE_DIRS}
    ${SERIAL_HDF5_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIR}
    ${TOMLPP_INCLUDE_DIR}
)
target_link_libraries(blast_core PUBLIC
    OpenMP::OpenMP_CXX
    spdlog::spdlog
    ${HDF5_LIBRARIES}
    ${FFTW3_OMP_LIB}
    ${FFTW3_LIB}
)

add_executable(blast_les src/main.cpp)
target_link_libraries(blast_les PRIVATE blast_core)

# === MPI library + executable (optional) ===
if(BLAST_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)

    # OpenMPI flavour of parallel HDF5 lives at /usr/include/hdf5/openmpi and
    # /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so on Ubuntu/Debian.
    find_path(PHDF5_INCLUDE_DIR hdf5.h
        PATHS /usr/include/hdf5/openmpi /usr/include/hdf5/mpich
        NO_DEFAULT_PATH)
    find_library(PHDF5_LIB hdf5
        PATHS /usr/lib/x86_64-linux-gnu/hdf5/openmpi
              /usr/lib/x86_64-linux-gnu/hdf5/mpich
        NO_DEFAULT_PATH)
    if(NOT PHDF5_INCLUDE_DIR OR NOT PHDF5_LIB)
        message(FATAL_ERROR "BLAST_MPI=ON but parallel HDF5 not found "
                "(install libhdf5-openmpi-dev)")
    endif()
    message(STATUS "Parallel HDF5: ${PHDF5_LIB}")
    message(STATUS "Parallel HDF5 includes: ${PHDF5_INCLUDE_DIR}")

    find_library(FFTW3_MPI_LIB fftw3_mpi)
    if(NOT FFTW3_MPI_LIB)
        message(FATAL_ERROR "BLAST_MPI=ON but libfftw3_mpi not found "
                "(install libfftw3-mpi-dev)")
    endif()

    add_library(blast_core_mpi STATIC
        ${BLAST_CORE_SOURCES}
        src/parallel/Domain.cpp
        src/parallel/Halo.cpp
    )
    target_compile_definitions(blast_core_mpi PUBLIC BLAST_MPI)
    target_include_directories(blast_core_mpi PUBLIC
        src
        ${PHDF5_INCLUDE_DIR}
        ${FFTW3_INCLUDE_DIR}
        ${TOMLPP_INCLUDE_DIR}
    )
    target_link_libraries(blast_core_mpi PUBLIC
        OpenMP::OpenMP_CXX
        spdlog::spdlog
        MPI::MPI_CXX
        ${PHDF5_LIB}
        ${FFTW3_MPI_LIB}
        ${FFTW3_OMP_LIB}
        ${FFTW3_LIB}
    )

    add_executable(blast_les_mpi src/main_mpi.cpp)
    target_link_libraries(blast_les_mpi PRIVATE blast_core_mpi)
endif()

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()
