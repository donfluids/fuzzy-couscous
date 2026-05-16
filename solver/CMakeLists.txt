cmake_minimum_required(VERSION 3.20)
project(blast_les CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Native-CPU optimization. Override with -DNATIVE_OPT=OFF for portable build.
option(NATIVE_OPT "Enable -march=native and aggressive vectorization" ON)
option(BUILD_TESTS "Build unit tests" ON)

if(NATIVE_OPT)
    add_compile_options(-march=native -mtune=native -ffast-math -fno-finite-math-only)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(spdlog REQUIRED)

# FFTW3: no CMake config in apt package; locate manually.
find_path(FFTW3_INCLUDE_DIR fftw3.h)
find_library(FFTW3_LIB fftw3)
find_library(FFTW3_OMP_LIB fftw3_omp)
if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIB OR NOT FFTW3_OMP_LIB)
    message(FATAL_ERROR "FFTW3 (+ fftw3_omp) not found")
endif()

# toml++: apt install puts headers at /usr/include/toml++/
find_path(TOMLPP_INCLUDE_DIR toml++/toml.hpp PATHS /usr/include /usr/local/include)
if(NOT TOMLPP_INCLUDE_DIR)
    message(FATAL_ERROR "toml++ headers not found")
endif()

add_library(blast_core STATIC
    src/core/Field3D.cpp
    src/core/Config.cpp
    src/io/Log.cpp
    src/bc/BC.cpp
    src/numerics/RHS.cpp
    src/numerics/RK3.cpp
    src/numerics/Ducros.cpp
    src/numerics/Gradients.cpp
    src/ic/Canonical.cpp
    src/physics/RiemannExact.cpp
    src/diagnostics/FFT.cpp
    src/diagnostics/Spectra.cpp
    src/diagnostics/Statistics.cpp
    src/io/HDF5Writer.cpp
    src/io/Restart.cpp
)
target_include_directories(blast_core PUBLIC
    src
    ${HDF5_INCLUDE_DIRS}
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

if(BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests)
endif()

add_executable(blast_les src/main.cpp)
target_link_libraries(blast_les PRIVATE blast_core)
