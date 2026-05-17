add_executable(unit_tests
    test_field3d.cpp
    test_config.cpp
    test_stencils.cpp
    test_bc.cpp
    test_advect_smooth.cpp
    test_sod_1d.cpp
    test_sedov_3d.cpp
    test_viscous_mms.cpp
    test_spectra.cpp
    test_statistics.cpp
    test_tgv.cpp
    test_chamber_smoke.cpp
    test_cj_detonation.cpp
    test_restart.cpp
    test_mms.cpp
    test_hyperdissipation.cpp
    test_slip_wall.cpp
    test_shu_osher_1d.cpp
)
target_link_libraries(unit_tests PRIVATE
    blast_core
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(unit_tests PRIVATE ../src)

add_test(NAME unit_tests
    COMMAND unit_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Grids in this suite are small (16^3 – 64^3); on many-core boxes (e.g. 192-core
# Zen5c) OpenMP barrier and cross-NUMA overhead dominate the arithmetic unless
# threads are pinned and bounded. Measured on 2026-05-17: MMS3D drops from
# 325 s (192 unbound) to 56 s (32 pinned). Override these in the environment
# if running on different hardware.
set_tests_properties(unit_tests PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=32;OMP_PROC_BIND=close;OMP_PLACES=cores")

# Make fixture configs available to tests at the configured location.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixtures/example.toml
               ${CMAKE_CURRENT_BINARY_DIR}/fixtures/example.toml COPYONLY)
