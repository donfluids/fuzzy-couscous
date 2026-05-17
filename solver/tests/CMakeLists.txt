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
    test_rogallo.cpp
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

# Make fixture configs available to tests at the configured location.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixtures/example.toml
               ${CMAKE_CURRENT_BINARY_DIR}/fixtures/example.toml COPYONLY)

if(BLAST_MPI)
    add_executable(test_mpi_halo test_mpi_halo.cpp)
    target_link_libraries(test_mpi_halo PRIVATE blast_core_mpi)

    add_executable(test_mpi_bitexact test_mpi_bitexact.cpp)
    target_link_libraries(test_mpi_bitexact PRIVATE blast_core_mpi)

    add_executable(test_mpi_restart test_mpi_restart.cpp)
    target_link_libraries(test_mpi_restart PRIVATE blast_core_mpi)

    # Run under mpirun at 1, 2, 4 ranks. OpenMPI in containers (Docker/etc.)
    # needs --oversubscribe + the root override.
    set(MPI_ENV "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1")
    foreach(N 1 2 4)
        add_test(NAME mpi_halo_${N}
            COMMAND ${MPIEXEC_EXECUTABLE}
                    --oversubscribe
                    -n ${N}
                    $<TARGET_FILE:test_mpi_halo>)
        set_tests_properties(mpi_halo_${N} PROPERTIES
            ENVIRONMENT "${MPI_ENV}")

        add_test(NAME mpi_bitexact_${N}
            COMMAND ${MPIEXEC_EXECUTABLE}
                    --oversubscribe
                    -n ${N}
                    $<TARGET_FILE:test_mpi_bitexact>)
        set_tests_properties(mpi_bitexact_${N} PROPERTIES
            ENVIRONMENT "${MPI_ENV}")

        add_test(NAME mpi_restart_${N}
            COMMAND ${MPIEXEC_EXECUTABLE}
                    --oversubscribe
                    -n ${N}
                    $<TARGET_FILE:test_mpi_restart>)
        set_tests_properties(mpi_restart_${N} PROPERTIES
            ENVIRONMENT "${MPI_ENV}")
    endforeach()
endif()
