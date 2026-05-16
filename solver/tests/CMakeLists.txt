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
