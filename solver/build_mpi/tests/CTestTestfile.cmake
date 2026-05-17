# CMake generated Testfile for 
# Source directory: /home/user/fuzzy-couscous/solver/tests
# Build directory: /home/user/fuzzy-couscous/solver/build_mpi/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[unit_tests]=] "/home/user/fuzzy-couscous/solver/build_mpi/tests/unit_tests")
set_tests_properties([=[unit_tests]=] PROPERTIES  WORKING_DIRECTORY "/home/user/fuzzy-couscous/solver/build_mpi/tests" _BACKTRACE_TRIPLES "/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;28;add_test;/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;0;")
add_test([=[mpi_halo_1]=] "/usr/bin/mpiexec" "--oversubscribe" "-n" "1" "/home/user/fuzzy-couscous/solver/build_mpi/tests/test_mpi_halo")
set_tests_properties([=[mpi_halo_1]=] PROPERTIES  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" _BACKTRACE_TRIPLES "/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;44;add_test;/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;0;")
add_test([=[mpi_halo_2]=] "/usr/bin/mpiexec" "--oversubscribe" "-n" "2" "/home/user/fuzzy-couscous/solver/build_mpi/tests/test_mpi_halo")
set_tests_properties([=[mpi_halo_2]=] PROPERTIES  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" _BACKTRACE_TRIPLES "/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;44;add_test;/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;0;")
add_test([=[mpi_halo_4]=] "/usr/bin/mpiexec" "--oversubscribe" "-n" "4" "/home/user/fuzzy-couscous/solver/build_mpi/tests/test_mpi_halo")
set_tests_properties([=[mpi_halo_4]=] PROPERTIES  ENVIRONMENT "OMPI_ALLOW_RUN_AS_ROOT=1;OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1" _BACKTRACE_TRIPLES "/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;44;add_test;/home/user/fuzzy-couscous/solver/tests/CMakeLists.txt;0;")
