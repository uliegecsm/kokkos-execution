# Find kokkos-utils.
find_package(KokkosUtils CONFIG REQUIRED)

# Find Google Test.
string(JSON GTest_REQUIRED_VERSION GET "${DEPENDENCIES_JSON}" github googletest)

find_package(GTest ${GTest_REQUIRED_VERSION} CONFIG REQUIRED)

# Helpers.
include(${CMAKE_SOURCE_DIR}/cmake/functions/path.cmake)

# Function to add a test (common steps).
function(add_one_test)

  cmake_parse_arguments(aot_args "" "NAME" "" ${ARGN})

  # Set executable name and primary source file.
  cmake_path(SET SOURCE_FILE "test_${aot_args_NAME}.cpp")
  cmake_path(ABSOLUTE_PATH SOURCE_FILE)

  get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

  set(aot_args_EXECUTABLE_NAME "${gprr_PARENT_RELPATH_REPR}_${aot_args_NAME}")

  # Create test executable.
  add_executable(${aot_args_EXECUTABLE_NAME})

  # Add the source file to the executable.
  target_sources(${aot_args_EXECUTABLE_NAME} PRIVATE ${SOURCE_FILE})

  # Link the executable to 'test_common'.
  target_link_libraries(${aot_args_EXECUTABLE_NAME} PRIVATE test_common)

  # Always link to 'kokkos-execution'.
  target_link_libraries(${aot_args_EXECUTABLE_NAME} PRIVATE Kokkos::Execution)

  # Always allow logging and kokkos-utils.
  target_link_libraries(${aot_args_EXECUTABLE_NAME} PRIVATE plog::plog Kokkos::kokkosutils)

  add_test(NAME ${aot_args_EXECUTABLE_NAME} COMMAND $<TARGET_FILE:${aot_args_EXECUTABLE_NAME}>
                                                    --gtest_fail_if_no_test_linked
  )

  # Ensure that GTest will shuffle the tests.
  set_property(TEST ${aot_args_EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_SHUFFLE=1")
  set_property(TEST ${aot_args_EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_RANDOM_SEED=0")
endfunction()

# Common test object library.
add_library(test_common OBJECT)
target_sources(test_common PRIVATE main.cpp)
target_link_libraries(test_common PUBLIC GTest::gtest Kokkos::kokkoscore)

# Include subdirectories.
add_subdirectory(execution_space)
