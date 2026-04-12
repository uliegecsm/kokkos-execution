# Find kokkos-utils.
find_package(KokkosUtils 0.0.4 CONFIG REQUIRED)

# Helpers.
include(${CMAKE_SOURCE_DIR}/cmake/functions/path.cmake)

# Mapping from enabled Kokkos devices to execution space type.
set(KOKKOS_DEVICE_TO_EXECUTION_SPACE_CUDA "Cuda")
set(KOKKOS_DEVICE_TO_EXECUTION_SPACE_HIP "HIP")
set(KOKKOS_DEVICE_TO_EXECUTION_SPACE_HPX "Experimental::HPX")
set(KOKKOS_DEVICE_TO_EXECUTION_SPACE_OPENMP "OpenMP")
set(KOKKOS_DEVICE_TO_EXECUTION_SPACE_SERIAL "Serial")
set(KOKKOS_DEVICE_TO_EXECUTION_SPACE_SYCL "SYCL")
set(KOKKOS_DEVICE_TO_EXECUTION_SPACE_THREADS "Threads")

# Function to add a test (common steps).
function(add_one_test)

  cmake_parse_arguments(aot_args "" "NAME" "ADDITIONAL_LINK_LIBRARIES" ${ARGN})

  # Set executable name and primary source file.
  cmake_path(SET SOURCE_FILE "test_${aot_args_NAME}.cpp")
  cmake_path(ABSOLUTE_PATH SOURCE_FILE)

  get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

  foreach(DEVICE IN LISTS Kokkos_DEVICES)
    if(NOT DEFINED KOKKOS_DEVICE_TO_EXECUTION_SPACE_${DEVICE})
      message(FATAL_ERROR "No execution space mapping for device '${DEVICE}'.")
    endif()

    set(aot_args_EXECUTABLE_NAME "${gprr_PARENT_RELPATH_REPR}_${aot_args_NAME}.${DEVICE}")

    # Create test executable.
    add_executable(${aot_args_EXECUTABLE_NAME})

    # Add the source file to the executable.
    target_sources(${aot_args_EXECUTABLE_NAME} PRIVATE ${SOURCE_FILE})

    # Tell the translation unit which execution space to use.
    target_compile_definitions(
      ${aot_args_EXECUTABLE_NAME} PRIVATE TEST_EXECUTION_SPACE=Kokkos::${KOKKOS_DEVICE_TO_EXECUTION_SPACE_${DEVICE}}
    )

    # Link the executable to 'test_common'.
    target_link_libraries(${aot_args_EXECUTABLE_NAME} PRIVATE test_common)

    # Always link to 'kokkos-execution'.
    target_link_libraries(${aot_args_EXECUTABLE_NAME} PRIVATE Kokkos::Execution)

    # Always allow logging and kokkos-utils.
    target_link_libraries(${aot_args_EXECUTABLE_NAME} PRIVATE plog::plog Kokkos::kokkosutils)

    if(DEFINED aot_args_ADDITIONAL_LINK_LIBRARIES)
      target_link_libraries(${aot_args_EXECUTABLE_NAME} PRIVATE ${aot_args_ADDITIONAL_LINK_LIBRARIES})
    endif()

    add_test(NAME ${aot_args_EXECUTABLE_NAME} COMMAND $<TARGET_FILE:${aot_args_EXECUTABLE_NAME}>
                                                      --gtest_fail_if_no_test_linked
    )

    # Ensure that GTest will shuffle the tests.
    set_property(TEST ${aot_args_EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_SHUFFLE=1")
    set_property(TEST ${aot_args_EXECUTABLE_NAME} APPEND PROPERTY ENVIRONMENT "GTEST_RANDOM_SEED=0")
  endforeach()
endfunction()

function(add_build_failure_test NAME TARGET REGEX)
  message(STATUS "Adding build failure test ${NAME} for target ${TARGET}.")

  set_target_properties(${TARGET} PROPERTIES EXCLUDE_FROM_ALL TRUE EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  set_tests_properties(${TARGET} PROPERTIES DISABLED TRUE)

  add_test(NAME ${NAME} COMMAND ${CMAKE_COMMAND} --build . --config $<CONFIGURATION> --target ${TARGET} --verbose
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(${NAME} PROPERTIES PASS_REGULAR_EXPRESSION "${REGEX}")
endfunction()

# Common test object library.
add_library(test_common OBJECT)
target_sources(test_common PRIVATE main.cpp)
target_link_libraries(test_common PUBLIC GTest::gtest Kokkos::kokkoscore)

# Include subdirectories.
add_subdirectory(execution_space)
add_subdirectory(graph)
add_subdirectory(impl)
add_subdirectory(utils)
