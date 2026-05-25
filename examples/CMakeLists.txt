# Function to add an example (common steps).
function(add_one_example)

  cmake_parse_arguments(aoe_args "" "NAME" "ADDITIONAL_LINK_LIBRARIES" ${ARGN})

  # Set executable name and primary source file.
  cmake_path(SET SOURCE_FILE "example_${aoe_args_NAME}.cpp")
  cmake_path(ABSOLUTE_PATH SOURCE_FILE)

  get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

  foreach(DEVICE IN LISTS Kokkos_DEVICES)
    if(NOT DEFINED KOKKOS_DEVICE_TO_EXECUTION_SPACE_${DEVICE})
      message(FATAL_ERROR "No execution space mapping for device '${DEVICE}'.")
    endif()

    set(aoe_args_EXECUTABLE_NAME "${gprr_PARENT_RELPATH_REPR}_${aoe_args_NAME}.${DEVICE}")

    # Create example executable.
    add_executable(${aoe_args_EXECUTABLE_NAME})

    # Add the source file to the executable.
    target_sources(${aoe_args_EXECUTABLE_NAME} PRIVATE ${SOURCE_FILE})

    # Tell the translation unit which execution space to use.
    target_compile_definitions(
      ${aoe_args_EXECUTABLE_NAME} PRIVATE EXAMPLE_EXECUTION_SPACE=Kokkos::${KOKKOS_DEVICE_TO_EXECUTION_SPACE_${DEVICE}}
    )

    # Always link to 'kokkos' and 'kokkos-execution'.
    target_link_libraries(${aoe_args_EXECUTABLE_NAME} PRIVATE Kokkos::kokkoscore Kokkos::Execution)

    # Always allow logging and kokkos-utils.
    target_link_libraries(${aoe_args_EXECUTABLE_NAME} PRIVATE plog::plog Kokkos::kokkosutils)

    if(DEFINED aot_args_ADDITIONAL_LINK_LIBRARIES)
      target_link_libraries(${aoe_args_EXECUTABLE_NAME} PRIVATE ${aoe_args_ADDITIONAL_LINK_LIBRARIES})
    endif()

    add_test(NAME ${aoe_args_EXECUTABLE_NAME} COMMAND $<TARGET_FILE:${aoe_args_EXECUTABLE_NAME}>)
  endforeach()
endfunction()

# Include subdirectories.
add_subdirectory(cg)
