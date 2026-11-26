if(CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")

  cmake_path(SET SOURCE_FILE "test_move_elision.cpp")
  cmake_path(ABSOLUTE_PATH SOURCE_FILE)

  get_parent_relpath_repr(DIRECTORY "${PROJECT_SOURCE_DIR}" FILENAME "${SOURCE_FILE}")

  set(TEST_NAME "${gprr_PARENT_RELPATH_REPR}_move_elision")

  # Lengths of the sender composition.
  foreach(
    LENGTH IN
    ITEMS 1
          2
          3
          4
          5
          28
          29
          30
  )
    set(TARGET_NAME "${TEST_NAME}.${LENGTH}")

    add_library(${TARGET_NAME} OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/test_move_elision.cpp)
    target_compile_definitions(${TARGET_NAME} PRIVATE COMPOSITION_LENGTH=${LENGTH})
    target_link_libraries(${TARGET_NAME} PRIVATE Kokkos::Execution)
  endforeach()

  find_package(Python REQUIRED COMPONENTS Interpreter)

  add_test(NAME ${TEST_NAME} COMMAND ${Python_EXECUTABLE} -m pytest -vvv --log-cli-level=info -s
                                     ${CMAKE_CURRENT_SOURCE_DIR}/test_move_elision.py
  )

  test_environment(${TEST_NAME} CMAKE_BINARY_DIR set ${CMAKE_BINARY_DIR})
  test_environment(${TEST_NAME} CMAKE_CURRENT_BINARY_DIR set ${CMAKE_CURRENT_BINARY_DIR})

  set_tests_properties(${TEST_NAME} PROPERTIES LABELS "codegen")
else()
  message(WARNING "Tests on the elision of moves are disabled because CMAKE_BUILD_TYPE is '${CMAKE_BUILD_TYPE}'.")
endif()
