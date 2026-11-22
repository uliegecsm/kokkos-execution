if(NOT CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
  message(STATUS "Tests on the elision of moves are disabled because CMAKE_BUILD_TYPE is '${CMAKE_BUILD_TYPE}'.")
  return()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(MOVES_ELIDED_UP_TO 4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  set(MOVES_ELIDED_UP_TO 5)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(MOVES_ELIDED_UP_TO 28)
else()
  message(
    STATUS "Tests on the elision of moves are disabled because CMAKE_CXX_COMPILER_ID is '${CMAKE_CXX_COMPILER_ID}'."
  )
  return()
endif()

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
  set(TARGET tests_codegen_elide_moves_${LENGTH})

  add_library(${TARGET} OBJECT elide_moves.cpp)
  target_compile_definitions(${TARGET} PRIVATE CHAIN_LENGTH=${LENGTH})
  target_link_libraries(${TARGET} PRIVATE Kokkos::Execution)

  if(LENGTH GREATER MOVES_ELIDED_UP_TO)
    set(EXPECT present)
  else()
    set(EXPECT absent)
  endif()

  add_test(NAME ${TARGET} COMMAND ${CMAKE_COMMAND} -DNM=${CMAKE_NM} "-DOBJECTS=$<TARGET_OBJECTS:${TARGET}>"
                                  -DEXPECT=${EXPECT} -P ${CMAKE_CURRENT_SOURCE_DIR}/check_refcount_symbols.cmake
  )

  set_tests_properties(${TARGET} PROPERTIES LABELS "codegen")
endforeach()
