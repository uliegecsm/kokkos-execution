# Detect if the compiler supports C++23 mdspan.

file(
  WRITE "${CMAKE_BINARY_DIR}/check_mdspan.cpp"
  [[
#if __has_include(<mdspan>)
#include <mdspan>
#endif

#ifndef __cpp_lib_mdspan
#error "C++23 mdspan is not supported"
#endif

int main() {}
]]
)

try_compile(
  CXX_COMPILER_SUPPORTS_CXX23_MDSPAN SOURCES "${CMAKE_BINARY_DIR}/check_mdspan.cpp" CXX_STANDARD 23
                                             CXX_STANDARD_REQUIRED ON OUTPUT_VARIABLE MDSPAN_CHECK_OUTPUT
)

if(CXX_COMPILER_SUPPORTS_CXX23_MDSPAN)
  message(STATUS "C++23 mdspan is supported.")
else()
  message(WARNING "C++23 mdspan is not supported:\n${MDSPAN_CHECK_OUTPUT}")
endif()
