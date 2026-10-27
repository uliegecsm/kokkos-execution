# Detect if the compiler supports C++26 reflection. For GCC 16.0.1, it requires adding the -freflection flag, which in
# turn makes the __cpp_impl_reflection preprocessor definition visible.

file(
  WRITE "${CMAKE_BINARY_DIR}/check_reflection.cpp"
  [[
#ifndef __cpp_impl_reflection
#error "C++26 reflection is not supported"
#endif

int main() {}
]]
)

try_compile(
  CXX_COMPILER_SUPPORTS_CXX26_REFLECTION SOURCES
  "${CMAKE_BINARY_DIR}/check_reflection.cpp" CXX_STANDARD 26 CXX_STANDARD_REQUIRED ON
  COMPILE_DEFINITIONS -freflection
  OUTPUT_VARIABLE REFLECTION_CHECK_OUTPUT
)

if(CXX_COMPILER_SUPPORTS_CXX26_REFLECTION)
  message(STATUS "C++26 reflection is supported.")
else()
  message(WARNING "C++26 reflection is not supported:\n${REFLECTION_CHECK_OUTPUT}")
endif()
