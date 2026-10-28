# Detect if the compiler supports C++26 constexpr exceptions.

file(
  WRITE "${CMAKE_BINARY_DIR}/check_constexpr_exception.cpp"
  [[
#if !(defined(__cpp_constexpr_exceptions) && __cpp_constexpr_exceptions >= 202411L)
#error "C++26 constexpr exception is not supported."
#endif

int main() {}
]]
)

try_compile(
  CXX_COMPILER_SUPPORTS_CXX26_CONSTEXPR_EXCEPTION SOURCES "${CMAKE_BINARY_DIR}/check_constexpr_exception.cpp"
                                                          CXX_STANDARD 26 CXX_STANDARD_REQUIRED ON
  OUTPUT_VARIABLE CONSTEXPR_EXCEPTION_CHECK_OUTPUT
)

if(CXX_COMPILER_SUPPORTS_CXX26_CONSTEXPR_EXCEPTION)
  message(STATUS "C++26 constexpr exception is supported.")
else()
  message(WARNING "C++26 constexpr exception is not supported:\n${CONSTEXPR_EXCEPTION_CHECK_OUTPUT}")
endif()
