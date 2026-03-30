# This file contains setup w.r.t. compiler flags related to warnings.
include_guard(GLOBAL)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  # For GNU GCC, warnings are listed at https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html.
  add_compile_options(
    -Wall
    -Wextra
    -Wdangling-else
    -Wpedantic
    -Wshadow
    -Wswitch-default
    -Wsuggest-override
    -Woverloaded-virtual
    -Werror
    -pedantic-errors
  )

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # For Clang, diagnostic flags are listed at https://clang.llvm.org/docs/DiagnosticsReference.html.
    add_compile_options(
      -Werror=unused-private-field -Werror=unused-lambda-capture -Werror=unused-member-function
      -Werror=delete-non-virtual-dtor -Werror=dangling-gsl -Wno-error=unknown-cuda-version
    )
  endif()

else()
  message(FATAL_ERROR "Your compiler ${CMAKE_CXX_COMPILER_ID} is not supported for additional warning options.")
endif()
