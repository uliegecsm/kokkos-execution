if(CXX_COMPILER_SUPPORTS_CXX26_CONSTEXPR_EXCEPTION)
  add_one_basic_test(NAME constexpr_exception CXX_STANDARD 26)
  add_one_basic_test(NAME constexpr_exception.failure CXX_STANDARD 26)

  add_build_failure_test(
    tests_std_constexpr_exception.failure.compiler tests_std_constexpr_exception.failure
    "error: uncaught exception of type 'std::invalid_argument'\\; 'what\\(\\)': 'escaping'"
  )
endif()

if(CXX_COMPILER_SUPPORTS_CXX26_REFLECTION)
  add_one_basic_test(NAME reflection CXX_STANDARD 26)
endif()
