if(CXX_COMPILER_SUPPORTS_CXX26_CONSTEXPR_EXCEPTION)
  add_one_basic_test(NAME constexpr_exception COMPILE_FEATURES cxx_std_26)
  add_one_basic_test(NAME constexpr_exception.failure COMPILE_FEATURES cxx_std_26)

  add_build_failure_test(
    tests_std_constexpr_exception.failure.compiler tests_std_constexpr_exception.failure
    "error: uncaught exception of type 'std::invalid_argument'\\; 'what\\(\\)': 'escaping'"
  )
endif()

if(CXX_COMPILER_SUPPORTS_CXX26_REFLECTION)
  add_one_basic_test(NAME reflection COMPILE_FEATURES cxx_std_26)
endif()

if(CXX_COMPILER_SUPPORTS_CXX23_MDSPAN)
  add_one_basic_test(NAME mdspan COMPILE_FEATURES cxx_std_23)
else()
  add_one_basic_test(NAME mdspan ADDITIONAL_LINK_LIBRARIES Kokkos::kokkoscore)
endif()
