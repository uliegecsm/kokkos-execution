#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CATEGORY_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CATEGORY_HPP

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define TEST_CATEGORY_IMPL(_name_, _suffix_) GTEST_CONCAT_TOKEN_(GTEST_CONCAT_TOKEN_(_name_, _), _suffix_)

#if !defined(__DOXYGEN__)
#    define TEST_CATEGORY(_name_) TEST_CATEGORY_IMPL(_name_, TEST_EXECUTION_SPACE_FOR_CATEGORY)
#else
#    define TEST_CATEGORY(_name_) _name_
#endif
// NOLINTEND(cppcoreguidelines-macro-usage)

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CATEGORY_HPP
