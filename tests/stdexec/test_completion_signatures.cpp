#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::completion_signatures
 * -------------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::completion_signatures.
 *
 * The tests can be found in @ref tests/stdexec/test_completion_signatures.cpp.
 */

namespace Tests {

consteval bool test_completion_signatures_operator_equal_checks_for_set_equality() {
    constexpr auto empty = stdexec::completion_signatures<>{};
    constexpr auto sigs1 =
        stdexec::completion_signatures<stdexec::set_value_t(int), stdexec::set_error_t(std::exception_ptr)>{};
    constexpr auto sigs2 =
        stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr), stdexec::set_value_t(int)>{};
    constexpr auto sigs3 = stdexec::completion_signatures<
        stdexec::set_error_t(std::exception_ptr),
        stdexec::set_value_t(int),
        stdexec::set_error_t(std::exception_ptr)
    >{};
    constexpr auto sigs4 = stdexec::completion_signatures<
        stdexec::set_error_t(std::exception_ptr),
        stdexec::set_value_t(int),
        stdexec::set_stopped_t()
    >{};

    static_assert(sigs1 != empty);
    static_assert(sigs1 == sigs2);
    static_assert(sigs1 == sigs3);
    static_assert(sigs1 != sigs4);

    return true;
}

/**
 * @test Check that the equality operator for @c stdexec::completion_signatures checks for set equality.
 *
 * See also:
 *  * https://github.com/NVIDIA/stdexec/blob/ead186b1d8db3ebe37a946ff84a6ce08bf795153/include/stdexec/__detail/__completion_signatures.hpp#L419-L426
 */
TEST(completion_signatures, operator_equal_checks_for_set_equality) {
    static_assert(test_completion_signatures_operator_equal_checks_for_set_equality());
}

} // namespace Tests
