#include "gtest/gtest.h"

#include "kokkos-execution/impl/compile_time_optional.hpp"

/**
 * @addtogroup unittests
 *
 * Compile-time optional
 * ---------------------
 *
 * This group of tests check the behavior of @ref Kokkos::Execution::Impl::CompileTimeOptional.
 *
 * The tests can be found in @ref tests/impl/test_compile_time_optional.cpp.
 */

namespace Tests::Impl {

struct TestType {
    constexpr TestType(int first, int second) noexcept
        : m_c(first + second) {
    }

    int m_c;
};

using disabled_t = Kokkos::Execution::Impl::CompileTimeOptional<false, TestType>;
using enabled_t = Kokkos::Execution::Impl::CompileTimeOptional<true, TestType>;

//! @test A disabled @ref Kokkos::Execution::Impl::CompileTimeOptional is empty and trivially copyable.
consteval bool test_disabled_is_empty_and_trivially_copyable() {
    return std::is_empty_v<disabled_t> && std::is_trivially_copyable_v<disabled_t>;
}

static_assert(test_disabled_is_empty_and_trivially_copyable());

//! @test The size of an enabled @ref Kokkos::Execution::Impl::CompileTimeOptional is equal to the size of its value type.
consteval bool test_size_of_enabled() {
    return sizeof(enabled_t) == sizeof(typename enabled_t::value_type);
}

static_assert(test_size_of_enabled());

//! @test Both enabled and disabled versions are constructible from the same arguments as their value type.
consteval bool test_constructible() {
    static_assert(
        std::is_nothrow_constructible_v<enabled_t, int, int> && std::is_nothrow_constructible_v<disabled_t, int, int>);

    return enabled_t(42, 666).value.m_c == 708;
}

static_assert(test_constructible());

//! Because we need at least one discoverable runtime test per unit.
TEST(CompileTimeOptionalTest, value) {
    ASSERT_EQ(enabled_t(42, 666).value.m_c, 708);
}

} // namespace Tests::Impl
