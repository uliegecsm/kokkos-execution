#include "gtest/gtest.h"

#include "kokkos-execution/impl/dispatch_label.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Kokkos::Execution::Impl::dispatch_label
 * ----------------------------------------------------
 *
 * This group of tests check the behavior of @ref Kokkos::Execution::Impl::dispatch_label.
 *
 * The tests can be found in @ref tests/impl/test_dispatch_label.cpp.
 */

namespace Tests::Impl {

constexpr char suffix[] = " with the given suffix";

constexpr std::string_view label_of_exec = Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name();

constexpr auto label_as_array = Kokkos::Execution::Impl::dispatch_label_v<TEST_EXECUTION_SPACE, suffix>;

constexpr std::string_view label_as_view = Kokkos::Execution::Impl::dispatch_label<TEST_EXECUTION_SPACE, suffix>();

//! @test Check that @ref Kokkos::Execution::Impl::dispatch_label_v works as expected.
consteval bool test_dispatch_label_v() {
    static_assert(
        std::same_as<decltype(label_as_array), const std::array<char, label_of_exec.size() + sizeof(suffix) - 1>>);

    return true;
}

static_assert(test_dispatch_label_v());

//! @test Check that @ref Kokkos::Execution::Impl::dispatch_label works as expected.
consteval bool test_dispatch_label() {
    static_assert(std::same_as<decltype(label_as_view), const std::string_view>);
    static_assert(label_as_view.starts_with(label_of_exec));
    static_assert(label_as_view.ends_with(suffix));
    static_assert(label_as_view.size() == label_of_exec.size() + sizeof(suffix) - 1);

    return true;
}

static_assert(test_dispatch_label());

//! @test Check whether @ref Kokkos::Execution::Impl::dispatch_label is @c noexcept.
consteval bool test_dispatch_label_noexcept() {
    static_assert(noexcept(Kokkos::Execution::Impl::dispatch_label<TEST_EXECUTION_SPACE, suffix>()));

    return true;
}

static_assert(test_dispatch_label_noexcept());

//! Because we need at least one discoverable runtime test per unit.
TEST(DispatchLabelTest, debug) {
    std::ostringstream oss_of_exec, oss_as_view;
    oss_of_exec << label_of_exec;
    oss_as_view << label_as_view;

    ASSERT_EQ(oss_as_view.str(), oss_of_exec.str() + suffix);
}

} // namespace Tests::Impl
