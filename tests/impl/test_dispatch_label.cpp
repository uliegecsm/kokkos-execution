#include "gtest/gtest.h"

#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/type_info.hpp"

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

constexpr std::string_view label_of_exec = Kokkos::Execution::Impl::display_string_of<TEST_EXECUTION_SPACE>();

constexpr std::string_view label_as_view = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(TEST_EXECUTION_SPACE, suffix);

#if !defined(__cpp_impl_reflection)
//! @test Check that @ref Kokkos::Execution::Impl::dispatch_label_v works as expected.
consteval bool test_dispatch_label_v() {
    constexpr auto label_as_array = Kokkos::Execution::Impl::dispatch_label_v<TEST_EXECUTION_SPACE, suffix>;

    static_assert(
        std::same_as<decltype(label_as_array), const std::array<char, label_of_exec.size() + sizeof(suffix) - 1>>);

    return true;
}

static_assert(test_dispatch_label_v());
#endif

//! @test Check that @ref Kokkos::Execution::Impl::dispatch_label works as expected.
consteval bool test_dispatch_label() {
    static_assert(
        std::same_as<decltype(KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(TEST_EXECUTION_SPACE, suffix)), std::string_view>);
    static_assert(std::same_as<decltype(label_as_view), const std::string_view>);
    static_assert(label_as_view.starts_with(label_of_exec));
    static_assert(label_as_view.ends_with(suffix));
    static_assert(label_as_view.size() == label_of_exec.size() + sizeof(suffix) - 1);

    return true;
}

static_assert(test_dispatch_label());

//! @test Check whether @ref Kokkos::Execution::Impl::dispatch_label is @c noexcept.
consteval bool test_dispatch_label_noexcept() {
    static_assert(noexcept(KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(TEST_EXECUTION_SPACE, suffix)));

    return true;
}

static_assert(test_dispatch_label_noexcept());

constexpr std::string_view view_a = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(TEST_EXECUTION_SPACE, suffix);
constexpr std::string_view view_b = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(TEST_EXECUTION_SPACE, suffix);

/**
 * @test Check that @ref Kokkos::Execution::Impl::dispatch_label does not create duplicated static storage strings
 *       for the same arguments.
 *
 * @note According to the possible implementation given in @cite P3491R3 (§3.4),
 *       two evaluatations of @c std::define_static_string with equal contents point to the same storage.
 */
TEST(DispatchLabelTest, deduplicate) {
    static_assert(static_cast<const void*>(view_a.data()) == static_cast<const void*>(view_b.data()));

    ASSERT_EQ(static_cast<const void*>(view_a.data()), static_cast<const void*>(view_b.data()));
}

} // namespace Tests::Impl
