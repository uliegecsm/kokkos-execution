#include "gtest/gtest.h"

#include "kokkos-execution/reuse.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"

/**
 * @addtogroup unittests
 *
 * Sender adaptor @c Kokkos::Execution::reuse
 * ------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::reuse.
 *
 * The tests can be found in @ref tests/test_reuse.cpp.
 */

namespace Tests {

//! @test Check that the @c noexcept contract of @ref Kokkos::Execution::reuse is as expected.
consteval bool test_reuse_noexcept_contract() {
    using resource_a_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using resource_b_t = Tests::Utils::Functors::NoOp<false, true, false>;
    using inner_sndr_t = decltype(stdexec::just());

    static_assert(noexcept(Kokkos::Execution::reuse(std::declval<resource_a_t>(), std::declval<inner_sndr_t>())));

    static_assert(!noexcept(Kokkos::Execution::reuse(std::declval<resource_b_t&>(), std::declval<inner_sndr_t>())));

    static_assert(noexcept(
        stdexec::just() | Kokkos::Execution::reuse(std::declval<resource_a_t>(), std::declval<inner_sndr_t>())));

    return true;
}

static_assert(test_reuse_noexcept_contract());

//! @test Check decomposition of @ref Kokkos::Execution::Impl::ReuseSender into the algorithm tag, data, and child sender.
consteval bool test_reuse_sndr_decomposition() {
    using just_sndr_t = decltype(stdexec::just());
    using inner_sndr_t = decltype(stdexec::just() | stdexec::then([]() { }));

    using resource_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using reuse_sndr_t = Kokkos::Execution::Impl::ReuseSender<just_sndr_t, resource_t, inner_sndr_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(stdexec::__sender_for<reuse_sndr_t, Kokkos::Execution::reuse_t>);

    static_assert(
        std::same_as<stdexec::__data_of<reuse_sndr_t>, Kokkos::Execution::Impl::ReuseData<resource_t, inner_sndr_t>>);

    static_assert(stdexec::__nbr_children_of<reuse_sndr_t> == 1);
    static_assert(std::same_as<stdexec::__child_of<reuse_sndr_t>, just_sndr_t>);

    return true;
}

static_assert(test_reuse_sndr_decomposition());

//! @test Check @ref Kokkos::Execution::reuse_t closure object creation perfectly forwards the resource.
TEST(ReuseTest, closure_object_creation_perfectly_forwards_functor) {
    using counter_t = Tests::Utils::Functors::Counter;

    const auto sndr = stdexec::just();

    counter_t counter;
    counter_t::reset();

    [[maybe_unused]]
    const auto closure_from_lvalue_resource = Kokkos::Execution::reuse(counter, sndr);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 0);

    [[maybe_unused]]
    const auto closure_from_xvalue_resource = Kokkos::Execution::reuse(std::move(counter), sndr);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 1);

    [[maybe_unused]]
    const auto closure_from_prvalue_resource = Kokkos::Execution::reuse(counter_t{}, sndr);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 2);
}

} // namespace Tests
