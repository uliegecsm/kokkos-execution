#include "gtest/gtest.h"

#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

#include "kokkos-execution/parallel_for.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"

/**
 * @addtogroup unittests
 *
 * Sender adaptor @c Kokkos::Execution::parallel_for
 * -------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::parallel_for.
 *
 * The tests can be found in @ref tests/test_parallel_for.cpp.
 */

namespace Tests {

//! @test Check that the @c noexcept contract of @ref Kokkos::Execution::parallel_for is as expected.
consteval bool test_parallel_for_noexcept_contract() {
    using label_t = std::string;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;

    static_assert(noexcept(Kokkos::Execution::parallel_for(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_for(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, true, false>&>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_for(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, true>>())));

    static_assert(noexcept(
        stdexec::just()
        | Kokkos::Execution::parallel_for(
            std::declval<label_t>(),
            std::declval<policy_t>(),
            std::declval<Tests::Utils::Functors::NoOp<false, false, false>>())));

    return true;
}

static_assert(test_parallel_for_noexcept_contract());

//! @test Check decomposition of @ref Kokkos::Execution::Impl::ParallelForSender into the algorithm tag, data, and child sender.
consteval bool test_parallel_for_sndr_decomposition() {
    //! @c stdexec::just sender.
    using just_sndr_t = decltype(stdexec::just());

    //! Parallel for sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using pfor_sndr_t = Kokkos::Execution::Impl::ParallelForSender<just_sndr_t, label_t, functor_t, policy_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(stdexec::__sender_for<pfor_sndr_t, Kokkos::Execution::parallel_for_t>);

    static_assert(std::same_as<
                  stdexec::__data_of<pfor_sndr_t>,
                  Kokkos::Execution::Impl::ParallelForData<label_t, functor_t, policy_t>
    >);

    static_assert(stdexec::__nbr_children_of<pfor_sndr_t> == 1);
    static_assert(std::same_as<stdexec::__child_of<pfor_sndr_t>, just_sndr_t>);

    return true;
}

static_assert(test_parallel_for_sndr_decomposition());

//! @test Check @ref Kokkos::Execution::parallel_for closure object creation perfectly forwards the functor.
TEST(ParallelForTest, closure_object_creation_perfectly_forwards_functor) {
    using counter_t = Tests::Utils::Functors::Counter;

    const Kokkos::RangePolicy<TEST_EXECUTION_SPACE> policy(0, 1);
    counter_t counter;
    counter_t::reset();

    [[maybe_unused]]
    const auto closure_from_lvalue_functor = Kokkos::Execution::parallel_for("label", policy, counter);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 0);

    [[maybe_unused]]
    const auto closure_from_xvalue_functor = Kokkos::Execution::parallel_for("label", policy, std::move(counter));

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 1);

    [[maybe_unused]]
    const auto closure_from_prvalue_functor = Kokkos::Execution::parallel_for("label", policy, counter_t{});

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 2);
}

} // namespace Tests
