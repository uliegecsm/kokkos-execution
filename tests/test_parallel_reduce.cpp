#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/parallel_reduce.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/functors/no_op.hpp"

/**
 * @addtogroup unittests
 *
 * Sender adaptor @c Kokkos::Execution::parallel_reduce
 * ----------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::parallel_reduce.
 *
 * The tests can be found in @ref tests/test_parallel_reduce.cpp.
 */

namespace Tests {

//! @test Check that the @c noexcept contract of @ref Kokkos::Execution::parallel_reduce is as expected.
consteval bool test_parallel_reduce_noexcept_contract() {
    using label_t = std::string;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using reducer_t = Kokkos::View<int, TEST_EXECUTION_SPACE>;

    static_assert(noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
        std::declval<reducer_t>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, true, false>&>(),
        std::declval<reducer_t>())));

    static_assert(!noexcept(Kokkos::Execution::parallel_reduce(
        std::declval<label_t>(),
        std::declval<policy_t>(),
        std::declval<Tests::Utils::Functors::NoOp<false, false, true>>(),
        std::declval<reducer_t>())));

    static_assert(noexcept(
        stdexec::just()
        | Kokkos::Execution::parallel_reduce(
            std::declval<label_t>(),
            std::declval<policy_t>(),
            std::declval<Tests::Utils::Functors::NoOp<false, false, false>>(),
            std::declval<reducer_t>())));

    return true;
}

static_assert(test_parallel_reduce_noexcept_contract());

//! @test Check decomposition of @ref Kokkos::Execution::Impl::ParallelReduceSender into the algorithm tag, data, and child sender.
consteval bool test_parallel_reduce_sndr_decomposition() {
    //! @c stdexec::just sender.
    using just_sndr_t = decltype(stdexec::just());

    //! Parallel reducer sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using reducer_t = Kokkos::View<int, TEST_EXECUTION_SPACE>;
    using pred_sndr_t =
        Kokkos::Execution::Impl::ParallelReduceSender<just_sndr_t, label_t, functor_t, policy_t, reducer_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(stdexec::__sender_for<pred_sndr_t, Kokkos::Execution::parallel_reduce_t>);

    static_assert(std::same_as<
                  stdexec::__data_of<pred_sndr_t>,
                  Kokkos::Execution::Impl::ParallelReduceData<label_t, functor_t, policy_t, reducer_t>
    >);

    static_assert(stdexec::__nbr_children_of<pred_sndr_t> == 1);
    static_assert(std::same_as<stdexec::__child_of<pred_sndr_t>, just_sndr_t>);

    return true;
}

static_assert(test_parallel_reduce_sndr_decomposition());

//! @test Check @ref Kokkos::Execution::parallel_reduce closure object creation perfectly forwards the functor.
TEST(ParallelReduceTest, closure_object_creation_perfectly_forwards_functor) {
    using counter_t = Tests::Utils::Functors::Counter;

    const Kokkos::RangePolicy<TEST_EXECUTION_SPACE> policy(0, 1);
    const Kokkos::View<int, TEST_EXECUTION_SPACE> reducer;
    counter_t counter;
    counter_t::reset();

    [[maybe_unused]]
    const auto closure_from_lvalue_functor = Kokkos::Execution::parallel_reduce("label", policy, counter, reducer);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 0);

    [[maybe_unused]]
    const auto closure_from_xvalue_functor =
        Kokkos::Execution::parallel_reduce("label", policy, std::move(counter), reducer);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 1);

    [[maybe_unused]]
    const auto closure_from_prvalue_functor = Kokkos::Execution::parallel_reduce("label", policy, counter_t{}, reducer);

    ASSERT_EQ(counter_t::copy_constructions, 1);
    ASSERT_EQ(counter_t::move_constructions, 2);
}

} // namespace Tests
