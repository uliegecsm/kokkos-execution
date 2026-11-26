#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Check the behavior of @c stdexec::then
 * --------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::then.
 *
 * The tests can be found in @ref tests/stdexec/test_then.cpp.
 */

namespace Tests {

template <typename ViewType>
struct ThenSetValueAndSetErrorTest : public testing::Test { };

TYPED_TEST_SUITE_P(ThenSetValueAndSetErrorTest);

/**
 * @test Ensure that a schedule sender followed by a @c stdexec::then that
 *       may throw properly advertises a completion scheduler on both value and error channels.
 */
TYPED_TEST_P(ThenSetValueAndSetErrorTest, get_completion_scheduler) {
    TypeParam scheduler_provider{};

    auto scheduler = scheduler_provider.get_scheduler();

    using scheduler_t = decltype(scheduler);

    auto sndr = stdexec::schedule(scheduler) | stdexec::then([]() { throw std::runtime_error{"error"}; });

    using sndr_t = decltype(sndr);

    static_assert(
        stdexec::get_completion_signatures<sndr_t, stdexec::env<>>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});

    static_assert(stdexec::__sends<stdexec::set_value_t, sndr_t, stdexec::env<>>);
    static_assert(stdexec::__sends<stdexec::set_error_t, sndr_t, stdexec::env<>>);

    static_assert(Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_value_t, stdexec::env<>>);
    static_assert(Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_error_t, stdexec::env<>>);

    static_assert(
        std::same_as<stdexec::__completion_scheduler_of_t<stdexec::set_value_t, sndr_t, stdexec::env<>>, scheduler_t>);
    static_assert(
        std::same_as<stdexec::__completion_scheduler_of_t<stdexec::set_error_t, sndr_t, stdexec::env<>>, scheduler_t>);

    ASSERT_EQ(
        stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), stdexec::env<>{}), scheduler);
    ASSERT_EQ(
        stdexec::get_completion_scheduler<stdexec::set_error_t>(stdexec::get_env(sndr), stdexec::env<>{}), scheduler);

    ASSERT_THROW(stdexec::sync_wait(std::move(sndr)), std::runtime_error);
}

REGISTER_TYPED_TEST_SUITE_P(ThenSetValueAndSetErrorTest, get_completion_scheduler);

using then_set_value_and_set_error_types_t =
    testing::Types<experimental::execution::single_thread_context, experimental::execution::static_thread_pool>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    ThenSetValueAndSetErrorTestInstantiation,
    ThenSetValueAndSetErrorTest,
    then_set_value_and_set_error_types_t);

template <typename T>
struct ThenCopiesAndMovesTest : public testing::Test {
    using counter_t = Tests::Utils::Functors::Counter;

    static constexpr typename T::value_type N = T::value; // NOLINT(readability-identifier-length)

    template <typename T::value_type M>
    stdexec::sender auto make_sndr_impl() {
        if constexpr (M == 0) {
            return stdexec::schedule(stc.get_scheduler());
        } else {
            return stdexec::then(make_sndr_impl<M - 1>(), counter);
        }
    }

    stdexec::sender auto make_sndr() {
        return make_sndr_impl<N>();
    }

    experimental::execution::single_thread_context stc;
    counter_t counter;
};

TYPED_TEST_SUITE_P(ThenCopiesAndMovesTest);

/**
 * For a sender with @c N  @c stdexec::then sender adapters:
 * - copy each functor into the by-value parameter of @c stdexec::then (N copies)
 * - move each functor from that parameter into its adapter (N moves)
 * - move each child with its subtree into its parent (N * (N-1) / 2 moves)
 * - move the transformed sender into the result variable (N moves)
 *
 * @note All senders use @ref Tests::Utils::Functors::Counter taken by @c lvalue.
 */
TYPED_TEST_P(ThenCopiesAndMovesTest, transform_sender) {
    using counter_t = typename TestFixture::counter_t;

    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr = stdexec::transform_sender(this->make_sndr(), stdexec::env<>{});

    CHECK_COUNTER_COPIES_AND_MOVES(EXPECT, 0, this->N, 0, (this->N * (this->N + 3)) / 2)
}

/**
 * For a sender with @c N  @c stdexec::then sender adapters:
 * - copy each functor into the by-value parameter of @c stdexec::then (N copies)
 * - move each functor from that parameter into its adapter (N moves)
 * - move each child with its subtree into its parent (N * (N-1) / 2 moves)
 * - move each functor into the operation state (N moves)
 *
 * @note All senders use @ref Tests::Utils::Functors::Counter taken by @c lvalue.
 */
TYPED_TEST_P(ThenCopiesAndMovesTest, connect) {
    using counter_t = typename TestFixture::counter_t;

    counter_t::reset();

    [[maybe_unused]]
    const auto opstate = stdexec::connect(this->make_sndr(), Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_COPIES_AND_MOVES(EXPECT, 0, this->N, 0, (this->N * (this->N + 3)) / 2)
}

REGISTER_TYPED_TEST_SUITE_P(ThenCopiesAndMovesTest, transform_sender, connect);

using then_copies_and_moves_types_t = ::testing::Types<
    std::integral_constant<size_t, 1>,
    std::integral_constant<size_t, 2>,
    std::integral_constant<size_t, 3>,
    std::integral_constant<size_t, 4>,
    std::integral_constant<size_t, 5>,
    std::integral_constant<size_t, 6>
>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    ThenCopiesAndMovesTestInstantiation,
    ThenCopiesAndMovesTest,
    then_copies_and_moves_types_t);

} // namespace Tests
