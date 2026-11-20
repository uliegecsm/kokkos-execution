#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/sink_receiver.hpp"

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

    exec::single_thread_context stc;
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
