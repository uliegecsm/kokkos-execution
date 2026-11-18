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

/**
 * @test Check copies and moves count while using @c stdexec::transform_sender and @c stdexec::connect.
 *
 * @note All senders use @ref Tests::Utils::Functors::Counter taken by @c lvalue.
 */
TEST(ThenTest, copies_and_moves_in_transform_sender_and_connect) {
    using counter_t = Tests::Utils::Functors::Counter;

    exec::single_thread_context stc{};

    const counter_t counter;

    //! @c stdexec::transform_sender with length of 1.
    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr_from_lvalue_functor_1 =
        stdexec::transform_sender(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), stdexec::env<>{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 1, 0, 2)

    //! @c stdexec::connect with length of 1.
    counter_t::reset();

    [[maybe_unused]]
    const auto opstate_from_lvalue_functor_1 =
        stdexec::connect(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 1, 0, 2)

    //! @c stdexec::transform_sender with length of 2.
    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr_from_lvalue_functor_2 = stdexec::transform_sender(
        stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter), stdexec::env<>{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 2, 0, 5)

    //! @c stdexec::connect with length of 2.
    counter_t::reset();

    [[maybe_unused]]
    const auto opstate_from_lvalue_functor_2 = stdexec::connect(
        stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter),
        Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 2, 0, 5)

    //! @c stdexec::transform_sender with length of 3.
    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr_from_lvalue_functor_3 = stdexec::transform_sender(
        stdexec::then(stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter), counter),
        stdexec::env<>{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 3, 0, 9)

    //! @c stdexec::connect with length of 3.
    counter_t::reset();

    [[maybe_unused]]
    const auto opstate_from_lvalue_functor_3 = stdexec::connect(
        stdexec::then(stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter), counter),
        Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 3, 0, 9)

    //! @c stdexec::transform_sender with length of 4.
    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr_from_lvalue_functor_4 = stdexec::transform_sender(
        stdexec::then(
            stdexec::then(
                stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter), counter),
            counter),
        stdexec::env<>{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 4, 0, 14)

    //! @c stdexec::connect with length of 4.
    counter_t::reset();

    [[maybe_unused]]
    const auto opstate_from_lvalue_functor_4 = stdexec::connect(
        stdexec::then(
            stdexec::then(
                stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter), counter),
            counter),
        Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 4, 0, 14)

    //! @c stdexec::transform_sender with length of 5.
    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr_from_lvalue_functor_5 = stdexec::transform_sender(
        stdexec::then(
            stdexec::then(
                stdexec::then(
                    stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter), counter),
                counter),
            counter),
        stdexec::env<>{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 5, 0, 20)

    //! @c stdexec::connect with length of 5.
    counter_t::reset();

    [[maybe_unused]]
    const auto opstate_from_lvalue_functor_5 = stdexec::connect(
        stdexec::then(
            stdexec::then(
                stdexec::then(
                    stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter), counter),
                counter),
            counter),
        Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 5, 0, 20)

    //! @c stdexec::transform_sender with length of 6.
    counter_t::reset();

    [[maybe_unused]]
    const auto trnsfrmd_sndr_from_lvalue_functor_6 = stdexec::transform_sender(
        stdexec::then(
            stdexec::then(
                stdexec::then(
                    stdexec::then(
                        stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter),
                        counter),
                    counter),
                counter),
            counter),
        stdexec::env<>{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 6, 0, 27)

    //! @c stdexec::connect with length of 6.
    counter_t::reset();

    [[maybe_unused]]
    const auto opstate_from_lvalue_functor_6 = stdexec::connect(
        stdexec::then(
            stdexec::then(
                stdexec::then(
                    stdexec::then(
                        stdexec::then(stdexec::then(stdexec::schedule(stc.get_scheduler()), counter), counter),
                        counter),
                    counter),
                counter),
            counter),
        Tests::Utils::SinkReceiver{});

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 6, 0, 27)
}

} // namespace Tests
