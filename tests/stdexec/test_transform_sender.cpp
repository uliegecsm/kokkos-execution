#include "gtest/gtest.h"

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/counter.hpp"
#include "tests/utils/sink_receiver.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::transform_sender
 * --------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::transform_sender.
 *
 * The tests can be found in @ref tests/stdexec/test_transform_sender.cpp.
 */

namespace Tests {

/**
 * @test Check that @c stdexec::transform_sender moves the incoming sender
 *       only once using @ref Tests::Utils::Functors::Counter.
 *
 * See also:
 *  * https://github.com/NVIDIA/stdexec/pull/2429
 *  * https://github.com/NVIDIA/stdexec/pull/2430
 */
TEST(TransformSender, transform_sender_moves_once) {
    using counter_t = Tests::Utils::Functors::Counter;
    using functor_t = std::identity;

    counter_t::reset();

    {
        [[maybe_unused]]
        const auto sndr = stdexec::then(
            stdexec::then(stdexec::then(stdexec::just(counter_t{}), functor_t{}), functor_t{}), functor_t{});

        ASSERT_EQ(counter_t::default_constructions, 1);
        ASSERT_EQ(counter_t::destructions, 4);
        ASSERT_EQ(counter_t::copy_constructions, 0);
        ASSERT_EQ(counter_t::copy_assignments, 0);
        ASSERT_EQ(counter_t::move_constructions, 4);
        ASSERT_EQ(counter_t::move_assignments, 0);
    }

    counter_t::reset();

    [[maybe_unused]]
    const auto opstate = stdexec::connect(
        stdexec::then(stdexec::then(stdexec::then(stdexec::just(counter_t{}), functor_t{}), functor_t{}), functor_t{}),
        Tests::Utils::SinkReceiver{});

    ASSERT_EQ(counter_t::default_constructions, 1);
    ASSERT_EQ(counter_t::destructions, 5);
    ASSERT_EQ(counter_t::copy_constructions, 0);
    ASSERT_EQ(counter_t::copy_assignments, 0);
    ASSERT_EQ(counter_t::move_constructions, 5);
    ASSERT_EQ(counter_t::move_assignments, 0);
}

//! @test Same test as @ref TransformSender_transform_sender_moves_once_Test, but in a constant expression.
consteval bool test_transform_sender_moves_once() {
    using counter_t = Tests::Utils::Functors::CounterRef;

    counter_t::counters counts{};

    [[maybe_unused]]
    const auto opstate = stdexec::connect(stdexec::just(counter_t(counts)), Tests::Utils::SinkReceiver{});
    return counts.constructions == 1 && counts.destructions == 2 && counts.copy_constructions == 0
        && counts.copy_assignments == 0 && counts.move_constructions == 2 && counts.move_assignments == 0;
}

static_assert(test_transform_sender_moves_once());

} // namespace Tests
