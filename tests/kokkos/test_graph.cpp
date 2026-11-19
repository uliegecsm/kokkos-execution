#include "gtest/gtest.h"

#include "Kokkos_Macros.hpp"
#if defined(KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES)
import kokkos.graph;
#else
#    include "Kokkos_Graph.hpp"
#endif

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "tests/utils/category.hpp"
#include "tests/utils/functors/counter.hpp"

/**
 * @addtogroup unittests
 *
 * Check the behavior of @c Kokkos::Experimental::Graph
 * ----------------------------------------------------
 *
 * This group of tests check the behavior of @c Kokkos::Experimental::Graph.
 *
 * The tests can be found in @ref tests/kokkos/test_graph.cpp.
 */

namespace Tests {

class TEST_CATEGORY(GraphTest)
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<TEST_EXECUTION_SPACE> { };

/**
 * @test Check how many copies and moves are involved during graph lifecycle.
 *
 * Update the expected counts once https://github.com/kokkos/kokkos/pull/9780 is merged.
 */
TEST_F(TEST_CATEGORY(GraphTest), copies_and_moves) {
    using counter_t = Tests::Utils::Functors::Counter;

    counter_t counter;

    counter_t::reset();

    [[maybe_unused]]
    const auto graph_from_lvalue_functor = Kokkos::Experimental::create_graph(
        Kokkos::Experimental::get_device_handle(exec), [&](const auto& root) { root.then_parallel_for(42, counter); });

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 3, 0, 0)

    counter_t::reset();

    [[maybe_unused]]
    const auto graph_from_xvalue_functor =
        Kokkos::Experimental::create_graph(Kokkos::Experimental::get_device_handle(exec), [&](const auto& root) {
            root.then_parallel_for(42, std::move(counter));
        });

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 2, 0, 1)

    counter_t::reset();

    [[maybe_unused]]
    const auto graph_from_prvalue_functor =
        Kokkos::Experimental::create_graph(Kokkos::Experimental::get_device_handle(exec), [](const auto& root) {
            root.then_parallel_for(42, counter_t{});
        });

    CHECK_COUNTER_MOVES_AND_COPIES(EXPECT, 0, 2, 0, 1)
}
} // namespace Tests
