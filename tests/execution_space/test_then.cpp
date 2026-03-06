#include "gtest/gtest.h"

#include "kokkos-execution/execution_space.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c stdexec::then by @c Kokkos::Execution::ExecutionSpaceContext
 * --------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly customizes
 * @c stdexec::then.
 *
 * The tests can be found in @ref tests/execution_space/test_then.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

/**
 * @test Check that @ref Kokkos::Execution::ExecutionSpaceContext does its duty well when used with @c stdexec::then
 *       within a chain started with @c stdexec::schedule.
 */
TEST(ThenTest, then_schedule) {
    using context_t = Kokkos::Execution::ExecutionSpaceContext<execution_space>;

    const execution_space exec{};

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler()) | stdexec::then([]() { });

    stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
}

} // namespace Tests::ExecutionSpaceImpl
