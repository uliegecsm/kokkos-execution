#include "kokkos-utils/callbacks/RecorderListener.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c sync_wait by @c Kokkos::Execution::ExecutionSpaceContext
 * ----------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly customizes
 * @c sync_wait.
 *
 * The tests can be found in @ref tests/execution_space/test_sync_wait.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using ExecutionSpaceContextTest = Tests::Utils::ExecutionSpaceContextTest<execution_space>;

/**
 * @test Ensure that @c sync_wait is properly customized.
 *
 * Improperly customized @c sync_wait should result in a missing synchronization.
 */
TEST_F(ExecutionSpaceContextTest, sync_wait) {
    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler());

    Kokkos::utils::callbacks::Manager::initialize();

    ASSERT_THAT(
        Kokkos::utils::callbacks::RecorderListener<Kokkos::utils::callbacks::BeginFenceEvent>::record(
            [chain = std::move(chain)]() mutable {                       // NOLINT(performance-move-const-arg)
                const auto value = stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
                static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
                ASSERT_TRUE(value.has_value());
            }),
        testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    Kokkos::utils::callbacks::Manager::finalize();
}

//! @test Check that @ref Kokkos::Execution::ExecutionSpaceImpl::SyncWait properly rethrows if needed.
TEST_F(ExecutionSpaceContextTest, rethrows) {
    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler()) | stdexec::then(Tests::Utils::Functors::ThrowsWhenCopied{});

    ASSERT_THAT(
        Tests::Utils::Functors::MutableMoveToSyncWait{.sndr = std::move(chain)},
        testing::ThrowsMessage<std::runtime_error>(testing::StrEq("ThrowsWhenCopied: Throwing in copy constructor!")));
}

} // namespace Tests::ExecutionSpaceImpl
