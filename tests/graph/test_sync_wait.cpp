#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_completion_scheduler_type.hpp"
#include "tests/utils/check_sync_wait.hpp"
#include "tests/utils/functors/store_thread_id.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c stdexec::sync_wait by @c Kokkos::Execution::GraphContext
 * ----------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext properly customizes
 * @c stdexec::sync_wait.
 *
 * The tests can be found in @ref tests/graph/test_sync_wait.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class SyncWaitTest
    : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<EventDiscardMatcher<TEST_EXECUTION_SPACE>, BeginFenceEvent>;
};

//! @test Check whether the sender can be nothrow applied.
static_assert(Tests::Utils::check_nothrow_apply_sender<
              Kokkos::Execution::GraphImpl::Domain,
              typename SyncWaitTest::schedule_sender_t
>());

/**
 * @test Check that the start scheduler that @c stdexec::sync_wait sets in the receiver environment
 *       is a @c run_loop scheduler on the thread on which it starts the operation state.
 */
TEST_F(SyncWaitTest, get_start_scheduler) {
    std::thread::id tid;

    const context_t gctx{exec};

    auto sndr = stdexec::read_env(stdexec::get_start_scheduler)
              | stdexec::let_value([&](auto schd) { return stdexec::schedule(schd) | THEN_STORE_THREAD_ID(&tid); })
              | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, stdexec::run_loop::scheduler>()
              | stdexec::continues_on(gctx.get_scheduler());

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, std::this_thread::get_id());
}

/**
 * @test Check that the completion-scheduler query for inline work resolves to
 *       the receiver's start scheduler, and that the work runs on the calling thread.
 */
TEST_F(SyncWaitTest, just) {
    std::thread::id tid;

    const context_t gctx{exec};

    auto sndr = stdexec::just() | THEN_STORE_THREAD_ID(&tid)
              | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, stdexec::run_loop::scheduler>()
              | stdexec::continues_on(gctx.get_scheduler());

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, std::this_thread::get_id());
}

//! @test Check that calling @c stdexec::sync_wait on a sender that does not have any operation in it will not result in a spurious fence.
TEST_F(SyncWaitTest, no_spurious_fence) {
    const context_t gctx{exec};

    auto sndr = stdexec::schedule(gctx.get_scheduler());

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            const auto value = stdexec::sync_wait(std::move(sndr));      // NOLINT(performance-move-const-arg)
            static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
            ASSERT_TRUE(value.has_value());
        }),
        testing::IsEmpty());
}

} // namespace Tests::GraphImpl
