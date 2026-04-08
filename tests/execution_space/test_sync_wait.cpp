#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/stdexec.hpp"

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

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class SyncWaitTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
};

//! @test Check whether the sender can be nothrow applied.
consteval bool test_nothrow_apply_sender() {
    static_assert(Tests::Utils::has_completion_signatures<
                  typename SyncWaitTest::schedule_sender_t,
                  stdexec::__mset<stdexec::set_value_t()>
    >);
    static_assert(Tests::Utils::has_nothrow_apply_sender<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::sync_wait_t,
                  typename SyncWaitTest::schedule_sender_t
    >);
    static_assert(
        !Tests::Utils::has_nothrow_apply_sender<stdexec::sync_wait_t, typename SyncWaitTest::schedule_sender_t>);
    return true;
}
static_assert(test_nothrow_apply_sender());

/**
 * @test Ensure that @c sync_wait is properly customized.
 *
 * Improperly customized @c sync_wait should result in a missing synchronization.
 */
TEST_F(SyncWaitTest, sync_wait) {
    const context_t esc{exec};

    auto sndr = stdexec::schedule(esc.get_scheduler());

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            const auto value = stdexec::sync_wait(std::move(sndr));      // NOLINT(performance-move-const-arg)
            static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
            ASSERT_TRUE(value.has_value());
        }),
        testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
}

//! @test Check that the @c stdexec::sync_wait of @ref Kokkos::Execution::ExecutionSpaceContext properly rethrows if needed.
TEST_F(SyncWaitTest, rethrows) {
    const context_t esc{exec};

    auto sndr = stdexec::schedule(esc.get_scheduler()) | stdexec::then(Tests::Utils::Functors::ThrowsWhenCopied{});

    ASSERT_THAT(
        Tests::Utils::Functors::MutableMoveToSyncWait{.sndr = std::move(sndr)},
        testing::ThrowsMessage<std::runtime_error>(testing::StrEq("ThrowsWhenCopied: Throwing in copy constructor!")));
}

} // namespace Tests::ExecutionSpaceImpl
