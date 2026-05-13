#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_sync_wait.hpp"
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
static_assert(Tests::Utils::check_nothrow_apply_sender<
              Kokkos::Execution::ExecutionSpaceImpl::Domain,
              typename SyncWaitTest::schedule_sender_t
>());

//! @test Check that calling @c stdexec::sync_wait on a sender that does not have any operation in it will not result in a spurious fence.
TEST_F(SyncWaitTest, sync_wait) {
    const context_t esc{exec};

    auto sndr = stdexec::schedule(esc.get_scheduler());

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            const auto value = stdexec::sync_wait(std::move(sndr));      // NOLINT(performance-move-const-arg)
            static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
            ASSERT_TRUE(value.has_value());
        }),
        testing::IsEmpty());
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
