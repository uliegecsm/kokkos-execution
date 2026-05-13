#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_sync_wait.hpp"
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

//! @test Check that calling @c stdexec::sync_wait on a sender that does not have any operation in it will not result in a spurious fence.
TEST_F(SyncWaitTest, sync_wait) {
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
