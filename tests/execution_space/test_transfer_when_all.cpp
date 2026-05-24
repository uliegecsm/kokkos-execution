#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/kokkos.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Interaction of @c Kokkos::Execution::ExecutionSpaceContext with @c stdexec::when_all
 * ------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly interacts with
 * @c stdexec::when_all.
 *
 * The tests can be found in @ref tests/execution_space/test_when_all.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class TransferWhenAllTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        BeginParallelForEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
};

//! @test Two branches on different execution space instances A and B of the same type, followed by a @c stdexec::then on instance A.
TEST_F(TransferWhenAllTest, Y) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const context_t esc_A{exec_A}, esc_B{exec_B};

    auto w_a = stdexec::transfer_when_all(
        esc_A.get_scheduler(),
        stdexec::schedule(esc_A.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data),
        stdexec::schedule(esc_B.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data));

    static_assert(std::same_as<
                  decltype(stdexec::get_completion_domain<stdexec::set_value_t>(stdexec::get_env(w_a))),
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! As opposed to @c stdexec::when_all, @c stdexec::transfer_when_all has a completion scheduler.
    static_assert(std::same_as<
                  decltype(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(w_a))),
                  scheduler_t
    >);
    static_assert(Kokkos::Execution::ExecutionSpaceImpl::execution_space_completing_sender<decltype(w_a)>);

    auto sndr = std::move(w_a) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
    } else {
        ASSERT_THAT(recorded_events, [&]() {
            if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                    MATCHER_FOR_RECORD_EVENT(exec_B),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait")));
            } else {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "after dispatch")),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait")));
            }
        }());
    }

    ASSERT_EQ(data(), 3);
}

} // namespace Tests::ExecutionSpaceImpl
