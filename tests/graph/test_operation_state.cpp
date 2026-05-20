#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Operation state @c Kokkos::Execution::GraphImpl::OpState and related
 * --------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::GraphImpl::OpState
 * and its related types.
 *
 * The tests can be found in @ref tests/graph/test_operation_state.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class RemainsOnGraphForTest
    : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        ConjunctionMatcher<EventDiscardMatcher<TEST_EXECUTION_SPACE>, GraphEventDiscardMatcher<TEST_EXECUTION_SPACE>>,
        BeginFenceEvent,
        BeginParallelForEvent,
        AllocateDataEvent,
        DeallocateDataEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent,
        Kokkos::Execution::GraphImpl::GraphAddNodeEvent,
        Kokkos::Execution::GraphImpl::GraphCreateEvent,
        Kokkos::Execution::GraphImpl::GraphInstantiateEvent,
        Kokkos::Execution::GraphImpl::GraphSubmitEvent
    >;

    using sync_wait_rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;
};

//! @test Check @ref Kokkos::Execution::GraphImpl::remains_on_graph_for for a non-dependent sender on @ref Kokkos::Execution::GraphImpl::Domain.
TEST_F(RemainsOnGraphForTest, non_dependent_sender) {
    const context_t gctx{exec};

    stdexec::sender auto sndr = stdexec::schedule(gctx.get_scheduler()) | stdexec::then([]() { });

    using sndr_t = decltype(sndr);

    static_assert(!stdexec::dependent_sender<sndr_t>);

    using outer_t = stdexec::connect_result_t<sndr_t, sync_wait_rcvr_t>;
    using inner_t = typename outer_t::inner_opstate_t;

    static_assert(stdexec::__is_instance_of<outer_t, Kokkos::Execution::GraphImpl::OpState>);
    static_assert(
        stdexec::__is_instance_of<inner_t, Kokkos::Execution::GraphImpl::Scheduler<TEST_EXECUTION_SPACE>::OpState>);

    static_assert(Kokkos::Execution::GraphImpl::RemainsOnGraphFor<outer_t, TEST_EXECUTION_SPACE>::value);
    static_assert(Kokkos::Execution::GraphImpl::RemainsOnGraphFor<inner_t, TEST_EXECUTION_SPACE>::value);

    static_assert(Kokkos::Execution::GraphImpl::remains_on_graph_for<TEST_EXECUTION_SPACE, sndr_t, sync_wait_rcvr_t>);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
}

//! @test Check @ref Kokkos::Execution::GraphImpl::remains_on_graph_for for a dependent sender partly on @ref Kokkos::Execution::GraphImpl::Domain.
TEST_F(RemainsOnGraphForTest, dependent_sender_partly_on_graph_domain) {
    experimental::execution::single_thread_context stc{};
    const context_t gctx{exec};

    stdexec::sender auto sndr = stdexec::schedule(stc.get_scheduler()) | stdexec::then([]() { })
                              | stdexec::continues_on(gctx.get_scheduler()) | stdexec::then([]() { });

    using sndr_t = decltype(sndr);

    static_assert(stdexec::dependent_sender<sndr_t>);

    static_assert(!Kokkos::Execution::GraphImpl::remains_on_graph_for<TEST_EXECUTION_SPACE, sndr_t, sync_wait_rcvr_t>);

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
}

/**
 * @test Check @ref Kokkos::Execution::GraphImpl::remains_on_graph_for for a non-dependent sender on @ref Kokkos::Execution::GraphImpl::Domain with @c stdexec::continues_on.
 *
 * @bug It should not create two graphs.
 */
TEST_F(RemainsOnGraphForTest, non_dependent_sender_with_continues_on) {
    const context_t gctx{exec};

    stdexec::sender auto sndr = stdexec::schedule(gctx.get_scheduler()) | stdexec::then([]() { })
                              | stdexec::continues_on(gctx.get_scheduler()) | stdexec::then([]() { });

    using sndr_t = decltype(sndr);

    static_assert(Kokkos::Execution::GraphImpl::remains_on_graph_for<TEST_EXECUTION_SPACE, sndr_t, sync_wait_rcvr_t>);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle, nullptr),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(2)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
}

} // namespace Tests::GraphImpl
