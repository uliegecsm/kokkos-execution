#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/repeat_until.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Use @c Kokkos::Execution::GraphContext with @c experimental:execution::repeat_until
 * -----------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext properly interacts with
 * @c experimental:execution::repeat_until.
 *
 * The tests can be found in @ref tests/graph/test_repeat_until.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(RepeatEffectUntilTest)
    : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        ConjunctionMatcher<EventDiscardMatcher<TEST_EXECUTION_SPACE>, GraphEventDiscardMatcher<TEST_EXECUTION_SPACE>>,
        BeginFenceEvent,
        BeginParallelForEvent,
        Kokkos::Execution::GraphImpl::GraphAddNodeEvent,
        Kokkos::Execution::GraphImpl::GraphCreateEvent,
        Kokkos::Execution::GraphImpl::GraphInstantiateEvent,
        Kokkos::Execution::GraphImpl::GraphSubmitEvent
    >;
};

/**
 * @test Check that @ref Kokkos::Execution::GraphContext can be properly embedded in a @c experimental:execution::repeat_until.
 *
 * @warning The underlying @c Kokkos::Graph is rebuilt each time, since the sender is re-connected at each submission.
 */
TEST_F(TEST_CATEGORY(RepeatEffectUntilTest), works) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    unsigned int guard = 0;

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        experimental::execution::repeat_until(
            stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT(data)
            | stdexec::continues_on(stdexec::inline_scheduler{})
            | stdexec::then([&guard]() -> bool { return (++guard) >= 3; })));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(4), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(4))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(4)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(8), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(8))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(8)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 3);
}

} // namespace Tests::GraphImpl
