#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"
#include "kokkos-execution/graph.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Interoperability of @c Kokkos::Execution::GraphContext with other schedulers
 * ----------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext can be used in
 * conjunction with other schedulers.
 *
 * The tests can be found in @ref tests/graph/test_inter_op.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class InterOpTest
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
};

TEST_F(InterOpTest, transition_to_same_execution_space_instance_does_not_synchronize) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};
    const Kokkos::Execution::ExecutionSpaceContext ectx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | stdexec::continues_on(ectx.get_scheduler())
              | stdexec::then(functor_t{.prev = 4, .value = 3, .data = data.data()});

    using sndr_t = decltype(sndr);

    static_assert(
        std::same_as<
            stdexec::__demangle_t<stdexec::transform_sender_result_t<sndr_t, stdexec::env<>>>,
            Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<
                stdexec::then_t,
                Tests::Utils::basic_sender_t<
                    stdexec::continues_on_t,
                    Kokkos::Execution::ExecutionSpaceImpl::Scheduler<TEST_EXECUTION_SPACE>,
                    Tests::Utils::basic_sender_t<
                        stdexec::schedule_from_t,
                        stdexec::__,
                        Tests::Utils::basic_sender_t<stdexec::then_t, functor_t, typename InterOpTest::schedule_sender_t>
                    >
                >,
                std::string_view,
                Kokkos::Execution::ExecutionSpaceImpl::ThenWrapper<functor_t>,
                Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::LaunchBounds<1>>
            >
        >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(recorded_events, testing::SizeIs(5));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 7);
}

} // namespace Tests::GraphImpl
