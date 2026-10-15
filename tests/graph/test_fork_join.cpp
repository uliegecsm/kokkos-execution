#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/check_node_type.hpp"
#include "tests/utils/check_scheduler_type.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/functors/tag_dispatch.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/just_stopped.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c exec::fork_join by @c Kokkos::Execution::GraphContext
 * -------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext properly customizes
 * @c exec::fork_join.
 *
 * The tests can be found in @ref tests/graph/test_fork_join.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(ForkJoinTest)
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
        Kokkos::Execution::GraphImpl::GraphAddAggregateNodeEvent,
        Kokkos::Execution::GraphImpl::GraphInstantiateEvent,
        Kokkos::Execution::GraphImpl::GraphSubmitEvent
    >;

    TEST_CATEGORY(ForkJoinTest)()
        : default_device_handle(TEST_EXECUTION_SPACE{}) {
    }
   protected:
    device_handle_t default_device_handle;
};

/**
 * @test A @c exec::fork_join with one branch and no sender before or after the fork.
 *
 * Because of @cite P4269R1, passing a single closure to @c exec::fork_join behaves as if removing
 * @c exec::fork_join entirely. Therefore, there shouldn't be any aggregate node created.
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), single_branch_no_aggregate) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr =
        stdexec::schedule(gctx.get_scheduler())
        | Tests::Utils::check_scheduler_type<stdexec::set_value_t, typename TEST_CATEGORY(ForkJoinTest)::scheduler_t>()
        | exec::fork_join(
            THEN_INCREMENT(data)
            | Tests::Utils::check_scheduler_type<
                stdexec::set_value_t,
                typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
            >())
        | Tests::Utils::check_scheduler_type<stdexec::set_value_t, typename TEST_CATEGORY(ForkJoinTest)::scheduler_t>();

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    static_assert(std::same_as<
                  stdexec::__demangle_t<sndr_t>,
                  Tests::Utils::CheckSchedulerTypeSender<
                      Tests::Utils::CheckSchedulerTypeSender<
                          stdexec::__basic_sender<
                              stdexec::then_t,
                              Tests::Utils::Functors::Increment<view_s_t>,
                              Tests::Utils::CheckSchedulerTypeSender<
                                  typename TEST_CATEGORY(ForkJoinTest)::schedule_sender_t,
                                  stdexec::set_value_t,
                                  typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
                              >
                          >::type,
                          stdexec::set_value_t,
                          typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
                      >,
                      stdexec::set_value_t,
                      typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
                  >
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

//! @test A @c exec::fork_join with 3 branches and no sender before or after the fork.
TEST_F(TEST_CATEGORY(ForkJoinTest), three_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | exec::fork_join(
                    THEN_INCREMENT_ATOMIC(Device, data),
                    THEN_INCREMENT_ATOMIC(Device, data),
                    THEN_INCREMENT_ATOMIC(Device, data));

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(2)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(3))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

//! @test Use @c exec::fork_join with a diamond topology.
TEST_F(TEST_CATEGORY(ForkJoinTest), diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    using expt_node_t = Kokkos::Experimental::GraphNodeRef<
        TEST_EXECUTION_SPACE,
        Kokkos::Impl::GraphNodeThenImpl<TEST_EXECUTION_SPACE, Kokkos::Experimental::ThenPolicy<>, functor_t>,
        Kokkos::Experimental::GraphNodeRef<
            TEST_EXECUTION_SPACE,
            Kokkos::Impl::GraphNodeAggregateDefaultImpl<TEST_EXECUTION_SPACE>
        >
    >;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | exec::fork_join(THEN_INCREMENT_ATOMIC(Device, data), THEN_INCREMENT_ATOMIC(Device, data))
              | stdexec::then(functor_t{.prev = 6, .value = 3, .data = data.data()})
              | Tests::Utils::check_node_type<expt_node_t>();

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(2)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(3))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_AGGREGATE_NODE_OF(recorded_events.at(4))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 9);
}

//! @test Use @c exec::fork_join with a double diamond topology.
TEST_F(TEST_CATEGORY(ForkJoinTest), double_diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | exec::fork_join(THEN_INCREMENT_ATOMIC(Device, data), THEN_INCREMENT_ATOMIC(Device, data))
              | stdexec::then(functor_t{.prev = 6, .value = 3, .data = data.data()})
              | exec::fork_join(BULK_SUM_INDICES(3, data), BULK_SUM_INDICES(3, data))
              | stdexec::then(functor_t{.prev = 15, .value = 5, .data = data.data()});

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(2)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(3))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_AGGREGATE_NODE_OF(recorded_events.at(4))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(5))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(5))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(6)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(7))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_AGGREGATE_NODE_OF(recorded_events.at(8))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 20);
}

/**
 * @test Use @c exec::fork_join after a @c stdexec::continues_on.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/issues/1823.
 * Following @cite P4269R1, it should not be too much of an issue since the test passes a single closure.
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), after_a_continues_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr =
        stdexec::just() | stdexec::continues_on(gctx.get_scheduler())
        | exec::fork_join(
            stdexec::continues_on(gctx.get_scheduler())
            | stdexec::then(
                Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 0, .value = 3, .data = data.data()}));

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test Use @c exec::fork_join before a @c stdexec::continues_on.
 *
 * @todo This should create a single @c Kokkos::Graph.
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), before_a_continues_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr =
        stdexec::schedule(gctx.get_scheduler())
        | exec::fork_join(
            stdexec::continues_on(gctx.get_scheduler())
            | stdexec::then(
                Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 0, .value = 3, .data = data.data()}))
        | stdexec::continues_on(gctx.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 3, .value = 3, .data = data.data()});

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(2), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(2))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(2)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 6);
}

//! @test Nesting @c exec::fork_join creates a single graph.
TEST_F(TEST_CATEGORY(ForkJoinTest), nested) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | exec::fork_join(
                    exec::fork_join(THEN_INCREMENT_ATOMIC(Device, data), THEN_INCREMENT_ATOMIC(Device, data)),
                    THEN_INCREMENT_ATOMIC(Device, data))
              | stdexec::then(functor_t{.prev = 7, .value = 5, .data = data.data()});

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    for (const auto& e: recorded_events)
        std::visit([](const auto& v) { std::cout << v << std::endl; }, e);

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(2)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(3))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_AGGREGATE_NODE_OF(recorded_events.at(4)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(5))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_AGGREGATE_NODE_OF(recorded_events.at(6))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 12);
}

} // namespace Tests::GraphImpl
