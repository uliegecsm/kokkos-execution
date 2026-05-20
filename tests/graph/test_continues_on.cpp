#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/check_continues_on.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/just_stopped.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c stdexec::continues_on by @c Kokkos::Execution::GraphContext
 * -------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext properly customizes
 * @c stdexec::continues_on.
 *
 * The tests can be found in @ref tests/graph/test_continues_on.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(ContinuesOnTest)
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

//! @test Check traits of the sender created by the customized @c stdexec::schedule_from.
consteval bool test_schedule_from_sndr_traits() {
    using schd_sndr_t = typename TEST_CATEGORY(ContinuesOnTest)::schedule_sender_t;

    using schedule_from_sndr_t = Kokkos::Execution::GraphImpl::ScheduleFromSender<TEST_EXECUTION_SPACE, schd_sndr_t>;

    static_assert(stdexec::__nothrow_connectable<schedule_from_sndr_t, Tests::Utils::SinkReceiver>);
    static_assert(stdexec::__nothrow_connectable<const schedule_from_sndr_t&, Tests::Utils::SinkReceiver>);

    return true;
}
static_assert(test_schedule_from_sndr_traits());

//! @test Check traits of the sender created by the customized @c stdexec::continues_on.
consteval bool test_continues_on_sndr_traits() {
    using schd_sndr_t = typename TEST_CATEGORY(ContinuesOnTest)::schedule_sender_t;

    using continues_on_sndr_t = Kokkos::Execution::GraphImpl::ContinuesOnSender<
        typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t,
        schd_sndr_t
    >;

    static_assert(stdexec::__nothrow_connectable<continues_on_sndr_t, Tests::Utils::SinkReceiver>);
    static_assert(stdexec::__nothrow_connectable<const continues_on_sndr_t&, Tests::Utils::SinkReceiver>);

    static_assert(Tests::Utils::check_continues_on<typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t>());

    static_assert(
        std::same_as<
            stdexec::connect_result_t<continues_on_sndr_t&&, typename TEST_CATEGORY(ContinuesOnTest)::sync_wait_rcvr_t>,
            Kokkos::Execution::GraphImpl::ContinuesOnOpState<
                typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t,
                typename TEST_CATEGORY(ContinuesOnTest)::schedule_sender_t,
                typename TEST_CATEGORY(ContinuesOnTest)::sync_wait_rcvr_t
            >
        >);
    static_assert(std::same_as<
                  stdexec::connect_result_t<
                      const continues_on_sndr_t&,
                      typename TEST_CATEGORY(ContinuesOnTest)::sync_wait_rcvr_t
                  >,
                  Kokkos::Execution::GraphImpl::ContinuesOnOpState<
                      typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t const &,
                      typename TEST_CATEGORY(ContinuesOnTest)::schedule_sender_t const &,
                      typename TEST_CATEGORY(ContinuesOnTest)::sync_wait_rcvr_t
                  >
    >);

    return true;
}
static_assert(test_continues_on_sndr_traits());

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_nothrow_transformable() {
    using continues_on_sndr_t =
        decltype(stdexec::just() | stdexec::continues_on(std::declval<typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t>()));

    static_assert(std::same_as<
                  stdexec::__demangle_t<continues_on_sndr_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::continues_on_t,
                      typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t,
                      Tests::Utils::basic_sender_t<
                          stdexec::schedule_from_t,
                          stdexec::__,
                          Tests::Utils::basic_sender_t<stdexec::just_t, stdexec::__tuple<>>
                      >
                  >
    >);

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  continues_on_sndr_t&&,
                  stdexec::env<>
    >);

    using schedule_from_sndr_t = decltype(stdexec::schedule_from(
        stdexec::schedule(std::declval<typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t>())));

    static_assert(std::same_as<
                  stdexec::__demangle_t<schedule_from_sndr_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::schedule_from_t,
                      stdexec::__,
                      typename TEST_CATEGORY(ContinuesOnTest)::schedule_sender_t
                  >
    >);

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  schedule_from_sndr_t&&,
                  stdexec::env<>
    >);

    return true;
}
static_assert(test_sndr_nothrow_transformable());

/**
 * @test This test ensures that the customization will create two distinct graphs, because
 *       of the portion of the sender that is outside of @ref Kokkos::Execution::GraphImpl::Domain.
 */
TEST_F(TEST_CATEGORY(ContinuesOnTest), then_continues_on_single_thread_context_continues_on_then) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};
    experimental::execution::single_thread_context stc{};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | stdexec::continues_on(stc.get_scheduler()) | stdexec::continues_on(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 4, .value = 3, .data = data.data()});

    using inner_sndr_t = Tests::Utils::basic_sender_t<
        stdexec::continues_on_t,
        typename TEST_CATEGORY(ContinuesOnTest)::scheduler_t,
        Tests::Utils::basic_sender_t<
            stdexec::schedule_from_t,
            stdexec::__,
            Tests::Utils::basic_sender_t<
                stdexec::continues_on_t,
                stdexec::run_loop::scheduler,
                Tests::Utils::basic_sender_t<
                    stdexec::schedule_from_t,
                    stdexec::__,
                    Tests::Utils::basic_sender_t<
                        stdexec::then_t,
                        functor_t,
                        typename TEST_CATEGORY(ContinuesOnTest)::schedule_sender_t
                    >
                >
            >
        >
    >;

    using transform_sender_result_t =
        stdexec::transform_sender_result_t<decltype(sndr), stdexec::env_of_t<sync_wait_rcvr_t>>;
    static_assert(std::same_as<
                  stdexec::__demangle_t<transform_sender_result_t>,
                  Kokkos::Execution::GraphImpl::ThenSender<TEST_EXECUTION_SPACE, inner_sndr_t, functor_t>
    >);

    using connect_result_t = stdexec::connect_result_t<decltype(sndr), sync_wait_rcvr_t>;
    static_assert(std::same_as<
                  stdexec::__demangle_t<connect_result_t>,
                  Kokkos::Execution::GraphImpl::OpState<
                      inner_sndr_t,
                      sync_wait_rcvr_t,
                      Kokkos::Execution::GraphImpl::ThenClosure<TEST_EXECUTION_SPACE, functor_t>
                  >
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

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
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(2)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 7);
}

/**
 * @test Using @c stdexec::continues_on from a scheduler to itself creates a single graph.
 *
 * @bug It should create a single graph.
 */
TEST_F(TEST_CATEGORY(ContinuesOnTest), transition_to_same_graph_scheduler_instance) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | stdexec::continues_on(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 4, .value = 3, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

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

    ASSERT_EQ(data(), 7);
}

/**
 * @test Using @c stdexec::continues_on to transition from a scheduler to another scheduler of the same type creates a single graph.
 *
 * The graph is submitted on the execution space instance of the first scheduler, and the fence must also occur
 * on that execution space instance.
 *
 * @bug It should create a single graph.
 */
TEST_F(TEST_CATEGORY(ContinuesOnTest), transition_to_another_graph_scheduler_instance_same_type) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    Tests::Utils::show_exec_space_id(exec_A, "exec_A");
    Tests::Utils::show_exec_space_id(exec_B, "exec_B");

    const context_t gctx_A{exec_A}, gctx_B{exec_B};
    const device_handle_t device_handle_A{exec_A}, device_handle_B{exec_B};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx_A.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | stdexec::continues_on(gctx_B.get_scheduler())
              | stdexec::then(functor_t{.prev = 4, .value = 3, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    if constexpr (Kokkos::Execution::Impl::has_exec_wait_event<TEST_EXECUTION_SPACE>) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle_A, nullptr),
                MATCHER_FOR_GRAPH_CREATE(device_handle_B),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_B, nullptr),
                MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(0)),
                MATCHER_FOR_RECORD_EVENT(exec_A),
                MATCHER_FOR_WAIT_EXEC_EVENT(exec_B, recorded_events.at(5)),
                MATCHER_FOR_GRAPH_SUBMIT(exec_B, recorded_events.at(2)),
                MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "sync_wait"))));
    } else {
        if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle_A, nullptr),
                    MATCHER_FOR_GRAPH_CREATE(device_handle_B),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_B, nullptr),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(0)),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_B, recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "sync_wait"))));
        } else {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle_A, nullptr),
                    MATCHER_FOR_GRAPH_CREATE(device_handle_B),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_B, nullptr),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(0)),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "dependency")),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_B, recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "sync_wait"))));
        }
    }

    ASSERT_EQ(data(), 7);
}

/**
 * @test Using @c stdexec::continues_on to transition from a scheduler to another scheduler of the same type (and back) creates a single graph.
 *
 * This test is similar to @ref Tests::GraphImpl::ContinuesOnTest_transition_to_another_graph_scheduler_instance_same_type_Test, but transitions
 * back to the original scheduler instance. The graph is submitted on the execution space instance of the first scheduler,
 * and the fence is on that instance too.
 *
 * @bug It should create a single graph.
 */
TEST_F(TEST_CATEGORY(ContinuesOnTest), transition_to_another_graph_scheduler_instance_and_back_same_type) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    Tests::Utils::show_exec_space_id(exec_A, "exec_A");
    Tests::Utils::show_exec_space_id(exec_B, "exec_B");

    const context_t gctx_A{exec_A}, gctx_B{exec_B};
    const device_handle_t device_handle_A{exec_A}, device_handle_B{exec_B};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx_A.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | stdexec::continues_on(gctx_B.get_scheduler())
              | stdexec::then(functor_t{.prev = 4, .value = 3, .data = data.data()})
              | stdexec::continues_on(gctx_A.get_scheduler())
              | stdexec::then(functor_t{.prev = 7, .value = 2, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    if constexpr (Kokkos::Execution::Impl::has_exec_wait_event<TEST_EXECUTION_SPACE>) {
        ASSERT_THAT(recorded_events, testing::SizeIs(14));
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle_A, nullptr),
                MATCHER_FOR_GRAPH_CREATE(device_handle_B),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_B, nullptr),
                MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(4), device_handle_A, nullptr),
                MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(0)),
                MATCHER_FOR_RECORD_EVENT(exec_A),
                MATCHER_FOR_WAIT_EXEC_EVENT(exec_B, recorded_events.at(7)),
                MATCHER_FOR_GRAPH_SUBMIT(exec_B, recorded_events.at(2)),
                MATCHER_FOR_RECORD_EVENT(exec_B),
                MATCHER_FOR_WAIT_EXEC_EVENT(exec_A, recorded_events.at(10)),
                MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(4)),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
    } else {
        if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle_A, nullptr),
                    MATCHER_FOR_GRAPH_CREATE(device_handle_B),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_B, nullptr),
                    MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(4), device_handle_A, nullptr),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(0)),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_B, recorded_events.at(2)),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(4)),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
        } else {
            ASSERT_THAT(recorded_events, testing::SizeIs(12));
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle_A, nullptr),
                    MATCHER_FOR_GRAPH_CREATE(device_handle_B),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_B, nullptr),
                    MATCHER_FOR_GRAPH_CREATE(device_handle_A),
                    MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(4), device_handle_A, nullptr),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(0)),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "dependency")),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_B, recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "dependency")),
                    MATCHER_FOR_GRAPH_SUBMIT(exec_A, recorded_events.at(4)),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
        }
    }

    ASSERT_EQ(data(), 9);
}

//! @test Using @c stdexec::continues_on from a scheduler to another scheduler of another type creates distinct graphs.
TEST_F(TEST_CATEGORY(ContinuesOnTest), transition_to_another_graph_scheduler_instance_and_back_different_type) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const Kokkos::DefaultHostExecutionSpace exec_h{};

    const Kokkos::Execution::GraphContext gctx_h{exec_h};
    const context_t gctx{exec};

    const auto device_handle_h = Kokkos::Experimental::get_device_handle(exec_h);

    using functor_device_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;
    using functor_host_t = Tests::Utils::Functors::LoadCheckAdd<value_t, false>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_device_t{.prev = 0, .value = 4, .data = data.data()})
              | stdexec::continues_on(gctx_h.get_scheduler())
              | stdexec::then(functor_host_t{.prev = 4, .value = 3, .data = data.data()})
              | stdexec::continues_on(gctx.get_scheduler())
              | stdexec::then(functor_device_t{.prev = 7, .value = 2, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    if (Tests::Utils::are_same_instances(exec, exec_h)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_GRAPH_CREATE(device_handle),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
                MATCHER_FOR_GRAPH_CREATE(device_handle_h),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_h, nullptr),
                MATCHER_FOR_GRAPH_CREATE(device_handle),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(4), device_handle, nullptr),
                MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
                MATCHER_FOR_GRAPH_SUBMIT(exec_h, recorded_events.at(2)),
                MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(4)),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    } else {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_GRAPH_CREATE(device_handle),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
                MATCHER_FOR_GRAPH_CREATE(device_handle_h),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(2), device_handle_h, nullptr),
                MATCHER_FOR_GRAPH_CREATE(device_handle),
                MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(4), device_handle, nullptr),
                MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "dependency")),
                MATCHER_FOR_GRAPH_SUBMIT(exec_h, recorded_events.at(2)),
                MATCHER_FOR_BEGIN_FENCE(exec_h, dispatch_label(exec_h, "dependency")),
                MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(4)),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    }

    ASSERT_EQ(data(), 9);
}

} // namespace Tests::GraphImpl
