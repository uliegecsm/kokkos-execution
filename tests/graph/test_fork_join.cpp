#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
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

class ForkJoinTest
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

    ForkJoinTest()
        : default_device_handle(TEST_EXECUTION_SPACE{}) {
    }
   protected:
    device_handle_t default_device_handle;
};

//! @test A @c exec::fork_join with one branch and no sender before or after the fork.
TEST_F(ForkJoinTest, single_branch) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | Tests::Utils::check_scheduler_type<stdexec::set_value_t, typename ForkJoinTest::scheduler_t>()
              | exec::fork_join(
                    THEN_INCREMENT(data)
                    | Tests::Utils::check_scheduler_type<stdexec::set_value_t, typename ForkJoinTest::scheduler_t>())
              | Tests::Utils::check_scheduler_type<stdexec::set_value_t, typename ForkJoinTest::scheduler_t>();

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    static_assert(
        std::same_as<
            stdexec::__demangle_t<sndr_t>,
            Tests::Utils::CheckSchedulerTypeSender<
                Tests::Utils::basic_sender_t<
                    exec::fork_join_t,
                    stdexec::__tuple<stdexec::__clsur::__compose<
                        stdexec::__closure<stdexec::then_t, Tests::Utils::Functors::Increment<view_s_t>>,
                        stdexec::__closure<
                            Tests::Utils::check_scheduler_type_t<stdexec::set_value_t, typename ForkJoinTest::scheduler_t>
                        >
                    >>,
                    Tests::Utils::CheckSchedulerTypeSender<
                        typename ForkJoinTest::schedule_sender_t,
                        stdexec::set_value_t,
                        typename ForkJoinTest::scheduler_t
                    >
                >,
                stdexec::set_value_t,
                typename ForkJoinTest::scheduler_t
            >
        >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0), MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

//! @test A @c exec::fork_join with 3 branches and no sender before or after the fork.
TEST_F(ForkJoinTest, three_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | exec::fork_join(THEN_INCREMENT_ATOMIC(data), THEN_INCREMENT_ATOMIC(data), THEN_INCREMENT_ATOMIC(data));

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
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(2)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(3))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}
#if true
//! @test Use @c exec::fork_join with a diamond topology.
TEST_F(ForkJoinTest, diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | exec::fork_join(THEN_INCREMENT_ATOMIC(data), THEN_INCREMENT_ATOMIC(data))
              | stdexec::then(functor_t{.prev = 6, .value = 3, .data = data.data()});

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
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
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
#endif
#if false
//! @test Use @c exec::fork_join with a double diamond topology.
TEST_F(ForkJoinTest, double_diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::LoadCheckAddFunctor<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | exec::fork_join(THEN_INCREMENT_ATOMIC(data), THEN_INCREMENT_ATOMIC(data))
              | stdexec::then(functor_t{.prev = 6, .value = 3, .data = data.data()})
              | exec::fork_join(BULK_SUM_INDICES(3, data), BULK_SUM_INDICES(3, data))
              | stdexec::then(functor_t{.prev = 15, .value = 5, .data = data.data()});

    using sndr_t = decltype(sndr);

    static_assert(std::same_as<
                  completion_signatures_of_t<sndr_t>,
                  stdexec::completion_signatures<stdexec::set_error_t(std::exception_ptr), stdexec::set_value_t()>
    >);

    static_assert(std::same_as<stdexec::__demangle_t<sndr_t>, int>);

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 20);
}
#endif

// /**
//  * @test Use @c exec::fork_join after a @c stdexec::continues_on.
//  *
//  * Inspired by https://github.com/NVIDIA/stdexec/issues/1823.
//  */
// TEST_F(ForkJoinTest, after_a_continues_on) {
//     const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

//     const context_t gctx{exec};

//     auto sndr =
//         ::stdexec::just() | ::stdexec::continues_on(gctx.get_scheduler())
//         | ::exec::fork_join(
//             ::stdexec::continues_on(gctx.get_scheduler())
//             | ::stdexec::then(
//                 ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 0, .value = 3, .data = data.data()}));

//     std::vector<::testing::Matcher<variant_t>> matchers{
//         MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
//         MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
//         KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

//     if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
//         if (execution_space{} != exec) {
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_ENDOF_SINK_SYNC(execution_space{}));
//         }
//     }
//     matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

//     ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

//     ASSERT_THAT(
//         recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
//             ::stdexec::sync_wait(std::move(sndr));                       // NOLINT(performance-move-const-arg)
//         }),
//         ::testing::ElementsAreArray(matchers));

//     ASSERT_EQ(data(), 3);
// }

// //! @test Use @c exec::fork_join before a @c stdexec::continues_on.
// TEST_F(ForkJoinTest, before_a_continues_on) {
//     const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

//     const context_t gctx{exec};

//     auto sndr =
//         ::stdexec::schedule(gctx.get_scheduler())
//         | ::exec::fork_join(
//             ::stdexec::continues_on(gctx.get_scheduler())
//             | ::stdexec::then(
//                 ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 0, .value = 3, .data = data.data()}))
//         | ::stdexec::continues_on(gctx.get_scheduler())
//         | ::stdexec::then(
//             ::tests::utils::LoadCheckAddFunctor<int, on_device>{.prev = 3, .value = 3, .data = data.data()});

//     std::vector<::testing::Matcher<variant_t>> matchers{
//         MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
//         MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
//         KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

//     if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
//         if (execution_space{} != exec) {
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(execution_space{}));
//         }
//     }
//     matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

//     ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

//     ASSERT_THAT(
//         recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
//             ::stdexec::sync_wait(std::move(sndr));                       // NOLINT(performance-move-const-arg)
//         }),
//         ::testing::ElementsAreArray(matchers));

//     ASSERT_EQ(data(), 6);
// }

// //! @test Nest @c exec::fork_join.
// TEST_F(ForkJoinTest, nested) {
//     const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

//     const context_t grc{exec};

//     using functor_t = ::tests::utils::LoadCheckAddFunctor<value_t, on_device>;

//     auto scheduler = grc.get_scheduler();

//     auto sndr = ::stdexec::schedule(scheduler) | ::stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
//               | ::exec::fork_join(
//                     ::stdexec::continues_on(scheduler)
//                         | ::exec::fork_join(
//                             ::stdexec::continues_on(scheduler) | ADD_THEN_ATOMIC,
//                             ::stdexec::continues_on(scheduler) | ADD_THEN_ATOMIC),
//                     ::stdexec::continues_on(scheduler) | ADD_THEN_ATOMIC)
//               | ::stdexec::then(functor_t{.prev = 7, .value = 5, .data = data.data()});

//     using outer_0 = ::stdexec::connect_result_t<decltype(sndr), ::tests::stdexec::SinkReceiver>;
//     static_assert(::stdexec::__is_instance_of<outer_0, Kokkos::Experimental::details::graph::ThenOpState>);

//     static_assert(impl::check_node_type<
//                   outer_0,
//                   const impl::then_node_t<execution_space, functor_t, impl::aggregate_node_t<execution_space>>&
//     >());

//     using outer_1 = typename outer_0::inner_opstate_t;
//     static_assert(::stdexec::__is_instance_of<outer_1, Kokkos::Experimental::details::graph::ForkJoinOpState>);

//     static_assert(impl::check_node_type<outer_1, const impl::aggregate_node_t<execution_space>&>());

//     std::vector<::testing::Matcher<variant_t>> matchers{
//         MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph instantiate")),
//         MATCHER_FOR_PROFILE_EVENT(dispatch_label(exec, "graph submit")),
//         KOKKOS_DEFAULTED_GRAPH_SUBMIT_FENCE(exec)};

//     if constexpr (::tests::kokkos_ext::impl::is_graph_defaulted<execution_space>) {
//         if (execution_space{} != exec) {
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(exec));
//             matchers.push_back(KOKKOS_DEFAULTED_GRAPH_SINK_SYNC(execution_space{}));
//         }
//     }
//     matchers.push_back(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

//     ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

//     ASSERT_THAT(
//         recorder_listener_t::record([sndr = std::move(sndr)]() mutable { ::stdexec::sync_wait(std::move(sndr)); }),
//         ::testing::ElementsAreArray(matchers));

//     ASSERT_EQ(data(), 12);
// }

} // namespace Tests::GraphImpl
