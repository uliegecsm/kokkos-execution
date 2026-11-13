#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Macros.hpp"
#include "impl/Kokkos_UnbindIndexArgument.hpp"

#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/check_completion_scheduler_type.hpp"
#include "tests/utils/check_node_type.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/graph.hpp"
#include "tests/utils/graph_context.hpp"
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
};

//! @test Check traits of sender returned by @c exec::fork_join when customized for @ref Kokkos::Execution::GraphImpl::Domain.
consteval bool test_sndr_traits() {
    using schd_sndr_t = typename TEST_CATEGORY(ForkJoinTest)::schedule_sender_t;

    using functor_t = Tests::Utils::Functors::Increment<TEST_CATEGORY(ForkJoinTest)::view_s_t, true>;
    using then_clsr_t = decltype(stdexec::then(std::declval<functor_t>()));

    using fork_join_sndr_t =
        decltype(std::declval<schd_sndr_t>() | exec::fork_join(std::declval<then_clsr_t>(), std::declval<then_clsr_t>()));

    using transformed_fork_join_sndr_t = stdexec::transform_sender_result_t<fork_join_sndr_t, stdexec::env<>>;

    //! Models the graph completing sender concept.
    static_assert(Kokkos::Execution::GraphImpl::graph_completing_sender<transformed_fork_join_sndr_t>);

    //! Does not model the dispatching sender concept.
    static_assert(!Kokkos::Execution::Impl::dispatching_sender<transformed_fork_join_sndr_t>);

    return true;
}

static_assert(test_sndr_traits());

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_nothrow_transformable() {
    using schd_sndr_t = typename TEST_CATEGORY(ForkJoinTest)::schedule_sender_t;

    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using then_clsr_t = decltype(stdexec::then(std::declval<functor_t>()));

    using fork_join_sndr_t =
        decltype(std::declval<schd_sndr_t>() | exec::fork_join(std::declval<then_clsr_t>(), std::declval<then_clsr_t>()));

    static_assert(std::same_as<
                  fork_join_sndr_t,
                  stdexec::__sexpr<exec::fork_join_t, stdexec::__tuple<then_clsr_t, then_clsr_t>, schd_sndr_t>
    >);

    //! @todo It should be nothrow transformable in this case.
    static_assert(!stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  fork_join_sndr_t&&,
                  stdexec::env<>
    >);

    using functor_maythrow_on_move_t = Tests::Utils::Functors::NoOp<false, false, true>;
    using then_clsr_maythrow_on_move_t = decltype(stdexec::then(std::declval<functor_maythrow_on_move_t>()));

    using fork_join_maythrow_on_move_sndr_t =
        decltype(std::declval<schd_sndr_t>() | exec::fork_join(std::declval<then_clsr_maythrow_on_move_t>(), std::declval<then_clsr_maythrow_on_move_t>()));

    static_assert(!stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  fork_join_maythrow_on_move_sndr_t&&,
                  stdexec::env<>
    >);

    return true;
}

static_assert(test_sndr_nothrow_transformable());

//! @test Check that it cannot be nothrow-connected.
consteval bool test_sndr_nothrow_connectable() {
    //! Upon @c stdexec::connect, the graph will be constructed.
    static_assert(!std::is_nothrow_constructible_v<Kokkos::Experimental::Graph<TEST_EXECUTION_SPACE>>);

    using schd_sndr_t = typename TEST_CATEGORY(ForkJoinTest)::schedule_sender_t;

    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using then_clsr_t = decltype(stdexec::then(std::declval<functor_t>()));

    using fork_join_sndr_t =
        decltype(std::declval<schd_sndr_t>() | exec::fork_join(std::declval<then_clsr_t>(), std::declval<then_clsr_t>()));

    static_assert(!stdexec::__nothrow_connectable<fork_join_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}

static_assert(test_sndr_nothrow_connectable());

/**
 * @test A @c exec::fork_join with one branch and no sender before or after the fork.
 *
 * @verbatim
 * schedule(gctx) | fork_join(then)
 * @endverbatim
 *
 * Because of @cite P4269R1, passing a single closure to @c exec::fork_join behaves as if removing
 * @c exec::fork_join entirely.
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), single_branch_no_aggregate) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using expt_node_t = Kokkos::Experimental::GraphNodeRef<
        TEST_EXECUTION_SPACE,
        Kokkos::Impl::GraphNodeThenImpl<
            TEST_EXECUTION_SPACE,
            Kokkos::Experimental::ThenPolicy<>,
            Tests::Utils::Functors::Increment<view_s_t>
        >,
        Kokkos::Experimental::GraphNodeRef<TEST_EXECUTION_SPACE, Kokkos::Experimental::GraphNodeRootTag>
    >;

    constexpr Tests::Utils::check_completion_scheduler_type_t<
        stdexec::set_value_t,
        typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
    >
        check_completion_scheduler_type;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | exec::fork_join(
                    check_completion_scheduler_type() | THEN_INCREMENT(data) | check_completion_scheduler_type())
              | check_completion_scheduler_type() | Tests::Utils::check_node_type<expt_node_t>();

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t>
    >);

    static_assert(std::same_as<
                  sndr_t,
                  Tests::Utils::CheckNodeTypeSender<
                      expt_node_t,
                      Tests::Utils::CheckCompletionSchedulerTypeSender<
                          Tests::Utils::CheckCompletionSchedulerTypeSender<
                              stdexec::then_t::__sender<
                                  Tests::Utils::Functors::Increment<view_s_t>,
                                  Tests::Utils::CheckCompletionSchedulerTypeSender<
                                      typename TEST_CATEGORY(ForkJoinTest)::schedule_sender_t,
                                      stdexec::set_value_t,
                                      typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
                                  >
                              >,
                              stdexec::set_value_t,
                              typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
                          >,
                          stdexec::set_value_t,
                          typename TEST_CATEGORY(ForkJoinTest)::scheduler_t
                      >
                  >
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test A @c exec::fork_join with 3 branches and no sender before or after the fork.
 *
 * @verbatim
 *                               / -- then_atomic -- \
 * schedule(gctx) | fork_join --   -- then_atomic --   --
 *                               \ -- then_atomic -- /
 * @endverbatim
 */
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
            MATCHER_FOR_GRAPH_CREATE(device_handle),
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
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test Use @c exec::fork_join with a diamond topology.
 *
 * @verbatim
 *                                      / -- then_atomic -- \
 * schedule(gctx) | then | fork_join --                       -- then
 *                                      \ -- then_atomic -- /
 * @endverbatim
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), diamond) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    using expt_node_t = Kokkos::Experimental::GraphNodeRef<
        TEST_EXECUTION_SPACE,
        Kokkos::Impl::GraphNodeThenImpl<TEST_EXECUTION_SPACE, Kokkos::Experimental::ThenPolicy<>, functor_t>,
        Tests::Utils::aggregate_node_t<TEST_EXECUTION_SPACE>
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
            MATCHER_FOR_GRAPH_CREATE(device_handle),
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
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 9);
}

/**
 * @test Use @c exec::fork_join with a double diamond topology.
 *
 * @verbatim
 *                                      / -- then_atomic -- \                        / -- then_atomic -- \
 * schedule(gctx) | then | fork_join --                       -- then | fork_join --                       -- then
 *                                      \ -- then_atomic -- /                        \ -- then_atomic -- /
 * @endverbatim
 */
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
            MATCHER_FOR_GRAPH_CREATE(device_handle),
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
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 20);
}

/**
 * @test Use @c exec::fork_join after a @c stdexec::continues_on.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/issues/1823.
 *
 * @note Following @cite P4269R1, it should not be too much of an issue since the test passes a single closure.
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
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

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
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(2), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(2))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(2)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 6);
}

/**
 * @test Nesting @c exec::fork_join creates a single graph.
 *
 * @verbatim
 *                                      / -- then_atomic ------------------------- \
 * schedule(gctx) | then | fork_join --                   / -- then_atomic -- \      -- then
 *                                      \ -- fork_join --                       -- /
 *                                                        \ -- then_atomic -- /
 * @endverbatim
 */
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

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
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
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 12);
}

/**
 * @test A @c stdexec::fork_join with branches that are not remaining in the graph is allowed.
 *
 * @verbatim
 *                                      / -- continues_on(stc) | then_atomic | continues_on(gctx) -- \
 * schedule(gctx) | pfor | fork_join --                                                                -- pfor
 *                                      \ -- then_atomic ------------------------------------------- /
 * @endverbatim
 *
 * Since the underlying @c when_all does not "remain on graph", this case must create two graphs.
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), mixed_branches) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t gctx{exec};
    exec::single_thread_context stc{};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;
    using wrapper_t = Kokkos::Impl::UnbindIndexArgument<functor_t>;
    using rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | Kokkos::Execution::parallel_for(
                    "before fork",
                    1,
                    wrapper_t{
                        functor_t{.prev = 0, .value = 4, .data = data.data()}
    })
              | exec::fork_join(
                    stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                        | stdexec::continues_on(gctx.get_scheduler()),
                    THEN_INCREMENT_ATOMIC(System, data))
              | Kokkos::Execution::parallel_for(
                    "after join", 1, wrapper_t{functor_t{.prev = 6, .value = 5, .data = data.data()}});

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t, stdexec::env_of_t<rcvr_t>>
    >);

    static_assert(stdexec::sender_in<sndr_t, rcvr_t>);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_t, stdexec::env_of_t<rcvr_t>>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_RECORD_EVENTS_BUT_THREADS_MAY_THROW_ON_SYNC_WAIT_SKIP(recorder_listener_t, sndr)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0),
                device_handle,
                MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0)),
                "before fork"),
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(2),
                device_handle,
                MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(2)),
                dispatch_label(exec, "then")),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(2), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(3)), "after join"),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, "inner when_all does not remain on graph"),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(2)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 11);
}

/**
 * @test A @c stdexec::fork_join with branches that are not remaining in the graph and no operation before @c exec::fork_join
 *       is allowed and does not create spurious graphs.
 *
 * There is still the @c stdexec::schedule on @ref Kokkos::Execution::GraphContext before the fork point.
 *
 * @verbatim
 *                               / -- continues_on(stc) | then_atomic | continues_on(gctx) -- \
 * schedule(gctx) | fork_join --                                                                -- pfor
 *                               \ -- then_atomic ------------------------------------------- /
 * @endverbatim
 *
 * Since the underlying @c when_all does not "remain on graph", this case must create two graphs.
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), mixed_branches_nothing_before_but_schedule_is_graph_domain) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t gctx{exec};
    exec::single_thread_context stc{};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;
    using wrapper_t = Kokkos::Impl::UnbindIndexArgument<functor_t>;
    using rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | exec::fork_join(
                    stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                        | stdexec::continues_on(gctx.get_scheduler()),
                    THEN_INCREMENT_ATOMIC(System, data))
              | Kokkos::Execution::parallel_for(
                    "after join",
                    1,
                    wrapper_t{
                        functor_t{.prev = 2, .value = 5, .data = data.data()}
    });

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t, stdexec::env_of_t<rcvr_t>>
    >);

    static_assert(stdexec::sender_in<sndr_t, rcvr_t>);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_t, stdexec::env_of_t<rcvr_t>>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_RECORD_EVENTS_BUT_THREADS_MAY_THROW_ON_SYNC_WAIT_SKIP(recorder_listener_t, sndr)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0),
                device_handle,
                MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0)),
                dispatch_label(exec, "then")),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1)), "after join"),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 7);
}

/**
 * @test A @c stdexec::fork_join with branches that are not remaining in the graph and no operation before @c exec::fork_join
 *       is allowed and does not create spurious graphs.
 *
 * The @c stdexec::schedule before the fork point is not on @ref Kokkos::Execution::GraphContext.
 *
 * @verbatim
 *                               / -- then_atomic | continues_on(gctx) -- \
 * schedule(stc) | fork_join --                                             -- pfor
 *                               \ -- continues_on(gctx) | then_atomic -- /
 * @endverbatim
 *
 * Since the underlying @c when_all does not "remain on graph", this case must create two graphs.
 */
TEST_F(TEST_CATEGORY(ForkJoinTest), mixed_branches_nothing_before_but_schedule_is_default_domain) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t gctx{exec};
    exec::single_thread_context stc{};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, false>;
    using wrapper_t = Kokkos::Impl::UnbindIndexArgument<functor_t>;
    using rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;

    auto sndr = stdexec::schedule(stc.get_scheduler())
              | exec::fork_join(
                    THEN_INCREMENT_ATOMIC(System, data) | stdexec::continues_on(gctx.get_scheduler()),
                    stdexec::continues_on(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data))
              | stdexec::bulk(
                    stdexec::par,
                    1,
                    wrapper_t{
                        functor_t{.prev = 2, .value = 5, .data = data.data()}
    });

    using sndr_t = decltype(sndr);

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::completion_signatures_of_t<sndr_t, stdexec::env_of_t<rcvr_t>>
    >);

    static_assert(stdexec::sender_in<sndr_t, rcvr_t>);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_t, stdexec::env_of_t<rcvr_t>>,
                  stdexec::default_domain
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_RECORD_EVENTS_BUT_THREADS_MAY_THROW_ON_SYNC_WAIT_SKIP(recorder_listener_t, sndr)

    const device_handle_t default_device_handle{};

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0),
                device_handle,
                MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0)),
                dispatch_label(exec, "then")),
            MATCHER_FOR_GRAPH_SUBMIT(default_device_handle.m_exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(
                default_device_handle.m_exec, dispatch_label(default_device_handle.m_exec, "after dispatch"))));

    ASSERT_EQ(data(), 7);
}

} // namespace Tests::GraphImpl
