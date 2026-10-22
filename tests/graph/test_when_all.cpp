#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/check_rcvr_env_queryable_with.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"
#include "tests/utils/tracking_allocator.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c stdexec::when_all by @c Kokkos::Execution::GraphContext
 * ---------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext properly customizes
 * @c stdexec::when_all.
 *
 * The tests can be found in @ref tests/graph/test_when_all.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(WhenAllTest)
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
        Kokkos::Execution::GraphImpl::GraphAddAggregateNodeEvent,
        Kokkos::Execution::GraphImpl::GraphAddNodeEvent,
        Kokkos::Execution::GraphImpl::GraphCreateEvent,
        Kokkos::Execution::GraphImpl::GraphInstantiateEvent,
        Kokkos::Execution::GraphImpl::GraphSubmitEvent
    >;

    TEST_CATEGORY(WhenAllTest)()
        : default_device_handle(TEST_EXECUTION_SPACE{}) {
    }
   protected:
    device_handle_t default_device_handle;
};

//! @test Check traits of sender returned by @c stdexec::when_all when customized for @ref Kokkos::Execution::GraphImpl::Domain.
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename TEST_CATEGORY(WhenAllTest)::schedule_sender_t;

    //! @c stdexec::then sender.
    using functor_t = Tests::Utils::Functors::Increment<TEST_CATEGORY(WhenAllTest)::view_s_t, true>;
    using then_sndr_t = decltype(stdexec::then(std::declval<schd_sndr_t>(), std::declval<functor_t>()));

    //! @c stdexec::when_all sender.
    using when_all_sndr_t = stdexec::transform_sender_result_t<
        decltype(stdexec::when_all(std::declval<then_sndr_t>(), std::declval<then_sndr_t>())),
        stdexec::env<>
    >;

    //! Does not model the graph completing sender concept.
    static_assert(!Kokkos::Execution::GraphImpl::graph_completing_sender<when_all_sndr_t>);

    //! Does not model the dispatching sender concept.
    static_assert(!Kokkos::Execution::Impl::dispatching_sender<when_all_sndr_t>);

    return true;
}

static_assert(test_sndr_traits());

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_nothrow_transformable() {
    using when_all_sndr_t = decltype(stdexec::when_all(
        stdexec::schedule(std::declval<typename TEST_CATEGORY(WhenAllTest)::scheduler_t>()),
        stdexec::schedule(std::declval<typename TEST_CATEGORY(WhenAllTest)::scheduler_t>())
            | stdexec::then(Tests::Utils::Functors::NoOp<false, false, false>{})));

    static_assert(std::same_as<
                  when_all_sndr_t,
                  stdexec::__sexpr<
                      stdexec::when_all_t,
                      stdexec::__,
                      typename TEST_CATEGORY(WhenAllTest)::schedule_sender_t,
                      stdexec::then_t::__sender<
                          Tests::Utils::Functors::NoOp<false, false, false>,
                          typename TEST_CATEGORY(WhenAllTest)::schedule_sender_t
                      >
                  >
    >);

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  when_all_sndr_t&&,
                  stdexec::env<>
    >);

    using when_all_maythrow_on_move_sndr_t = decltype(stdexec::when_all(
        stdexec::schedule(std::declval<typename TEST_CATEGORY(WhenAllTest)::scheduler_t>()),
        stdexec::schedule(std::declval<typename TEST_CATEGORY(WhenAllTest)::scheduler_t>())
            | stdexec::then(Tests::Utils::Functors::NoOp<false, false, true>{})));

    static_assert(!stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  when_all_maythrow_on_move_sndr_t&&,
                  stdexec::env<>
    >);

    return true;
}

static_assert(test_sndr_nothrow_transformable());

//! @test Check that it cannot be nothrow-connected.
consteval bool test_sndr_nothrow_connectable() {
    //! Upon @c stdexec::connect, the graph will be constructed.
    static_assert(!std::is_nothrow_constructible_v<Kokkos::Experimental::Graph<TEST_EXECUTION_SPACE>>);

    using when_all_sndr_t = decltype(stdexec::when_all(
        stdexec::schedule(std::declval<typename TEST_CATEGORY(WhenAllTest)::scheduler_t>()),
        stdexec::schedule(std::declval<typename TEST_CATEGORY(WhenAllTest)::scheduler_t>())
            | stdexec::then(Tests::Utils::Functors::NoOp<false, false, false>{})));

    static_assert(!stdexec::__nothrow_connectable<when_all_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}

static_assert(test_sndr_nothrow_connectable());

/**
 * @test Check that the customization properly rejects cases where the
 *       execution space types of the branches are not all the same.
 */
template <typename ExecA, typename ExecB>
consteval bool test_sndr_cannot_mix_execution_space_type() {
    if constexpr (std::same_as<ExecA, ExecB>)
        return true;
    else {
        using sndr_t = stdexec::transform_sender_result_t<
            decltype(stdexec::when_all(
                stdexec::schedule(Kokkos::Execution::GraphContext{std::declval<ExecA>()}.get_scheduler()),
                stdexec::schedule(Kokkos::Execution::GraphContext{std::declval<ExecB>()}.get_scheduler()))),
            Tests::Utils::SinkReceiver
        >;

        static_assert(
            std::same_as<
                sndr_t,
                stdexec::__not_a_sender<
                    stdexec::_WHAT_(Kokkos::Execution::GraphImpl::CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_GRAPH_SCHEDULER),
                    stdexec::_WHY_(Kokkos::Execution::GraphImpl::BECAUSE_THE_EXECUTION_SPACE_TYPE_IS_NOT_HOMOGENEOUS),
                    stdexec::_WHERE_(stdexec::_IN_ALGORITHM_, stdexec::when_all_t),
                    Kokkos::Execution::GraphImpl::WITH_SENDER_AT_INDEX<
                        1,
                        typename Kokkos::Execution::GraphImpl::Scheduler<ExecB>::Sender
                    >,
                    stdexec::_WITH_SENDERS_<
                        typename Kokkos::Execution::GraphImpl::Scheduler<ExecA>::Sender,
                        typename Kokkos::Execution::GraphImpl::Scheduler<ExecB>::Sender
                    >,
                    stdexec::_WITH_ENVIRONMENT_(Tests::Utils::SinkReceiver)
                >
            >);

        return true;
    }
}

static_assert(test_sndr_cannot_mix_execution_space_type<TEST_EXECUTION_SPACE, Kokkos::DefaultHostExecutionSpace>());

/**
 * @test A @c stdexec::when_all with two branches that contain only schedule senders
 *       for @ref Kokkos::Execution::GraphContext.
 *
 * Nothing happens.
 */
TEST_F(TEST_CATEGORY(WhenAllTest), two_branches_with_schedule_senders_only) {
    const context_t gctx{exec};

    auto sndr = stdexec::when_all(stdexec::schedule(gctx.get_scheduler()), stdexec::schedule(gctx.get_scheduler()));

    static_assert(std::same_as<stdexec::tag_of_t<decltype(sndr)>, stdexec::when_all_t>);
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, decltype(sndr)>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(recorded_events, testing::IsEmpty());
}

/**
 * @test Check that @c stdexec::when_all allows a branch to consist of only the schedule sender @ref Kokkos::Execution::GraphImpl::Scheduler::Sender.
 *
 * @verbatim
 *                   schedule(gctx) -------------- \
 *                                                  when_all
 * schedule(stc) --> continues_on(gctx) -- then -- /
 * @endverbatim
 *
 * @todo Make sure that a bare @c continues_on(gctx) works as well.
 */
TEST_F(TEST_CATEGORY(WhenAllTest), schedule_sender) {
    experimental::execution::single_thread_context stc{};
    const context_t gctx{exec};

    auto sndr = stdexec::when_all(
        stdexec::schedule(gctx.get_scheduler()),
        stdexec::schedule(stc.get_scheduler()) | stdexec::continues_on(gctx.get_scheduler()) | stdexec::then([]() { }));

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));
}

/**
 * @test A @c stdexec::when_all with a single branch on @ref Kokkos::Execution::GraphContext.
 *
 * @verbatim
 * schedule(gctx) | then -- when_all
 * @endverbatim
 *
 * @note After the implementation of @cite P4269R1 in https://github.com/NVIDIA/stdexec/pull/2124,
 *       @c when_all(sndr) with a single sender is expression-equivalent to @c auto(sndr). Hence, the sender
 *       returned by @c when_all(sndr) may have a completion scheduler. Notably, for a graph completing
 *       sender @c sndr, @c when_all(sndr) returns a graph completing sender.
 */
TEST_F(TEST_CATEGORY(WhenAllTest), single_branch) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr = stdexec::when_all(stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT(data));

    static_assert(std::same_as<
                  decltype(stdexec::get_completion_domain<stdexec::set_value_t>(stdexec::get_env(sndr))),
                  Kokkos::Execution::GraphImpl::Domain
    >);

    static_assert(std::same_as<stdexec::tag_of_t<decltype(sndr)>, stdexec::then_t>);

    //! It has a completion scheduler.
    static_assert(Kokkos::Execution::GraphImpl::graph_completing_sender<decltype(sndr)>);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

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
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well
 *       when used with a @c stdexec::when_all with a schedule sender in one branch and a single other branch.
 *
 * @verbatim
 * schedule(gctx) --------- \
 *                           when_all
 * schedule(gctx) | then -- /
 * @endverbatim
 */
TEST_F(TEST_CATEGORY(WhenAllTest), schedule_sender_and_single_branch) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr = stdexec::when_all(
        stdexec::schedule(gctx.get_scheduler()), stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT(data));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well
 *       when used with a two-branches @c stdexec::when_all.
 *
 * @verbatim
 * schedule(gctx) | then_atomic -- \
 *                                  when_all
 * schedule(gctx) | then_atomic -- /
 * @endverbatim
 */
TEST_F(TEST_CATEGORY(WhenAllTest), two_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto branch_a = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data);
    auto branch_b = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data);

    auto sndr = stdexec::when_all(std::move(branch_a), std::move(branch_b));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(2))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 2);
}

/**
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well
 *       when used with a three-branches @c stdexec::when_all.
 *
 * @verbatim
 * schedule(gctx) | then_atomic -- \
 *                                  \
 * schedule(gctx) | then_atomic --   when_all
 *                                  /
 * schedule(gctx) | then_atomic -- /
 * @endverbatim
 */
TEST_F(TEST_CATEGORY(WhenAllTest), three_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto branch_a = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data);
    auto branch_b = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data);
    auto branch_c = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data);

    auto sndr = stdexec::when_all(std::move(branch_a), std::move(branch_b), std::move(branch_c));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

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
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test Similar to @ref Tests::GraphImpl::WhenAllTest_three_branches_Test, but each branch starts with
 *       some work on @c experimental::execution::single_thread_context.
 *
 * This test ensures that if the @c stdexec::when_all creates a single graph with the termination
 * of each branch, the graph is submitted only when all branches have complete their work on their respective
 * @c experimental::execution::single_thread_context.
 *
 * @verbatim
 * schedule(stc_a) | then --> continues_on(gctx) | then -- \
 *                                                          \
 * schedule(stc_b) | then --> continues_on(gctx) | then --   when_all
 *                                                          /
 * schedule(stc_c) | then --> continues_on(gctx) | then -- /
 * @endverbatim
 */
TEST_F(TEST_CATEGORY(WhenAllTest), three_branches_starting_on_single_thread_context) {
    const Kokkos::View<value_t[3], Kokkos::SharedSpace> data_per_branch(Kokkos::view_alloc("data - shared space"));

    experimental::execution::single_thread_context stc_a{}, stc_b{}, stc_c{};

    const context_t gctx{exec};

    using functor_h_t = Tests::Utils::Functors::LoadCheckAdd<value_t, false>;
    using functor_d_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto branch_a = stdexec::schedule(stc_a.get_scheduler())
                  | stdexec::then(functor_h_t{.prev = 0, .value = 2, .data = &data_per_branch(0)})
                  | stdexec::continues_on(gctx.get_scheduler())
                  | stdexec::then(functor_d_t{.prev = 2, .value = 3, .data = &data_per_branch(0)});
    auto branch_b = stdexec::schedule(stc_b.get_scheduler())
                  | stdexec::then(functor_h_t{.prev = 0, .value = 3, .data = &data_per_branch(1)})
                  | stdexec::continues_on(gctx.get_scheduler())
                  | stdexec::then(functor_d_t{.prev = 3, .value = 4, .data = &data_per_branch(1)});
    auto branch_c = stdexec::schedule(stc_c.get_scheduler())
                  | stdexec::then(functor_h_t{.prev = 0, .value = 4, .data = &data_per_branch(2)})
                  | stdexec::continues_on(gctx.get_scheduler())
                  | stdexec::then(functor_d_t{.prev = 4, .value = 5, .data = &data_per_branch(2)});

    auto sndr = stdexec::when_all(
        std::move(branch_a), std::move(branch_b), std::move(branch_c)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(Tests::Utils::span_from(data_per_branch), testing::Each(testing::Eq(0)))
        << "Eager execution is not " "allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

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
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_THAT(Tests::Utils::span_from(data_per_branch), testing::ElementsAre(5, 7, 9));
}

/**
 * @test Similar to @ref Tests::GraphImpl::WhenAllTest_three_branches_starting_on_single_thread_context_Test,
 *       but some branches start with some work on @c experimental::execution::single_thread_context,
 *       whereas others directly start on @ref Kokkos::Execution::GraphImpl::Domain.
 *
 * @verbatim
 * schedule(stc_a) | then --> continues_on(gctx) | then -- \
 *                                                          \
 * schedule(stc_b) | then --> continues_on(gctx) | then --   when_all
 *                                                          /
 *                            schedule(gctx)     | then -- /
 * @endverbatim
 */
TEST_F(TEST_CATEGORY(WhenAllTest), three_branches_some_starting_on_single_thread_context) {
    const Kokkos::View<value_t[3], Kokkos::SharedSpace> data_per_branch(Kokkos::view_alloc("data - shared space"));

    experimental::execution::single_thread_context stc_a{}, stc_b{};

    const context_t gctx{exec};

    using functor_h_t = Tests::Utils::Functors::LoadCheckAdd<value_t, false>;
    using functor_d_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto branch_a = stdexec::schedule(stc_a.get_scheduler())
                  | stdexec::then(functor_h_t{.prev = 0, .value = 2, .data = &data_per_branch(0)})
                  | stdexec::continues_on(gctx.get_scheduler())
                  | stdexec::then(functor_d_t{.prev = 2, .value = 3, .data = &data_per_branch(0)});
    auto branch_b = stdexec::schedule(stc_b.get_scheduler())
                  | stdexec::then(functor_h_t{.prev = 0, .value = 3, .data = &data_per_branch(1)})
                  | stdexec::continues_on(gctx.get_scheduler())
                  | stdexec::then(functor_d_t{.prev = 3, .value = 4, .data = &data_per_branch(1)});
    auto branch_c = stdexec::schedule(gctx.get_scheduler())
                  | stdexec::then(functor_d_t{.prev = 0, .value = 4, .data = &data_per_branch(2)});

    auto sndr = stdexec::when_all(
        std::move(branch_a), std::move(branch_b), std::move(branch_c)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(Tests::Utils::span_from(data_per_branch), testing::Each(testing::Eq(0)))
        << "Eager execution is not " "allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

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
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_THAT(Tests::Utils::span_from(data_per_branch), testing::ElementsAre(5, 7, 4));
}

/**
 * @test The customization of @c stdexec::when_all properly forwards forwarding queries.
 *
 * @verbatim
 *                    schedule(gctx) --------------------------------------------- \
 *                                                                                  when_all | write_env
 * read_env | then -> continues_on(gctx) | check_rcvr_env_queryable_with | then -- /
 * @endverbatim
 */
TEST_F(TEST_CATEGORY(WhenAllTest), forwarding_env) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    std::atomic<size_t> count = 0;

    int value;

    const context_t gctx{exec};

    stdexec::sender auto sndr =
        stdexec::when_all(
            stdexec::schedule(gctx.get_scheduler()),
            stdexec::read_env(stdexec::get_allocator)
                | stdexec::then([&value](auto allocator) { value = Tests::Utils::round_trip_allocate(allocator, 42); })
                | stdexec::continues_on(gctx.get_scheduler())
                | Tests::Utils::check_rcvr_env_queryable_with<stdexec::get_allocator_t>() | THEN_INCREMENT(data))
        | stdexec::write_env(stdexec::prop{stdexec::get_allocator, Tests::Utils::TrackingAllocator<int>{&count}});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0),
                MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0)),
                MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 1);

    ASSERT_EQ(value, 42);
    ASSERT_EQ(count, 1);
}

/**
 * @test Check that @ref Kokkos::Execution::GraphImpl::RemainsOnGraphFor can properly flag the resulting operation state
 *       as "remains on the graph".
 */
consteval bool test_when_all_remains_on_graph_for_two_branches() {
    using schd_sndr_t = typename TEST_CATEGORY(WhenAllTest)::schedule_sender_t;

    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;

    using br_A_t = decltype(std::declval<schd_sndr_t>() | stdexec::then(functor_t{}));
    using br_B_t = decltype(std::declval<schd_sndr_t>() | stdexec::then(functor_t{}));

    //! The resulting @c stdexec::when_all operation is well formed, and remains on the graph.
    using sndr_t = decltype(stdexec::when_all(std::declval<br_A_t>(), std::declval<br_B_t>()));

    using op_state_t = stdexec::connect_result_t<sndr_t, rcvr_t>;

    using res_t = Kokkos::Execution::GraphImpl::RemainsOnGraphFor<op_state_t, TEST_EXECUTION_SPACE>;

    using diagnose_t = Kokkos::Execution::Impl::diagnose_t<res_t>;

    static_assert(std::same_as<diagnose_t, stdexec::__msuccess>);

    static_assert(Kokkos::Execution::Impl::mandate_or_explain<res_t>());

    return true;
}

static_assert(test_when_all_remains_on_graph_for_two_branches());

/**
 * @test Check that @ref Kokkos::Execution::GraphImpl::RemainsOnGraphFor can properly flag the resulting operation state
 *       as "not remains on the graph" with a useful diagnostic.
 *
 * One branch out of the two remains on the graph.
 */
consteval bool test_when_all_does_not_remain_on_graph_for_two_branches() {
    using scheduler_t = TEST_CATEGORY(WhenAllTest)::scheduler_t;
    using schd_sndr_t = typename TEST_CATEGORY(WhenAllTest)::schedule_sender_t;

    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;

    //! This branch will not remain on the graph.
    using br_A_t =
        decltype(stdexec::schedule(std::declval<exec::single_thread_context&>().get_scheduler()) | stdexec::continues_on(std::declval<scheduler_t>()) | stdexec::then(functor_t{}));

    //! This branch will remain on the graph.
    using br_B_t = decltype(std::declval<schd_sndr_t>() | stdexec::then(functor_t{}));

    //! The resulting @c stdexec::when_all operation is well formed, but does not remain on the graph.
    using sndr_t = decltype(stdexec::when_all(std::declval<br_A_t>(), std::declval<br_B_t>()));

    using op_state_t = stdexec::connect_result_t<sndr_t, rcvr_t>;

    using op_state_br_A_t = stdexec::connect_result_t<br_A_t, typename op_state_t::WhenAllChildReceiver>;

    using diagnose_t = Kokkos::Execution::Impl::diagnose_t<
        Kokkos::Execution::GraphImpl::RemainsOnGraphFor<op_state_t, TEST_EXECUTION_SPACE>
    >;

    //! The diagnosis is quite verbose, but reading it points exactly to which parts of the operation states are rejected.
    static_assert(
        std::same_as<
            diagnose_t,
            stdexec::_ERROR_<
                stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                stdexec::_WHY_(Kokkos::Execution::GraphImpl::BECAUSE_AT_LEAST_ONE_CHILD_DOES_NOT_REMAIN_ON_GRAPH),
                Kokkos::Execution::Impl::WITH_CHILDREN_ERRORS<stdexec::_ERROR_<
                    stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                    stdexec::_WHY_(
                        Kokkos::Execution::GraphImpl::BECAUSE_THE_INNER_OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                    stdexec::_ERROR_<
                        stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                        stdexec::_WHY_(
                            Kokkos::Execution::GraphImpl::BECAUSE_THE_INNER_OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                        stdexec::_ERROR_<
                            stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                            Kokkos::Execution::Impl::WITH_OPSTATE<
                                typename op_state_br_A_t::inner_op_state_t::inner_op_state_t
                            >,
                            Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                        >,
                        Kokkos::Execution::Impl::WITH_OPSTATE<typename op_state_br_A_t::inner_op_state_t>,
                        Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                    >,
                    Kokkos::Execution::Impl::WITH_OPSTATE<op_state_br_A_t>,
                    Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                >>,
                Kokkos::Execution::Impl::WITH_OPSTATE<op_state_t>
            >
        >);

    return true;
}

static_assert(test_when_all_does_not_remain_on_graph_for_two_branches());

/**
 * @test Check that @ref Kokkos::Execution::GraphImpl::RemainsOnGraphFor can properly flag the resulting operation state
 *       as "not remains on the graph" with a useful diagnostic.
 *
 * One branch out of the three remains on the graph.
 */
consteval bool test_when_all_does_not_remain_on_graph_for_three_branches() {
    using scheduler_t = TEST_CATEGORY(WhenAllTest)::scheduler_t;
    using schd_sndr_t = typename TEST_CATEGORY(WhenAllTest)::schedule_sender_t;

    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;
    using rcvr_t = Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>;

    //! This branch will not remain on the graph.
    using br_A_t =
        decltype(stdexec::schedule(std::declval<exec::single_thread_context&>().get_scheduler()) | stdexec::continues_on(std::declval<scheduler_t>()) | stdexec::then(functor_t{}));

    //! This branch will remain on the graph.
    using br_B_t = decltype(std::declval<schd_sndr_t>() | stdexec::then(functor_t{}));

    //! This branch will not remain on the graph.
    using br_C_t =
        decltype(stdexec::schedule(std::declval<exec::single_thread_context&>().get_scheduler()) | stdexec::continues_on(std::declval<scheduler_t>()) | stdexec::then(functor_t{}));

    //! The resulting @c stdexec::when_all operation is well formed, but does not remain on the graph.
    using sndr_t = decltype(stdexec::when_all(std::declval<br_A_t>(), std::declval<br_B_t>(), std::declval<br_C_t>()));

    using op_state_t = stdexec::connect_result_t<sndr_t, rcvr_t>;

    using op_state_br_A_t = stdexec::connect_result_t<br_A_t, typename op_state_t::WhenAllChildReceiver>;
    using op_state_br_C_t = stdexec::connect_result_t<br_C_t, typename op_state_t::WhenAllChildReceiver>;

    using diagnose_t = Kokkos::Execution::Impl::diagnose_t<
        Kokkos::Execution::GraphImpl::RemainsOnGraphFor<op_state_t, TEST_EXECUTION_SPACE>
    >;

    //! The diagnosis is quite verbose, but reading it points exactly to which parts of the operation states are rejected.
    static_assert(
        std::same_as<
            diagnose_t,
            stdexec::_ERROR_<
                stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                stdexec::_WHY_(Kokkos::Execution::GraphImpl::BECAUSE_AT_LEAST_ONE_CHILD_DOES_NOT_REMAIN_ON_GRAPH),
                Kokkos::Execution::Impl::WITH_CHILDREN_ERRORS<
                    stdexec::_ERROR_<
                        stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                        stdexec::_WHY_(
                            Kokkos::Execution::GraphImpl::BECAUSE_THE_INNER_OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                        stdexec::_ERROR_<
                            stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                            stdexec::_WHY_(
                                Kokkos::Execution::GraphImpl::BECAUSE_THE_INNER_OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                            stdexec::_ERROR_<
                                stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                                Kokkos::Execution::Impl::WITH_OPSTATE<
                                    typename op_state_br_A_t::inner_op_state_t::inner_op_state_t
                                >,
                                Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                            >,
                            Kokkos::Execution::Impl::WITH_OPSTATE<typename op_state_br_A_t::inner_op_state_t>,
                            Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                        >,
                        Kokkos::Execution::Impl::WITH_OPSTATE<op_state_br_A_t>,
                        Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                    >,
                    stdexec::_ERROR_<
                        stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                        stdexec::_WHY_(
                            Kokkos::Execution::GraphImpl::BECAUSE_THE_INNER_OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                        stdexec::_ERROR_<
                            stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                            stdexec::_WHY_(
                                Kokkos::Execution::GraphImpl::BECAUSE_THE_INNER_OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                            stdexec::_ERROR_<
                                stdexec::_WHAT_(Kokkos::Execution::GraphImpl::OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                                Kokkos::Execution::Impl::WITH_OPSTATE<
                                    typename op_state_br_C_t::inner_op_state_t::inner_op_state_t
                                >,
                                Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                            >,
                            Kokkos::Execution::Impl::WITH_OPSTATE<typename op_state_br_C_t::inner_op_state_t>,
                            Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                        >,
                        Kokkos::Execution::Impl::WITH_OPSTATE<op_state_br_C_t>,
                        Kokkos::Execution::Impl::WITH_EXECUTION_SPACE<TEST_EXECUTION_SPACE>
                    >
                >,
                Kokkos::Execution::Impl::WITH_OPSTATE<op_state_t>
            >
        >);

    return true;
}

static_assert(test_when_all_does_not_remain_on_graph_for_three_branches());

} // namespace Tests::GraphImpl
