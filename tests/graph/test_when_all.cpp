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
        stdexec::schedule(std::declval<typename TEST_CATEGORY(WhenAllTest)::scheduler_t>())
        | stdexec::then(Tests::Utils::Functors::NoOp<false, false, false>{})));

    static_assert(std::same_as<
                  stdexec::__demangle_t<when_all_sndr_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::when_all_t,
                      stdexec::__,
                      Tests::Utils::basic_sender_t<
                          stdexec::then_t,
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
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well
 *       when used with a single-branch @c stdexec::when_all.
 */
TEST_F(TEST_CATEGORY(WhenAllTest), one_branch) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr = stdexec::when_all(stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT(data));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(default_device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0), MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well
 *       when used with a two-branches @c stdexec::when_all.
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
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
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
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
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
 */
TEST_F(TEST_CATEGORY(WhenAllTest), three_branches_starting_on_single_thread_context) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    experimental::execution::single_thread_context stc_a{}, stc_b{}, stc_c{};

    const context_t gctx{exec};

    auto branch_a = stdexec::schedule(stc_a.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                  | stdexec::continues_on(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data);
    auto branch_b = stdexec::schedule(stc_b.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                  | stdexec::continues_on(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data);
    auto branch_c = stdexec::schedule(stc_c.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                  | stdexec::continues_on(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data);

    auto sndr = stdexec::when_all(std::move(branch_a), std::move(branch_b), std::move(branch_c));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

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
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 6);
}

/**
 * @test Similar to @ref Tests::GraphImpl::WhenAllTest_three_branches_starting_on_single_thread_context_Test,
 *       but some branches start with some work on @c experimental::execution::single_thread_context,
 *       whereas others directly start on @ref Kokkos::Execution::GraphImpl::Domain.
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

    auto sndr = stdexec::when_all(std::move(branch_a), std::move(branch_b), std::move(branch_c));

    ASSERT_THAT(Tests::Utils::span_from(data_per_branch), testing::Each(testing::Eq(0)))
        << "Eager execution is not " "allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    KOKKOS_EXECUTION_TEST_UTILS_GRAPH_FENCE(exec);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

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
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_THAT(Tests::Utils::span_from(data_per_branch), testing::ElementsAre(5, 7, 4));
}

//! @test The customization of @c stdexec::when_all properly forwards forwarding queries.
TEST_F(TEST_CATEGORY(WhenAllTest), forwarding_env) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    std::atomic<size_t> count = 0;

    int value;

    const context_t gctx{exec};

    stdexec::sender auto sndr =
        stdexec::when_all(
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
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(
                recorded_events.at(0), MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 1);

    ASSERT_EQ(value, 42);
    ASSERT_EQ(count, 1);
}

} // namespace Tests::GraphImpl
