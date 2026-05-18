#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_rcvr_env_queryable_with.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/no_op.hpp"
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

class WhenAllTest
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
    using variant_t = typename recorder_listener_t::event_variant_t;

    WhenAllTest()
        : default_device_handle(TEST_EXECUTION_SPACE{}) {
    }
   protected:
    device_handle_t default_device_handle;
};

//! @test Check traits of sender returned by @c stdexec::when_all when customized for @ref Kokkos::Execution::GraphImpl::Domain.
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename WhenAllTest::schedule_sender_t;

    //! @c stdexec::then sender.
    using functor_t = Tests::Utils::Functors::Increment<WhenAllTest::view_s_t, true>;
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
        stdexec::schedule(std::declval<typename WhenAllTest::scheduler_t>())
        | stdexec::then(Tests::Utils::Functors::NoOp<false, false, false>{})));

    static_assert(std::same_as<
                  stdexec::__demangle_t<when_all_sndr_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::when_all_t,
                      stdexec::__,
                      Tests::Utils::basic_sender_t<
                          stdexec::then_t,
                          Tests::Utils::Functors::NoOp<false, false, false>,
                          typename WhenAllTest::schedule_sender_t
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
        stdexec::schedule(std::declval<typename WhenAllTest::scheduler_t>())
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
        stdexec::schedule(std::declval<typename WhenAllTest::scheduler_t>())
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
TEST_F(WhenAllTest, one_branch) {
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
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well
 *       when used with a two-branches @c stdexec::when_all.
 */
TEST_F(WhenAllTest, two_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto branch_a = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(data);
    auto branch_b = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(data);

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
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 2);
}

/**
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well
 *       when used with a three-branches @c stdexec::when_all.
 */
TEST_F(WhenAllTest, three_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto branch_a = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(data);
    auto branch_b = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(data);
    auto branch_c = stdexec::schedule(gctx.get_scheduler()) | THEN_INCREMENT_ATOMIC(data);

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
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 3);
}

//! @test The customization of @c stdexec::when_all properly forwards forwarding queries.
TEST_F(WhenAllTest, forwarding_env) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    std::atomic<size_t> count = 0;

    const context_t gctx{exec};

    //! @todo Use @ref Tests::Utils::round_trip_allocate as in @ref Tests::ExecutionSpaceImpl::ThenTest_forwarding_env_Test.
    stdexec::sender auto sndr =
        stdexec::when_all(
            stdexec::schedule(gctx.get_scheduler())
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
            MATCHER_FOR_GRAPH_SUBMIT(TEST_EXECUTION_SPACE{}, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(TEST_EXECUTION_SPACE{}, dispatch_label(TEST_EXECUTION_SPACE{}, "after dispatch"))));

    ASSERT_EQ(data(), 1);
}

} // namespace Tests::GraphImpl
