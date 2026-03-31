#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/functors/increment.hpp"
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
 * Customization of @c stdexec::then by @c Kokkos::Execution::GraphContext
 * -----------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::GraphContext properly customizes
 * @c stdexec::then.
 *
 * The tests can be found in @ref tests/graph/test_then.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class ThenTest
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

//! @test Check traits of sender returned by @c stdexec::then when customized for @ref Kokkos::Execution::GraphImpl::Domain.
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename ThenTest::schedule_sender_t;

    //! Then sender.
    using functor_t = Tests::Utils::Functors::Increment<ThenTest::view_s_t, true>;
    using then_sndr_t = stdexec::transform_sender_result_t<
        decltype(stdexec::then(std::declval<schd_sndr_t>(), std::declval<functor_t>())),
        stdexec::env<>
    >;

    //! Models the graph completing sender concept.
    static_assert(Kokkos::Execution::GraphImpl::graph_completing_sender<then_sndr_t>);
    static_assert(std::same_as<Kokkos::Execution::Impl::exec_of_t<then_sndr_t>, TEST_EXECUTION_SPACE>);

    //! Does not model the dispatching sender concept.
    static_assert(!Kokkos::Execution::Impl::dispatching_sender<then_sndr_t>);

    return true;
}
static_assert(test_sndr_traits());

//! @test Our customization is not selected. No value channel is added, such that it is not sync-waitable.
static_assert(Tests::Utils::check_continues_on_after_just_stopped<
              typename ThenTest::scheduler_t,
              stdexec::then_t,
              Tests::Utils::Functors::NoOp<false, false, false>
>());

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_nothrow_transformable() {
    using sndr_then_t =
        decltype(stdexec::schedule(std::declval<typename ThenTest::scheduler_t>()) | stdexec::then(Tests::Utils::Functors::NoOp<false, false, false>{}));

    static_assert(std::same_as<
                  stdexec::__demangle_t<sndr_then_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::then_t,
                      Tests::Utils::Functors::NoOp<false, false, false>,
                      typename ThenTest::schedule_sender_t
                  >
    >);

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  sndr_then_t&&,
                  stdexec::env<>
    >);

    using sndr_then_maythrow_on_move_t =
        decltype(stdexec::schedule(std::declval<typename ThenTest::scheduler_t>()) | stdexec::then(Tests::Utils::Functors::NoOp<false, false, true>{}));

    static_assert(!stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::GraphImpl::Domain,
                  stdexec::set_value_t,
                  sndr_then_maythrow_on_move_t&&,
                  stdexec::env<>
    >);

    return true;
}
static_assert(test_sndr_nothrow_transformable());

//! @test Check that it cannot be nothrow-connected.
consteval bool test_sndr_nothrow_connectable() {
    //! Upon @c stdexec::connect, the graph will be constructed.
    static_assert(!std::is_nothrow_constructible_v<Kokkos::Experimental::Graph<TEST_EXECUTION_SPACE>>);

    using sndr_then_t =
        decltype(stdexec::schedule(std::declval<typename ThenTest::scheduler_t>()) | stdexec::then(Tests::Utils::Functors::NoOp<false, false, false>{}));

    static_assert(!stdexec::__nothrow_connectable<sndr_then_t, Tests::Utils::SinkReceiver>);

    return true;
}
static_assert(test_sndr_nothrow_connectable());

//! @test Check that the @c stdexec::connect result is @ref Kokkos::Execution::GraphImpl::OpState, and the @c Kokkos node types are hierarchical and fully typed on predecessors.
consteval bool test_then_opstate_traits() {
    using functor_t = Tests::Utils::Functors::NoOp<false, false, false>;

    using sndr_t =
        decltype(stdexec::schedule(std::declval<typename ThenTest::scheduler_t>()) | stdexec::then(functor_t{}) | stdexec::then(functor_t{}));
    using connect_result_t = stdexec::connect_result_t<sndr_t, Tests::Utils::SinkReceiver>;

    static_assert(stdexec::__is_instance_of<connect_result_t, Kokkos::Execution::GraphImpl::OpState>);

    using root_t = typename ThenTest::graph_t::root_t;
    using then_A_t = Kokkos::Experimental::GraphNodeRef<
        TEST_EXECUTION_SPACE,
        Kokkos::Impl::GraphNodeThenImpl<TEST_EXECUTION_SPACE, Kokkos::Experimental::ThenPolicy<>, functor_t>,
        root_t
    >;
    using then_B_t = Kokkos::Experimental::GraphNodeRef<
        TEST_EXECUTION_SPACE,
        Kokkos::Impl::GraphNodeThenImpl<TEST_EXECUTION_SPACE, Kokkos::Experimental::ThenPolicy<>, functor_t>,
        then_A_t
    >;

    static_assert(std::same_as<typename connect_result_t::predecessor_t, root_t>);
    static_assert(std::same_as<typename connect_result_t::node_t, then_B_t>);

    return true;
}
static_assert(test_then_opstate_traits());

/**
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well when used with @c stdexec::then
 *       within a chain started with @c stdexec::schedule.
 */
TEST_F(ThenTest, then_schedule) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>;

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | stdexec::then(functor_t{.prev = 0, .value = 4, .data = data.data()})
              | stdexec::then(functor_t{.prev = 4, .value = 3, .data = data.data()});

    using sndr_t = decltype(sndr);

    static_assert(std::same_as<
                  stdexec::__demangle_t<stdexec::transform_sender_result_t<sndr_t, stdexec::env<>>>,
                  Kokkos::Execution::GraphImpl::ThenSender<
                      TEST_EXECUTION_SPACE,
                      Tests::Utils::basic_sender_t<stdexec::then_t, functor_t, typename ThenTest::schedule_sender_t>,
                      functor_t
                  >
    >);

    //! The sender environment advertises the default domain, and completes on the @ref Kokkos::Execution::GraphImpl::Domain domain.
    static_assert(std::same_as<stdexec::__domain_of_t<stdexec::env_of_t<sndr_t>>, stdexec::default_domain>);
    static_assert(std::same_as<
                  stdexec::__detail::__completing_domain_t<stdexec::set_value_t, sndr_t>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    //! It has a completion scheduler for the value channel.
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, sndr_t>,
                  typename ThenTest::scheduler_t
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 7);
}

/**
 * @test Similar to @ref Tests::GraphImpl::ThenTest_then_schedule_Test, but the sender starts
 *       with a @c stdexec::starts_on.
 */
TEST_F(ThenTest, then_starts_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    //! Create a sender that does not start with a schedule sender.
    auto work = stdexec::just() | THEN_INCREMENT(data);

    /// The sender cannot be queried for a completion scheduler, nor does it have a determinate domain.
    /// It may complete on the value channel or the error channel, since the @c stdexec::then functor is not @c noexcept.
    using work_t = decltype(work);

    static_assert(!Tests::Utils::has_completion_scheduler_for<work_t, stdexec::set_value_t>);
    static_assert(Tests::Utils::has_completion_signatures<
                  work_t,
                  stdexec::__mset<stdexec::set_error_t(std::exception_ptr), stdexec::set_value_t()>
    >);
    static_assert(
        std::same_as<stdexec::__completion_domain_of_t<stdexec::set_value_t, work_t>, stdexec::indeterminate_domain<>>);

    //! Call @c stdexec::starts_on.
    auto sndr = stdexec::starts_on(gctx.get_scheduler(), std::move(work));

    using sndr_t = decltype(sndr);

    //! It has a completion scheduler for the value channel, but it is a dependent sender.
    static_assert(stdexec::dependent_sender<sndr_t>);
    static_assert(Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_value_t>);
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, sndr_t>,
                  typename ThenTest::scheduler_t
    >);
    static_assert(Tests::Utils::has_completion_signatures<
                  sndr_t,
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  stdexec::env<>
    >);

    //! The completion domain will be @ref Kokkos::Execution::GraphImpl::Domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, sndr_t, stdexec::env<>>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

} // namespace Tests::GraphImpl
