#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
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
 * Customization of @c Kokkos::Execution::parallel_for by @c Kokkos::Execution::GraphContext
 * -----------------------------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::GraphImpl::ParallelForSender.
 *
 * The tests can be found in @ref tests/graph/test_parallel_for.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class ParallelForTest
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
};

/**
 * @test Check traits of sender returned by @ref Kokkos::Execution::parallel_for either uncustomized
 *       or customized for @ref Kokkos::Execution::GraphContext.
 */
template <template <typename...> class SndrAdptr, bool IsDispatchingSender, typename... Args>
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename ParallelForTest::schedule_sender_t;

    //! Parallel for sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::SumIndices<typename ParallelForTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using pfor_sndr_t = SndrAdptr<schd_sndr_t, label_t, functor_t, policy_t>;

    //! Models the graph completing sender concept.
    static_assert(Kokkos::Execution::GraphImpl::graph_completing_sender<pfor_sndr_t>);

    //! Models the dispatching sender concept.
    static_assert(Kokkos::Execution::Impl::dispatching_sender<pfor_sndr_t> == IsDispatchingSender);

    //! Has the expected completion signatures.
    using completion_signatures_t = stdexec::__completion_signatures_of_t<pfor_sndr_t, stdexec::env<>>;

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  completion_signatures_t
    >);

    //! Has the expected completion domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, pfor_sndr_t, stdexec::env<>>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    //! Has the expected completion scheduler.
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, pfor_sndr_t>,
                  Kokkos::Execution::GraphImpl::Scheduler<TEST_EXECUTION_SPACE>
    >);

    //! Is connectable.
    static_assert(stdexec::sender_to<pfor_sndr_t, Tests::Utils::SinkReceiver>);

    static_assert(std::same_as<
                  stdexec::transform_sender_result_t<pfor_sndr_t, stdexec::env_of_t<Tests::Utils::SinkReceiver>>,
                  Kokkos::Execution::GraphImpl::ParallelForSender<schd_sndr_t, label_t, functor_t, policy_t>
    >);

    //! It is not nothrow connectable.
    static_assert(!stdexec::__nothrow_connectable<pfor_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}
static_assert(test_sndr_traits<Kokkos::Execution::Impl::ParallelForSender, true>());
static_assert(test_sndr_traits<Kokkos::Execution::GraphImpl::ParallelForSender, false>());

//! @test Check traits of @ref Kokkos::Execution::GraphImpl::ParallelForClosure.
template <typename ViewType>
consteval bool test_closure_traits() {
    using functor_t = Tests::Utils::Functors::SumIndices<ViewType>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using closure_t = Kokkos::Execution::GraphImpl::ParallelForClosure<std::string, functor_t, policy_t>;

    //! Models the @ref Kokkos::Execution::GraphImpl::Closure concept.
    static_assert(Kokkos::Execution::GraphImpl::Closure<closure_t>);

    static_assert(std::is_nothrow_move_constructible_v<closure_t>);

    return true;
}
static_assert(test_closure_traits<typename ParallelForTest::view_s_t>());
static_assert(test_closure_traits<std::span<int>>());

//! @test Our customization is not selected. No value channel is added, such that it is not sync-waitable.
static_assert(Tests::Utils::check_continues_on_after_just_stopped<
              typename ParallelForTest::scheduler_t,
              Kokkos::Execution::parallel_for_t,
              Kokkos::RangePolicy<TEST_EXECUTION_SPACE>,
              Tests::Utils::Functors::NoOp<false, false, false>
>());

/**
 * @test Check that @ref Kokkos::Execution::GraphContext does its duty well when used with @ref Kokkos::Execution::parallel_for_t
 *       within a chain started with @c stdexec::schedule.
 */
TEST_F(ParallelForTest, parallel_for_schedule) {
    constexpr size_t size = 10;

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    auto sndr = stdexec::schedule(gctx.get_scheduler())
              | Kokkos::Execution::parallel_for(
                    std::format("{}: hello from pfor", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                    Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
                    Tests::Utils::Functors::SumIndices{.data = data});

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), size / 2 * (size - 1));
}

/**
 * @test Similar to @ref Tests::GraphImpl::ParallelForTest_parallel_for_schedule_Test, but the sender starts
 *       with a @c stdexec::starts_on.
 */
TEST_F(ParallelForTest, parallel_for_starts_on) {
    constexpr size_t size = 10;

    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    auto sndr = stdexec::just()
              | Kokkos::Execution::parallel_for(
                    std::format("{}: hello from pfor", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                    Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
                    Tests::Utils::Functors::SumIndices{.data = data});

    const context_t gctx{exec};
    auto starts_on = stdexec::starts_on(gctx.get_scheduler(), std::move(sndr));

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(starts_on));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), size / 2 * (size - 1));
}

/**
 * @test Similar to @ref Tests::GraphImpl::ParallelForTest_parallel_for_schedule_Test,
 *       but using a tagged operator.
 */
TEST_F(ParallelForTest, parallel_for_schedule_tagged_operator) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    using functor_t = Tests::Utils::Functors::TagDispatch<view_s_t>;

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        stdexec::schedule(gctx.get_scheduler())
        | Kokkos::Execution::parallel_for(
            Kokkos::RangePolicy<typename functor_t::Tag, TEST_EXECUTION_SPACE>(0, 1), functor_t{.data = data}));

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
