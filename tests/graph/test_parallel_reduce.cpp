#include "kokkos-utils/callbacks/ConjunctionMatcher.hpp"
#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/maxloc.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/reduce_indices.hpp"
#include "tests/utils/functors/tag_dispatch.hpp"
#include "tests/utils/graph_context.hpp"
#include "tests/utils/just_stopped.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c Kokkos::Execution::parallel_reduce by @c Kokkos::Execution::GraphContext
 * --------------------------------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::GraphImpl::ParallelReduceSender.
 *
 * The tests can be found in @ref tests/graph/test_parallel_reduce.cpp.
 */

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(ParallelReduceTest)
    : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        ConjunctionMatcher<EventDiscardMatcher<TEST_EXECUTION_SPACE>, GraphEventDiscardMatcher<TEST_EXECUTION_SPACE>>,
        BeginFenceEvent,
        BeginParallelForEvent,
        BeginParallelReduceEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent,
        Kokkos::Execution::GraphImpl::GraphAddNodeEvent,
        Kokkos::Execution::GraphImpl::GraphCreateEvent,
        Kokkos::Execution::GraphImpl::GraphInstantiateEvent,
        Kokkos::Execution::GraphImpl::GraphSubmitEvent
    >;
};

/**
 * @test Check traits of sender returned by @ref Kokkos::Execution::parallel_reduce either uncustomized
 *       or customized for @ref Kokkos::Execution::GraphContext.
 */
template <template <typename...> class SndrAdptr, bool IsDispatchingSender, typename... Args>
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename TEST_CATEGORY(ParallelReduceTest)::schedule_sender_t;

    //! Parallel reduce sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::ReduceIndices;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using reducer_t = typename TEST_CATEGORY(ParallelReduceTest)::view_s_t;
    using pred_sndr_t = SndrAdptr<Args..., schd_sndr_t, TEST_EXECUTION_SPACE, label_t, functor_t, policy_t, reducer_t>;

    //! Models the graph completing sender concept.
    static_assert(Kokkos::Execution::GraphImpl::graph_completing_sender<pred_sndr_t>);

    //! Models the dispatching sender concept.
    static_assert(Kokkos::Execution::Impl::dispatching_sender<pred_sndr_t> == IsDispatchingSender);

    //! Has the expected completion signatures.
    static_assert(!stdexec::dependent_sender<pred_sndr_t>);
    static_assert(
        stdexec::get_completion_signatures<pred_sndr_t>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});

    //! Has the expected completion domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, pred_sndr_t, stdexec::env<>>,
                  Kokkos::Execution::GraphImpl::Domain
    >);

    //! Has the expected completion scheduler.
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, pred_sndr_t>,
                  Kokkos::Execution::GraphImpl::Scheduler<TEST_EXECUTION_SPACE>
    >);

    //! Is connectable.
    static_assert(stdexec::sender_to<pred_sndr_t, Tests::Utils::SinkReceiver>);

    static_assert(std::same_as<
                  stdexec::transform_sender_result_t<pred_sndr_t, stdexec::env_of_t<Tests::Utils::SinkReceiver>>,
                  Kokkos::Execution::GraphImpl::ParallelReduceSender<
                      schd_sndr_t,
                      Kokkos::Execution::GraphImpl::node_props_t<TEST_EXECUTION_SPACE, label_t>,
                      functor_t,
                      policy_t,
                      reducer_t
                  >
    >);

    //! It is not nothrow connectable.
    static_assert(!stdexec::__nothrow_connectable<pred_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}

template <typename Sndr, typename Exec, typename Label, typename Functor, typename Policy, typename Reducer>
using generic_t = Kokkos::Execution::Impl::ParallelReduceSender<Sndr, Label, Functor, Policy, Reducer>;

template <typename Sndr, typename Exec, typename Label, typename Functor, typename Policy, typename Reducer>
using impl_t = Kokkos::Execution::GraphImpl::ParallelReduceSender<
    Sndr,
    Kokkos::Execution::GraphImpl::node_props_t<Exec, Label>,
    Functor,
    Policy,
    Reducer
>;

static_assert(test_sndr_traits<generic_t, true>());
static_assert(test_sndr_traits<impl_t, false>());

//! @test Check traits of @ref Kokkos::Execution::GraphImpl::ParallelReduceClosure.
template <typename ViewType>
consteval bool test_closure_traits() {
    using functor_t = Tests::Utils::Functors::ReduceIndices;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using iter_space_t = Kokkos::dims<1, int>;
    using reducer_t = ViewType;
    using closure_policy_t = Kokkos::Execution::GraphImpl::ParallelReduceClosure<
        Kokkos::Execution::GraphImpl::node_props_t<TEST_EXECUTION_SPACE, std::string>,
        functor_t,
        policy_t,
        reducer_t
    >;
    using closure_iter_space_t = Kokkos::Execution::GraphImpl::ParallelReduceClosure<
        Kokkos::Execution::GraphImpl::node_props_t<TEST_EXECUTION_SPACE, std::string>,
        functor_t,
        iter_space_t,
        reducer_t
    >;

    //! Models the @ref Kokkos::Execution::GraphImpl::Closure concept.
    static_assert(Kokkos::Execution::GraphImpl::Closure<closure_policy_t>);
    static_assert(Kokkos::Execution::GraphImpl::Closure<closure_iter_space_t>);

    static_assert(std::is_nothrow_move_constructible_v<closure_policy_t>);
    static_assert(std::is_nothrow_move_constructible_v<closure_iter_space_t>);

    return true;
}

static_assert(test_closure_traits<typename TEST_CATEGORY(ParallelReduceTest)::view_s_t>());
static_assert(test_closure_traits<std::span<int>>());

//! @test Our customization is not selected. No value channel is added, such that it is not sync-waitable.
static_assert(Tests::Utils::check_continues_on_after_just_stopped<
              typename TEST_CATEGORY(ParallelReduceTest)::scheduler_t,
              Kokkos::Execution::parallel_reduce_t,
              Kokkos::RangePolicy<TEST_EXECUTION_SPACE>,
              Tests::Utils::Functors::NoOp<false, false, false>,
              typename TEST_CATEGORY(ParallelReduceTest)::view_s_t
>());

//! @test Check @ref Kokkos::Execution::parallel_reduce with a range policy and a rank-0 @c Kokkos::View as reduction target.
TEST_F(TEST_CATEGORY(ParallelReduceTest), range_policy_into_rank_0_view) {
    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t gctx{exec};

    constexpr size_t size = 10;

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        stdexec::schedule(gctx.get_scheduler())
        | Kokkos::Execution::parallel_reduce(
            "hello from pred",
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
            Tests::Utils::Functors::ReduceIndices{},
            target));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(target(), size / 2 * (size - 1));
}

//! @test Check @ref Kokkos::Execution::parallel_reduce with a range policy and a @c Kokkos::MaxLoc reducer.
TEST_F(TEST_CATEGORY(ParallelReduceTest), range_policy_maxloc_reducer) {
    constexpr size_t size = 5;

    using data_view_t = Kokkos::View<value_t[size], TEST_EXECUTION_SPACE>;
    using max_loc_t = Kokkos::MaxLoc<value_t, typename TEST_EXECUTION_SPACE::index_type, Kokkos::SharedSpace>;
    using max_loc_functor_t = Tests::Utils::Functors::MaxLoc<data_view_t>;

    const data_view_t data(Kokkos::view_alloc(Kokkos::WithoutInitializing, exec));

    const context_t gctx{exec};

    const typename max_loc_t::result_view_type target(Kokkos::view_alloc(exec));

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        stdexec::schedule(gctx.get_scheduler())
        | Kokkos::Execution::parallel_for(
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 1),
            KOKKOS_LAMBDA<std::integral T>(const T) {
                data(0) = 42;
                data(1) = -5;
                data(2) = 6;
                data(3) = 56;
                data(4) = -6;
            })
        | Kokkos::Execution::parallel_reduce(
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size), max_loc_functor_t{.data = data}, max_loc_t{target}));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(target().val, 56);
    ASSERT_EQ(target().loc, 3);
}

//! @test Check that @ref Kokkos::Execution::parallel_reduce with a @c stdexec::starts_on works.
TEST_F(TEST_CATEGORY(ParallelReduceTest), starts_on_parallel_region) {
    constexpr size_t size = 10;

    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    auto chain =
        stdexec::just()
        | Kokkos::Execution::parallel_reduce(
            std::format("{}: hello from pred", Kokkos::Execution::Impl::display_string_of<TEST_EXECUTION_SPACE>()),
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
            Tests::Utils::Functors::ReduceIndices{},
            target);

    using chain_t = decltype(chain);

    static_assert(
        stdexec::get_completion_signatures<chain_t>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});
    static_assert(!Tests::Utils::has_completion_scheduler_for<chain_t, stdexec::set_value_t>);
    static_assert(
        std::same_as<stdexec::__completion_domain_of_t<stdexec::set_value_t, chain_t>, stdexec::indeterminate_domain<>>);

    const context_t gctx{exec};
    auto starts_on = stdexec::starts_on(gctx.get_scheduler(), std::move(chain));

    using starts_on_t = decltype(starts_on);

    static_assert(stdexec::__has_eptr_completion<chain_t>);
    static_assert(!stdexec::dependent_sender<starts_on_t>);
    static_assert(
        stdexec::get_completion_signatures<starts_on_t>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(starts_on));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_ROOT_NODE_OF(recorded_events.at(0))),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0)),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch"))));

    ASSERT_EQ(target(), size / 2 * (size - 1));
}

} // namespace Tests::GraphImpl
