#include <bit>

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/check_rcvr_env_queryable_with.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/reduce_indices.hpp"
#include "tests/utils/just_stopped.hpp"
#include "tests/utils/reduction_workaround_9641.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/sync_wait.hpp"
#include "tests/utils/tracking_allocator.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c Kokkos::Execution::parallel_reduce by @c Kokkos::Execution::ExecutionSpaceContext
 * -----------------------------------------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::ExecutionSpaceImpl::ParallelReduceSender.
 *
 * The tests can be found in @ref tests/execution_space/test_parallel_reduce.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class TEST_CATEGORY(ParallelReduceTest)
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginDeepCopyEvent,
        BeginFenceEvent,
        BeginParallelForEvent,
        BeginParallelReduceEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
    using variant_t = typename recorder_listener_t::event_variant_t;
};

/**
 * @test Check traits of sender returned by @ref Kokkos::Execution::parallel_reduce either uncustomized
 *       or customized for @ref Kokkos::Execution::ExecutionSpaceContext.
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
    using pred_sndr_t = SndrAdptr<Args..., schd_sndr_t, label_t, functor_t, policy_t, reducer_t>;

    //! Models the execution space completing sender concept.
    static_assert(Kokkos::Execution::ExecutionSpaceImpl::execution_space_completing_sender<pred_sndr_t>);
    static_assert(std::same_as<Kokkos::Execution::Impl::exec_of_t<pred_sndr_t>, TEST_EXECUTION_SPACE>);

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
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! Has the expected completion scheduler.
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, pred_sndr_t>,
                  Kokkos::Execution::ExecutionSpaceImpl::Scheduler<TEST_EXECUTION_SPACE>
    >);

    //! Is connectable.
    static_assert(stdexec::sender_to<pred_sndr_t, Tests::Utils::SinkReceiver>);

    static_assert(std::same_as<
                  stdexec::transform_sender_result_t<pred_sndr_t, stdexec::env_of_t<Tests::Utils::SinkReceiver>>,
                  Kokkos::Execution::ExecutionSpaceImpl::ParallelReduceSender<
                      Kokkos::Execution::parallel_reduce_t,
                      schd_sndr_t,
                      label_t,
                      functor_t,
                      policy_t,
                      reducer_t
                  >
    >);

    //! It is nothrow connectable.
    static_assert(stdexec::__nothrow_connectable<pred_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}

static_assert(test_sndr_traits<Kokkos::Execution::Impl::ParallelReduceSender, true>());
static_assert(test_sndr_traits<
              Kokkos::Execution::ExecutionSpaceImpl::ParallelReduceSender,
              false,
              Kokkos::Execution::parallel_reduce_t
>());

//! @test Check decomposition of @ref Kokkos::Execution::Impl::ParallelReduceSender into the algorithm tag, data, and child sender.
consteval bool test_sndr_decomposition() {
    //! Schedule sender.
    using schd_sndr_t = typename TEST_CATEGORY(ParallelReduceTest)::schedule_sender_t;

    //! Parallel for sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::ReduceIndices;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using reducer_t = typename TEST_CATEGORY(ParallelReduceTest)::view_s_t;
    using pred_sndr_t =
        Kokkos::Execution::Impl::ParallelReduceSender<schd_sndr_t, label_t, functor_t, policy_t, reducer_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(stdexec::__sender_for<pred_sndr_t, Kokkos::Execution::parallel_reduce_t>);

    static_assert(std::same_as<
                  stdexec::__data_of<pred_sndr_t>,
                  Kokkos::Execution::Impl::ParallelReduceData<label_t, functor_t, policy_t, reducer_t>
    >);

    static_assert(stdexec::__nbr_children_of<pred_sndr_t> == 1);
    static_assert(std::same_as<stdexec::__child_of<pred_sndr_t>, schd_sndr_t>);

    //! Is transformable via @ref Kokkos::Execution::ExecutionSpaceImpl::TransformSenderFor.
    static_assert(stdexec::__applicable<
                  Kokkos::Execution::ExecutionSpaceImpl::TransformSenderFor<stdexec::tag_of_t<pred_sndr_t>>,
                  pred_sndr_t,
                  const stdexec::env<>&
    >);

    return true;
}

static_assert(test_sndr_decomposition());

//! @test Check traits of @ref Kokkos::Execution::ExecutionSpaceImpl::ParallelReduceClosure.
template <typename ViewType>
consteval bool test_closure_traits() {
    using functor_t = Tests::Utils::Functors::ReduceIndices;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using reducer_t = ViewType;
    using closure_t =
        Kokkos::Execution::ExecutionSpaceImpl::ParallelReduceClosure<std::string, functor_t, policy_t, reducer_t>;

    //! Models the @ref Kokkos::Execution::ExecutionSpaceImpl::Closure concept.
    static_assert(Kokkos::Execution::ExecutionSpaceImpl::Closure<closure_t>);

    static_assert(std::is_nothrow_move_constructible_v<closure_t>);

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
TEST_F(TEST_CATEGORY(ParallelReduceTest), range_policy_rank_0_view) {
    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    constexpr size_t size = 10;

    stdexec::sync_wait(
        stdexec::schedule(esc.get_scheduler())
        | Kokkos::Execution::parallel_reduce(
            "hello from pred",
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
            Tests::Utils::Functors::ReduceIndices{},
            Tests::Utils::reduction_workaround_9641<TEST_EXECUTION_SPACE>(target)));

    ASSERT_EQ(target(), size / 2 * (size - 1));
}

//! @test Check @ref Kokkos::Execution::parallel_reduce with a team policy.
TEST_F(TEST_CATEGORY(ParallelReduceTest), team_policy) {
    constexpr int size = 32;

    const auto [num_teams, team_size] = [&]() {
//! @c Kokkos hardcodes a maximum team size of 1 on @c HPX. See also https://github.com/kokkos/kokkos/blob/1c6efc105c2366a95fa3d0012b38bbf4c03aecd5/core/src/HPX/Kokkos_HPX.hpp#L742-L747.
#if defined(KOKKOS_ENABLE_HPX)
        if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::Experimental::HPX>) {
            return std::make_tuple(size, 1);
        }
#endif
        const int team_size_ = std::bit_floor(static_cast<unsigned short>(std::min(exec.concurrency(), size / 2)));
        return std::make_tuple(size / team_size_, team_size_);
    }();

    ASSERT_EQ(team_size * num_teams, size);

    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    stdexec::sync_wait(
        stdexec::schedule(esc.get_scheduler())
        | Kokkos::Execution::parallel_reduce(
            Kokkos::TeamPolicy<TEST_EXECUTION_SPACE>(num_teams, team_size),
            Tests::Utils::Functors::ReduceIndicesWithTeam<
                TEST_EXECUTION_SPACE,
                typename TEST_CATEGORY(ParallelReduceTest)::value_t
            >{},
            Tests::Utils::reduction_workaround_9641<TEST_EXECUTION_SPACE>(target)));

    ASSERT_EQ(target(), size / 2 * (size - 1));
}

template <typename ViewType, Kokkos::ExecutionSpace Exec>
auto closure_object_creation_overloads(
    const size_t size,
    const ViewType& target,
    const Kokkos::Execution::ExecutionSpaceContext<Exec>& esc) -> stdexec::sender auto {
    auto chain = stdexec::schedule(esc.get_scheduler())
               | Kokkos::Execution::parallel_reduce(
                     "passing label, execution policy, functor and target",
                     Kokkos::RangePolicy<Exec>(0, size),
                     Tests::Utils::Functors::ReduceIndices{},
                     Tests::Utils::reduction_workaround_9641<Exec>(target))
               | Kokkos::Execution::parallel_reduce(
                     Kokkos::RangePolicy<Exec>(0, size),
                     Tests::Utils::Functors::ReduceIndices{},
                     Tests::Utils::reduction_workaround_9641<Exec>(target));

    if constexpr (std::same_as<Exec, Kokkos::DefaultExecutionSpace>) {
        return std::move(chain)
             | Kokkos::Execution::parallel_reduce(
                   "passing label, work count, functor and target",
                   size,
                   Tests::Utils::Functors::ReduceIndices{},
                   Tests::Utils::reduction_workaround_9641<Exec>(target))
             | Kokkos::Execution::parallel_reduce(
                   size,
                   Tests::Utils::Functors::ReduceIndices{},
                   Tests::Utils::reduction_workaround_9641<Exec>(target));
    } else {
        return chain;
    }
}

//! @test Check @ref Kokkos::Execution::parallel_reduce closure object creation overloads.
TEST_F(TEST_CATEGORY(ParallelReduceTest), closure_object_creation_overloads) {
    constexpr size_t size = 10;

    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        closure_object_creation_overloads(size, target, context_t{exec}));

    unsigned short int ievent = 0;

    ASSERT_GE(recorded_events.size(), 3);

    using functor_t = Tests::Utils::Functors::ReduceIndices;
    using reducer_t = Kokkos::Impl::CombinedFunctorReducer<
        functor_t,
        Kokkos::Impl::FunctorAnalysis<
            Kokkos::Impl::FunctorPatternInterface::REDUCE,
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>,
            functor_t,
            int
        >::Reducer,
        void
    >;

    ASSERT_THAT(
        recorded_events,
        ElementAt<variant_t>(
            ievent++, MATCHER_FOR_BEGIN_PRED(exec, "passing label, execution policy, functor and target")));
    ASSERT_THAT(
        recorded_events,
        ElementAt<variant_t>(ievent++, MATCHER_FOR_BEGIN_PRED(exec, Kokkos::Impl::TypeInfo<reducer_t>::name())));

    if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::DefaultExecutionSpace>) {
        ASSERT_THAT(
            recorded_events.at(ievent++),
            MATCHER_FOR_BEGIN_PRED(exec, "passing label, work count, functor and target"));
        ASSERT_THAT(
            recorded_events.at(ievent++), MATCHER_FOR_BEGIN_PRED(exec, Kokkos::Impl::TypeInfo<reducer_t>::name()));
    }

    ASSERT_THAT(recorded_events.at(ievent), MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(target(), size / 2 * (size - 1));
}

//! @test Check @ref Kokkos::Execution::parallel_reduce with two consecutive parallel regions and check there is no fence in between.
TEST_F(TEST_CATEGORY(ParallelReduceTest), two_parallel_regions) {
    constexpr size_t size = 10;

    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler())
               | Kokkos::Execution::parallel_reduce(
                     std::format("{}: hello from pred", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                     Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
                     Tests::Utils::Functors::ReduceIndices{},
                     Tests::Utils::reduction_workaround_9641<TEST_EXECUTION_SPACE>(target))
               | stdexec::then(
                     Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{
                         .prev = size / 2 * (size - 1), .value = 0, .data = target.data()})
               | Kokkos::Execution::parallel_reduce(
                     std::format("{}: hello again from pred", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                     Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 2 * size),
                     Tests::Utils::Functors::ReduceIndices{},
                     Tests::Utils::reduction_workaround_9641<TEST_EXECUTION_SPACE>(target));

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(chain)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PRED(exec, dispatch_label(exec, "hello from pred")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PRED(exec, dispatch_label(exec, "hello again from pred")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(target(), 2 * size * (2 * size - 1) / 2);
}

//! @test Check that @ref Kokkos::Execution::parallel_reduce with a @c stdexec::starts_on works.
TEST_F(TEST_CATEGORY(ParallelReduceTest), starts_on_parallel_region) {
    constexpr size_t size = 10;

    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    auto chain = stdexec::just()
               | Kokkos::Execution::parallel_reduce(
                     std::format("{}: hello from pred", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                     Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
                     Tests::Utils::Functors::ReduceIndices{},
                     Tests::Utils::reduction_workaround_9641<TEST_EXECUTION_SPACE>(target));

    using chain_t = decltype(chain);

    static_assert(
        stdexec::get_completion_signatures<chain_t>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});
    static_assert(!Tests::Utils::has_completion_scheduler_for<chain_t, stdexec::set_value_t>);
    static_assert(
        std::same_as<stdexec::__completion_domain_of_t<stdexec::set_value_t, chain_t>, stdexec::indeterminate_domain<>>);

    const context_t esc{exec};
    auto starts_on = stdexec::starts_on(esc.get_scheduler(), std::move(chain));

    using starts_on_t = decltype(starts_on);

    static_assert(stdexec::__has_eptr_completion<chain_t>);
    static_assert(!stdexec::dependent_sender<starts_on_t>);
    static_assert(
        stdexec::get_completion_signatures<starts_on_t>()
        == stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{});

    /**
     * Note that @c stdexec transforms the @c stdexec::starts_on sender into a sequence sender. This is why the operation
     * state of the customized sender does not let the @c stdexec::sync_wait handle the synchronization.
     *
     * See https://github.com/NVIDIA/stdexec/blob/5473e9daf50cb8829cfe12fb6b64f5f74a08bcf7/include/stdexec/__detail/__starts_on.hpp#L128.
     */
    static_assert(stdexec::__is_instance_of<
                  stdexec::transform_sender_result_t<
                      decltype(starts_on),
                      stdexec::env_of_t<Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE>>
                  >,
                  stdexec::__seq::__sndr
    >);

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(starts_on));

    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PRED(exec, dispatch_label(exec, "hello from pred")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PRED(exec, dispatch_label(exec, "hello from pred")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")));
        }
    }());

    ASSERT_EQ(target(), size / 2 * (size - 1));
}

//! @test The customization of @ref Kokkos::Execution::parallel_reduce properly forwards forwarding queries.
TEST_F(TEST_CATEGORY(ParallelReduceTest), forwarding_env) {
    constexpr size_t size = 10;

    const view_s_t target(Kokkos::view_alloc(exec, "data - shared space"));

    std::atomic<size_t> count = 0;

    int value;

    const context_t esc{exec};

    stdexec::sender auto sndr =
        stdexec::read_env(stdexec::get_allocator)
        | stdexec::then([&value](auto allocator) { value = Tests::Utils::round_trip_allocate(allocator, 42); })
        | stdexec::continues_on(esc.get_scheduler())
        | Tests::Utils::check_rcvr_env_queryable_with<stdexec::get_allocator_t>()
        | Kokkos::Execution::parallel_reduce(
            "my pred",
            Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
            Tests::Utils::Functors::ReduceIndices{},
            Tests::Utils::reduction_workaround_9641<TEST_EXECUTION_SPACE>(target))
        | stdexec::write_env(stdexec::prop{stdexec::get_allocator, Tests::Utils::TrackingAllocator<int>{&count}});

    ASSERT_EQ(target(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PRED(exec, "my pred"),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PRED(exec, "my pred"),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")));
        }
    }());

    ASSERT_EQ(target(), size / 2 * (size - 1));

    ASSERT_EQ(value, 42);
    ASSERT_EQ(count, 1);
}

} // namespace Tests::ExecutionSpaceImpl
