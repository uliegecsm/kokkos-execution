#include <bit>

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/just_stopped.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c Kokkos::Execution::parallel_for by @c Kokkos::Execution::ExecutionSpaceContext
 * --------------------------------------------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender.
 *
 * The tests can be found in @ref tests/execution_space/test_parallel_for.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class ParallelForTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        BeginParallelForEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
    using variant_t = typename recorder_listener_t::event_variant_t;
};

/**
 * @test Check traits of sender returned by @ref Kokkos::Execution::parallel_for either uncustomized
 *       or customized for @ref Kokkos::Execution::ExecutionSpaceContext.
 */
template <template <typename...> class SndrAdptr, bool IsDispatchingSender, typename... Args>
consteval bool test_sndr_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename ParallelForTest::schedule_sender_t;

    //! Parallel for sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::SumIndices<typename ParallelForTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using pfor_sndr_t = SndrAdptr<Args..., schd_sndr_t, label_t, functor_t, policy_t>;

    //! Models the execution space completing sender concept.
    static_assert(Kokkos::Execution::ExecutionSpaceImpl::execution_space_completing_sender<pfor_sndr_t>);
    static_assert(std::same_as<Kokkos::Execution::Impl::exec_of_t<pfor_sndr_t>, TEST_EXECUTION_SPACE>);

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
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! Has the expected completion scheduler.
    static_assert(std::same_as<
                  Kokkos::Execution::Impl::completion_scheduler_of_t<stdexec::set_value_t, pfor_sndr_t>,
                  Kokkos::Execution::ExecutionSpaceImpl::Scheduler<TEST_EXECUTION_SPACE>
    >);

    //! Is connectable.
    static_assert(stdexec::sender_to<pfor_sndr_t, Tests::Utils::SinkReceiver>);

    static_assert(std::same_as<
                  stdexec::transform_sender_result_t<pfor_sndr_t, stdexec::env_of_t<Tests::Utils::SinkReceiver>>,
                  Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<
                      Kokkos::Execution::parallel_for_t,
                      schd_sndr_t,
                      label_t,
                      functor_t,
                      policy_t
                  >
    >);

    //! It is nothrow connectable.
    static_assert(stdexec::__nothrow_connectable<pfor_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}
static_assert(test_sndr_traits<Kokkos::Execution::Impl::ParallelForSender, true>());
static_assert(test_sndr_traits<
              Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender,
              false,
              Kokkos::Execution::parallel_for_t
>());

//! @test Check decomposition of @ref Kokkos::Execution::Impl::ParallelForSender into the algorithm tag, data, and child sender.
consteval bool test_sndr_decomposition() {
    //! Schedule sender.
    using schd_sndr_t = typename ParallelForTest::schedule_sender_t;

    //! Parallel for sender.
    using label_t = std::string;
    using functor_t = Tests::Utils::Functors::SumIndices<typename ParallelForTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using pfor_sndr_t = Kokkos::Execution::Impl::ParallelForSender<schd_sndr_t, label_t, functor_t, policy_t>;

    //! Is decomposable into the expected algorithm tag, data, and child sender.
    static_assert(stdexec::__sender_for<pfor_sndr_t, Kokkos::Execution::parallel_for_t>);

    static_assert(std::same_as<
                  stdexec::__data_of<pfor_sndr_t>,
                  Kokkos::Execution::Impl::ParallelForData<label_t, functor_t, policy_t>
    >);

    static_assert(stdexec::__nbr_children_of<pfor_sndr_t> == 1);
    static_assert(std::same_as<stdexec::__child_of<pfor_sndr_t>, schd_sndr_t>);

    //! Is transformable via @ref Kokkos::Execution::ExecutionSpaceImpl::TransformSenderFor.
    static_assert(stdexec::__applicable<
                  Kokkos::Execution::ExecutionSpaceImpl::TransformSenderFor<stdexec::tag_of_t<pfor_sndr_t>>,
                  pfor_sndr_t,
                  const stdexec::env<>&
    >);

    return true;
}
static_assert(test_sndr_decomposition());

//! @test Check traits of @ref Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure.
template <typename ViewType>
consteval bool test_closure_traits() {
    using functor_t = Tests::Utils::Functors::SumIndices<ViewType>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using closure_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<std::string, functor_t, policy_t>;

    //! Models the @ref Kokkos::Execution::ExecutionSpaceImpl::Closure concept.
    static_assert(Kokkos::Execution::ExecutionSpaceImpl::Closure<closure_t>);

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

//! @test Check @ref Kokkos::Execution::parallel_for with a team policy.
TEST_F(ParallelForTest, team_policy) {
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

    const view_s_t witness(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler())
               | Kokkos::Execution::parallel_for(
                     "hello from pfor",
                     Kokkos::TeamPolicy<TEST_EXECUTION_SPACE>(num_teams, team_size),
                     Tests::Utils::Functors::SumIndices{.data = witness});

    stdexec::sync_wait(std::move(chain));

    ASSERT_EQ(witness(), size / 2 * (size - 1));
}

template <typename ViewType, Kokkos::ExecutionSpace Exec>
auto closure_object_creation_overloads(
    const size_t size,
    const ViewType& witness,
    const Kokkos::Execution::ExecutionSpaceContext<Exec>& esc) -> stdexec::sender auto {
    auto chain = stdexec::schedule(esc.get_scheduler())
               | Kokkos::Execution::parallel_for(
                     "passing label, execution policy and functor",
                     Kokkos::RangePolicy<Exec>(0, size),
                     Tests::Utils::Functors::SumIndices{.data = witness})
               | Kokkos::Execution::parallel_for(
                     Kokkos::RangePolicy<Exec>(0, size), Tests::Utils::Functors::SumIndices{.data = witness});

    if constexpr (std::same_as<Exec, Kokkos::DefaultExecutionSpace>) {
        return std::move(chain)
             | Kokkos::Execution::parallel_for(
                   "passing label, work count and functor", size, Tests::Utils::Functors::SumIndices{.data = witness})
             | Kokkos::Execution::parallel_for(size, Tests::Utils::Functors::SumIndices{.data = witness});
    } else {
        return chain;
    }
}

//! @test Check @ref Kokkos::Execution::parallel_for closure object creation overloads.
TEST_F(ParallelForTest, closure_object_creation_overloads) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "data - shared space"));

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(
        closure_object_creation_overloads(size, witness, context_t{exec}));

    unsigned short int ievent = 0;

    ASSERT_GE(recorded_events.size(), 3);

    using functor_t = Tests::Utils::Functors::SumIndices<view_s_t>;

    ASSERT_THAT(
        recorded_events,
        ElementAt<variant_t>(ievent++, MATCHER_FOR_BEGIN_PFOR(exec, "passing label, execution policy and functor")));
    ASSERT_THAT(
        recorded_events,
        ElementAt<variant_t>(ievent++, MATCHER_FOR_BEGIN_PFOR(exec, Kokkos::Impl::TypeInfo<functor_t>::name())));

    if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::DefaultExecutionSpace>) {
        ASSERT_THAT(
            recorded_events.at(ievent++), MATCHER_FOR_BEGIN_PFOR(exec, "passing label, work count and functor"));
        ASSERT_THAT(
            recorded_events.at(ievent++), MATCHER_FOR_BEGIN_PFOR(exec, Kokkos::Impl::TypeInfo<functor_t>::name()));
    }

    ASSERT_THAT(recorded_events.at(ievent), MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));

    ASSERT_EQ(witness(), ievent * size / 2 * (size - 1));
}

//! @test Check @ref Kokkos::Execution::parallel_for with two consecutive parallel regions and check there is no fence in between.
TEST_F(ParallelForTest, two_parallel_regions) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = stdexec::schedule(esc.get_scheduler())
               | Kokkos::Execution::parallel_for(
                     std::format("{}: hello from pfor", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                     Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
                     Tests::Utils::Functors::SumIndices{.data = witness})
               | stdexec::then(
                     Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{
                         .prev = size / 2 * (size - 1), .value = 4, .data = witness.data()})
               | Kokkos::Execution::parallel_for(
                     std::format("{}: hello again from pfor", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                     Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, 2 * size),
                     Tests::Utils::Functors::SumIndices{.data = witness});

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(chain)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "hello from pfor")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "hello again from pfor")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(witness(), size / 2 * (size - 1) + 4 + 2 * size * (2 * size - 1) / 2);
}

/**
 * @test Check that @ref Kokkos::Execution::parallel_for with a @c stdexec::starts_on works.
 *
 * @todo Too many synchronizations.
 */
TEST_F(ParallelForTest, starts_on_parallel_region) {
    constexpr size_t size = 10;

    const view_s_t witness(Kokkos::view_alloc(exec, "data - shared space"));

    auto sndr = stdexec::just()
              | Kokkos::Execution::parallel_for(
                    std::format("{}: hello from pfor", Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name()),
                    Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size),
                    Tests::Utils::Functors::SumIndices{.data = witness});

    const context_t esc{exec};
    auto starts_on = stdexec::starts_on(esc.get_scheduler(), std::move(sndr));

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
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "hello from pfor")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "hello from pfor")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        }
    }());

    ASSERT_EQ(witness(), size / 2 * (size - 1));
}

} // namespace Tests::ExecutionSpaceImpl
