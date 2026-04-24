#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/sink_receiver.hpp"

/**
 * @addtogroup unittests
 *
 * Completion signal @c Kokkos::Execution::Impl::CompletionSignal
 * --------------------------------------------------------------
 *
 * This group of tests check @ref Kokkos::Execution::Impl::CompletionSignal
 * and its related types.
 *
 * The tests can be found in @ref tests/impl/test_completion_signal.cpp.
 */

namespace Tests::Impl {

using namespace Kokkos::utils::callbacks;

class CompletionSignalTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using parallel_for_data_t = Kokkos::Execution::Impl::ParallelForData<
        std::string,
        Tests::Utils::Functors::SumIndices<view_s_t>,
        Kokkos::RangePolicy<TEST_EXECUTION_SPACE>
    >;

    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;

   public:
    CompletionSignalTest()
        : data(Kokkos::view_alloc(exec, "data - shared space"))
        , esc{exec}
        , pfor_data{
              "hello from pfor",
              Tests::Utils::Functors::SumIndices{.data = data},
              Kokkos::RangePolicy<TEST_EXECUTION_SPACE>(0, size)} {
    }

   protected:
    static constexpr size_t size = 10;
    view_s_t data;
    context_t esc;
    parallel_for_data_t pfor_data;
};

//! @test Check the size of @ref Kokkos::Execution::Impl::CompletionSignal.
consteval bool test_completion_signal_traits() {
    using completion_signal_inline_fence_exec_t = Kokkos::Execution::Impl::CompletionSignal<
        Kokkos::Execution::Impl::SyncPolicy::InlineFenceExec,
        TEST_EXECUTION_SPACE,
        Tests::Utils::SinkReceiver
    >;

    static_assert(sizeof(completion_signal_inline_fence_exec_t) == sizeof(Tests::Utils::SinkReceiver));

    using completion_signal_schedule_wait_event_t = Kokkos::Execution::Impl::CompletionSignal<
        Kokkos::Execution::Impl::SyncPolicy::ScheduleWaitEvent,
        TEST_EXECUTION_SPACE,
        Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE>
    >;

    static_assert(
        sizeof(completion_signal_schedule_wait_event_t)
        > sizeof(Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE>)
              + sizeof(Kokkos::Execution::Impl::event_storage_t<TEST_EXECUTION_SPACE>));

    using completion_signal_passthrough_event_t = Kokkos::Execution::Impl::CompletionSignal<
        Kokkos::Execution::Impl::SyncPolicy::PassThrough,
        TEST_EXECUTION_SPACE,
        Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE>
    >;

    static_assert(
        sizeof(completion_signal_passthrough_event_t)
        == sizeof(Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE>));

    using completion_signal_defer_wait_event_t = Kokkos::Execution::Impl::CompletionSignal<
        Kokkos::Execution::Impl::SyncPolicy::DeferWaitEvent,
        TEST_EXECUTION_SPACE,
        Tests::Utils::SinkReceiver
    >;

    //! The size is larger than the sum of the sizes of the members because of padding.
    static_assert(
        sizeof(completion_signal_defer_wait_event_t)
        > sizeof(Tests::Utils::SinkReceiver) + sizeof(Kokkos::Execution::Impl::event_storage_t<TEST_EXECUTION_SPACE>));

    return true;
}
static_assert(test_completion_signal_traits());

//! @test Check @ref Kokkos::Execution::Impl::CompletionSignal with @ref Kokkos::Execution::Impl::SyncPolicy::InlineFenceExec.
TEST_F(CompletionSignalTest, inline_fence_exec_policy) {
    auto op_state = stdexec::connect(
        stdexec::schedule(esc.get_scheduler())
            | Kokkos::Execution::parallel_for(pfor_data.label, pfor_data.policy, pfor_data.functor),
        Tests::Utils::SinkReceiver{});

    static_assert(
        std::same_as<typename decltype(op_state)::sync_policy_t, Kokkos::Execution::Impl::SyncPolicy::InlineFenceExec>);

    ASSERT_THAT(
        recorder_listener_t::record([&]() { op_state.start(); }),
        testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch"))));

    ASSERT_EQ(data(), size / 2 * (size - 1));
}

template <Kokkos::ExecutionSpace Exec, bool IsDeferredCompletionReceiver>
struct Receiver {
    using receiver_concept = std::conditional_t<
        IsDeferredCompletionReceiver,
        Kokkos::Execution::Impl::DeferredCompletionReceiverTag,
        stdexec::receiver_tag
    >;

    Kokkos::Execution::Impl::State<Exec> const * state;
    stdexec::run_loop* loop;

    void set_value() && noexcept {
        loop->finish();
    }

    template <typename Error>
    void set_error(Error&&) && noexcept {
        loop->finish();
    }

    void continues_after() && noexcept requires(IsDeferredCompletionReceiver)
    {
        loop->finish();
    }

    void continues_after(const Kokkos::Execution::Impl::Event<Exec>& event) && noexcept
        requires(IsDeferredCompletionReceiver)
    {
        Kokkos::Execution::Impl::wait(event);
        loop->finish();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept {
        return Kokkos::Execution::ExecutionSpaceImpl::join_env_with_exec(
            stdexec::prop{stdexec::get_delegation_scheduler, loop->get_scheduler()}, state->exec);
    }
};

//! @test Check @ref Kokkos::Execution::Impl::CompletionSignal with @ref Kokkos::Execution::Impl::SyncPolicy::ScheduleWaitEvent.
TEST_F(CompletionSignalTest, schedule_wait_event_policy) {
    if constexpr (!Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
        GTEST_SKIP() << "The execution space does not have non-blocking dispatch.";
    }

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);
    const context_t esc_A{exec_A}, esc_B{exec_B};

    stdexec::run_loop loop;

    auto op_state = stdexec::connect(
        stdexec::schedule(esc_A.get_scheduler())
            | Kokkos::Execution::parallel_for(pfor_data.label, pfor_data.policy, pfor_data.functor),
        Receiver<TEST_EXECUTION_SPACE, false>{&esc_B.m_state, &loop});

    const auto recorded_events_before_run = recorder_listener_t::record([&]() { op_state.start(); });

    ASSERT_THAT(recorded_events_before_run, testing::ElementsAre(MATCHER_FOR_RECORD_EVENT(exec_A)));

    const auto recorded_events_after_run = recorder_listener_t::record([&]() { loop.run(); });

    ASSERT_THAT(
        recorded_events_after_run, testing::ElementsAre(MATCHER_FOR_WAIT_EVENT(recorded_events_before_run.at(0))));

    ASSERT_EQ(data(), size / 2 * (size - 1));
}

//! @test Check @ref Kokkos::Execution::Impl::CompletionSignal with @ref Kokkos::Execution::Impl::SyncPolicy::PassThrough.
TEST_F(CompletionSignalTest, passthrough_policy) {
    Kokkos::Execution::Impl::SyncWait::State<std::true_type> runloop_state;
    std::optional<std::tuple<>> result;

    auto op_state = stdexec::connect(
        stdexec::schedule(esc.get_scheduler())
            | Kokkos::Execution::parallel_for(pfor_data.label, pfor_data.policy, pfor_data.functor),
        Kokkos::Execution::Impl::SyncWait::Receiver<TEST_EXECUTION_SPACE, std::true_type>{
            .state = std::addressof(esc.m_state),
            .runloop_state = std::addressof(runloop_state),
            .result = std::addressof(result)});

    static_assert(
        std::same_as<typename decltype(op_state)::sync_policy_t, Kokkos::Execution::Impl::SyncPolicy::PassThrough>);

    ASSERT_THAT(
        recorder_listener_t::record([&]() { op_state.start(); }),
        testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), size / 2 * (size - 1));

    runloop_state.loop.run();
}

//! @test Check @ref Kokkos::Execution::Impl::CompletionSignal with @ref Kokkos::Execution::Impl::SyncPolicy::DeferWaitEvent.
TEST_F(CompletionSignalTest, defer_wait_event_policy) {
    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);
    const context_t esc_A{exec_A}, esc_B{exec_B};

    stdexec::run_loop loop;

    auto op_state = stdexec::connect(
        stdexec::schedule(esc_A.get_scheduler())
            | Kokkos::Execution::parallel_for(pfor_data.label, pfor_data.policy, pfor_data.functor),
        Receiver<TEST_EXECUTION_SPACE, true>{&esc_B.m_state, &loop});

    static_assert(Kokkos::Execution::Impl::deferred_completion_receiver<
                  Receiver<TEST_EXECUTION_SPACE, true>,
                  TEST_EXECUTION_SPACE
    >);
    static_assert(
        std::same_as<typename decltype(op_state)::sync_policy_t, Kokkos::Execution::Impl::SyncPolicy::DeferWaitEvent>);

    const auto recorded_events = recorder_listener_t::record([&]() { op_state.start(); });

    if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(recorded_events, testing::IsEmpty());
    } else {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(MATCHER_FOR_RECORD_EVENT(exec_A), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0))));
    }

    ASSERT_EQ(data(), size / 2 * (size - 1));

    loop.run();
}

} // namespace Tests::Impl
