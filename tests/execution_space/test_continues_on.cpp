#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_continues_on.hpp"
#include "tests/utils/check_rcvr_env.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/labeled.hpp"
#include "tests/utils/kokkos.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c continues_on by @c Kokkos::Execution::ExecutionSpaceContext
 * -------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly customizes
 * @c continues_on.
 *
 * The tests can be found in @ref tests/execution_space/test_continues_on.cpp.
 */

using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class ContinuesOnTest
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
};

//! @test Check traits of the sender created by the customized @c stdexec::schedule_from.
consteval bool test_schedule_from_sndr_traits() {
    using schd_t = typename ContinuesOnTest::scheduler_t;
    using schd_sndr_t = typename ContinuesOnTest::schedule_sender_t;

    using schedule_from_sndr_t = Kokkos::Execution::ExecutionSpaceImpl::ScheduleFromSender<
        Kokkos::Execution::ExecutionSpaceImpl::WithDelegatedSyncPolicy,
        schd_t,
        schd_sndr_t
    >;

    static_assert(stdexec::__nothrow_connectable<schedule_from_sndr_t, Tests::Utils::SinkReceiver>);

    return true;
}
static_assert(test_schedule_from_sndr_traits());

//! @test Check traits of the sender created by the customized @c stdexec::continues_on.
consteval bool test_continues_on_sndr_traits() {
    static_assert(Tests::Utils::check_continues_on<typename ContinuesOnTest::scheduler_t>());

    return true;
}
static_assert(test_continues_on_sndr_traits());

//! @test Check that the @ref Kokkos::Execution::Impl::get_exec_t query is forwarded as expected.
TEST_F(ContinuesOnTest, queryable_get_exec) {
    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const host_execution_space exec_h{};

    const Kokkos::Execution::ExecutionSpaceContext esc_h{exec_h};
    const context_t esc_A{exec_A}, esc_B{exec_B};

    auto schs_A = stdexec::schedule(esc_A.get_scheduler());

    //! The schedule sender environment is not queryable with @ref Kokkos::Execution::Impl::get_exec_t.
    static_assert(!stdexec::__queryable_with<decltype(stdexec::get_env(schs_A)), Kokkos::Execution::Impl::get_exec_t>);

    auto schs_A_then = std::move(schs_A) | THEN_LABELED('A'); // NOLINT(performance-move-const-arg)

    const auto sch_B = esc_B.get_scheduler();
    auto schs_A_then_con_B = std::move(schs_A_then) // NOLINT(performance-move-const-arg)
                           | stdexec::continues_on(sch_B);

    //! The default implementation of @c continues_on has set the completion scheduler to @c sch_B.
    ASSERT_EQ(stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(schs_A_then_con_B)), sch_B);

    auto schs_A_then_con_B_then = std::move(schs_A_then_con_B) // NOLINT(performance-move-const-arg)
                                | THEN_LABELED('B');

    static_assert(std::same_as<
                  stdexec::__demangle_t<decltype(schs_A_then_con_B_then)>,
                  Tests::Utils::basic_sender_t<
                      stdexec::then_t,
                      Tests::Utils::Functors::Labeled<'B'>,
                      Tests::Utils::basic_sender_t<
                          stdexec::continues_on_t,
                          scheduler_t,
                          Tests::Utils::basic_sender_t<
                              stdexec::schedule_from_t,
                              stdexec::__,
                              Tests::Utils::basic_sender_t<
                                  stdexec::then_t,
                                  Tests::Utils::Functors::Labeled<'A'>,
                                  schedule_sender_t
                              >
                          >
                      >
                  >
    >);

    //! Continue on an execution space instance of a different type.
    auto sch_h = esc_h.get_scheduler();
    auto schs_A_then_con_B_then_con_h_then = std::move(schs_A_then_con_B_then) // NOLINT(performance-move-const-arg)
                                           | stdexec::continues_on(sch_h) | THEN_LABELED('h');
    ASSERT_EQ(
        stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(schs_A_then_con_B_then_con_h_then)),
        sch_h);

    auto op_state = stdexec::connect(
        std::move(schs_A_then_con_B_then_con_h_then), // NOLINT(performance-move-const-arg)
        Kokkos::Execution::Impl::SyncWait::Receiver<host_execution_space>{
            .state = std::addressof(esc_h.m_state), .runloop_state = nullptr, .result = nullptr});

    /**
     * The environment of the receiver created by the customization of the most downstream @c then
     * is queryable with @ref Kokkos::Execution::Impl::get_exec_t.
     */
    using then_rcvr_t = Kokkos::Execution::Impl::Receiver<Kokkos::Execution::ExecutionSpaceImpl::OpStateBase<
        Kokkos::Execution::Impl::SyncWait::Receiver<host_execution_space>,
        Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
            std::string_view,
            Kokkos::Execution::ExecutionSpaceImpl::ThenWrapper<Tests::Utils::Functors::Labeled<'h'>>,
            Kokkos::RangePolicy<host_execution_space, Kokkos::LaunchBounds<1>>
        >
    >>;
    static_assert(std::same_as<decltype(op_state.inner_opstate.rcvr.rcvr.rcvr), then_rcvr_t>);
    static_assert(!stdexec::__queryable_with<stdexec::env_of_t<then_rcvr_t>, Kokkos::Execution::Impl::get_exec_t>);

    //! Our customization of @c continues_on forwards the environment.
    using con_h_then_rcvr_t = Kokkos::Execution::ExecutionSpaceImpl::ContinuesOnReceiver<
        Kokkos::Execution::ExecutionSpaceImpl::WithExecEnvPolicy,
        Kokkos::Execution::ExecutionSpaceImpl::Scheduler<host_execution_space>,
        then_rcvr_t
    >;
    static_assert(std::same_as<decltype(op_state.inner_opstate.rcvr.rcvr), con_h_then_rcvr_t>);
    static_assert(stdexec::__queryable_with<stdexec::env_of_t<con_h_then_rcvr_t>, Kokkos::Execution::Impl::get_exec_t>);
    static_assert(std::same_as<
                  stdexec::env_of_t<con_h_then_rcvr_t>,
                  stdexec::env<
                      stdexec::prop<
                          Kokkos::Execution::Impl::get_exec_t,
                          Kokkos::Execution::Impl::ExecutionSpaceRef<host_execution_space>
                      >,
                      stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>
                  >
    >);
    ASSERT_EQ(Kokkos::Execution::Impl::get_exec(stdexec::get_env(op_state.inner_opstate.rcvr.rcvr)).get(), exec_h);

    //! @ref Kokkos::Execution::ExecutionSpaceImpl::ScheduleFromReceiver updates the @ref Kokkos::Execution::Impl::get_exec_t query.
    using sfrom_con_h_then_rcvr_t = Kokkos::Execution::ExecutionSpaceImpl::ScheduleFromReceiver<
        Kokkos::Execution::ExecutionSpaceImpl::WithDelegatedSyncPolicy,
        Kokkos::Execution::ExecutionSpaceImpl::WithoutExecEnvPolicy,
        scheduler_t,
        con_h_then_rcvr_t
    >;
    static_assert(std::same_as<decltype(op_state.inner_opstate.rcvr), sfrom_con_h_then_rcvr_t>);
    static_assert(
        !stdexec::__queryable_with<stdexec::env_of_t<sfrom_con_h_then_rcvr_t>, Kokkos::Execution::Impl::get_exec_t>);

    using then_sfrom_con_h_then_rcvr_t =
        Kokkos::Execution::Impl::Receiver<Kokkos::Execution::ExecutionSpaceImpl::OpStateBase<
            sfrom_con_h_then_rcvr_t,
            Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
                std::string_view,
                Kokkos::Execution::ExecutionSpaceImpl::ThenWrapper<Tests::Utils::Functors::Labeled<'B'>>,
                Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::LaunchBounds<1>>
            >
        >>;
    static_assert(
        std::same_as<decltype(op_state.inner_opstate.inner_opstate.rcvr.rcvr.rcvr), then_sfrom_con_h_then_rcvr_t>);
    static_assert(!stdexec::__queryable_with<
                  stdexec::env_of_t<then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Execution::Impl::get_exec_t
    >);

    using con_B_then_sfrom_con_h_then_rcvr_t = Kokkos::Execution::ExecutionSpaceImpl::ContinuesOnReceiver<
        Kokkos::Execution::ExecutionSpaceImpl::WithExecEnvPolicy,
        scheduler_t,
        then_sfrom_con_h_then_rcvr_t
    >;
    static_assert(
        std::same_as<decltype(op_state.inner_opstate.inner_opstate.rcvr.rcvr), con_B_then_sfrom_con_h_then_rcvr_t>);
    static_assert(stdexec::__queryable_with<
                  stdexec::env_of_t<con_B_then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Execution::Impl::get_exec_t
    >);
    static_assert(std::same_as<
                  stdexec::env_of_t<con_B_then_sfrom_con_h_then_rcvr_t>,
                  stdexec::env<
                      stdexec::prop<
                          Kokkos::Execution::Impl::get_exec_t,
                          Kokkos::Execution::Impl::ExecutionSpaceRef<TEST_EXECUTION_SPACE>
                      >,
                      stdexec::__env::__fwd<stdexec::env<
                          stdexec::prop<
                              Kokkos::Execution::Impl::get_exec_t,
                              Kokkos::Execution::Impl::ExecutionSpaceRef<host_execution_space>
                          >,
                          stdexec::__env::__fwd<Kokkos::Execution::Impl::SyncWait::env>
                      >>
                  >
    >);
    ASSERT_EQ(
        Kokkos::Execution::Impl::get_exec(stdexec::get_env(op_state.inner_opstate.inner_opstate.rcvr.rcvr)).get(),
        exec_B);

    using sfrom_con_B_then_sfrom_con_h_then_rcvr_t = Kokkos::Execution::ExecutionSpaceImpl::ScheduleFromReceiver<
        Kokkos::Execution::ExecutionSpaceImpl::WithDelegatedSyncPolicy,
        Kokkos::Execution::ExecutionSpaceImpl::WithoutExecEnvPolicy,
        scheduler_t,
        con_B_then_sfrom_con_h_then_rcvr_t
    >;
    static_assert(
        std::same_as<decltype(op_state.inner_opstate.inner_opstate.rcvr), sfrom_con_B_then_sfrom_con_h_then_rcvr_t>);
    static_assert(!stdexec::__queryable_with<
                  stdexec::env_of_t<sfrom_con_B_then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Execution::Impl::get_exec_t
    >);

    using then_sfrom_con_B_then_sfrom_con_h_then_rcvr_t =
        Kokkos::Execution::Impl::Receiver<Kokkos::Execution::ExecutionSpaceImpl::OpStateBase<
            sfrom_con_B_then_sfrom_con_h_then_rcvr_t,
            Kokkos::Execution::ExecutionSpaceImpl::ParallelForClosure<
                std::string_view,
                Kokkos::Execution::ExecutionSpaceImpl::ThenWrapper<Tests::Utils::Functors::Labeled<'A'>>,
                Kokkos::RangePolicy<TEST_EXECUTION_SPACE, Kokkos::LaunchBounds<1>>
            >
        >>;
    static_assert(std::same_as<
                  decltype(op_state.inner_opstate.inner_opstate.inner_opstate.rcvr),
                  then_sfrom_con_B_then_sfrom_con_h_then_rcvr_t
    >);
    static_assert(!stdexec::__queryable_with<
                  stdexec::env_of_t<then_sfrom_con_B_then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Execution::Impl::get_exec_t
    >);
}

//! @test A @c then and a @c sync_wait following a @c continues_on properly use the execution space instance.
TEST_F(ContinuesOnTest, then_sync_wait) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    stdexec::sender auto chain = stdexec::just() | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(chain)),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization)
 *       when using it many times on the same execution space instance.
 *
 * There shouldn't be any fencing required in this case.
 */
TEST_F(ContinuesOnTest, transition_to_same_execution_space_instance) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto chain = stdexec::just() | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data)
               | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data)
               | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(chain)),
        ::testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization)
 *       when transitioning from one execution space instance to another (of the same type).
 */
TEST_F(ContinuesOnTest, transition_to_another_execution_space_instance_and_back_same_type) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const context_t esc_A{exec_A};
    Tests::Utils::show_exec_space_id(exec_A, "exec_A");
    const context_t esc_B{exec_B};
    Tests::Utils::show_exec_space_id(exec_B, "exec_B");

    auto chain = stdexec::just() | stdexec::continues_on(esc_A.get_scheduler()) | THEN_INCREMENT(data)
               | stdexec::continues_on(esc_B.get_scheduler()) | THEN_INCREMENT(data)
               | stdexec::continues_on(esc_A.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(chain));

    if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(
            recorded_events,
            ::testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "sync_wait"))));
    } else {
        ASSERT_THAT(
            recorded_events,
            ::testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "sync_wait"))));
    }

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

/**
 * @test Check that @c continues_on is properly customized (with appropriate synchronization)
 *       when transitioning from one execution space instance to another (of different type).
 */
TEST_F(ContinuesOnTest, transition_to_another_execution_space_instance_and_back_different_type) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const host_execution_space exec_h{};

    const Kokkos::Execution::ExecutionSpaceContext esc_h{exec_h};
    const context_t esc{exec};

    Tests::Utils::show_exec_space_id(exec, "exec");
    Tests::Utils::show_exec_space_id(exec_h, "exec_h");

    using level_C_env_t = Kokkos::Execution::Impl::SyncWait::env;
    using level_B_env_t = stdexec::__env::__fwd<stdexec::env<
        stdexec::prop<
            Kokkos::Execution::Impl::get_exec_t,
            Kokkos::Execution::Impl::ExecutionSpaceRef<TEST_EXECUTION_SPACE>
        >,
        stdexec::__env::__fwd<level_C_env_t>
    >>;
    using level_A_env_t = stdexec::__env::__fwd<stdexec::env<
        stdexec::prop<
            Kokkos::Execution::Impl::get_exec_t,
            Kokkos::Execution::Impl::ExecutionSpaceRef<host_execution_space>
        >,
        level_B_env_t
    >>;
    auto chain = stdexec::just() | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data)
               | Tests::Utils::check_rcvr_env<level_A_env_t>() | stdexec::continues_on(esc_h.get_scheduler())
               | THEN_INCREMENT(data) | Tests::Utils::check_rcvr_env<level_B_env_t>()
               | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data)
               | Tests::Utils::check_rcvr_env<level_C_env_t>();

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(chain));

    if (Tests::Utils::are_same_instances(exec, exec_h)) {
        ASSERT_THAT(
            recorded_events,
            ::testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    } else {
        ASSERT_THAT(
            recorded_events,
            ::testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_h, dispatch_label(exec_h, "schedule_from")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    }

    ASSERT_EQ(data(), 3) << "A synchronization is missing.";
}

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_nothrow_transformable() {
    using sndr_continues_on_t =
        decltype(stdexec::just() | stdexec::continues_on(std::declval<typename ContinuesOnTest::scheduler_t>()));

    static_assert(std::same_as<
                  stdexec::__demangle_t<sndr_continues_on_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::continues_on_t,
                      typename ContinuesOnTest::scheduler_t,
                      Tests::Utils::basic_sender_t<
                          stdexec::schedule_from_t,
                          stdexec::__,
                          Tests::Utils::basic_sender_t<stdexec::just_t, stdexec::__tuple<>>
                      >
                  >
    >);

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::set_value_t,
                  sndr_continues_on_t&&,
                  stdexec::env<>
    >);

    using sndr_schedule_from_t = decltype(stdexec::schedule_from(
        stdexec::schedule(std::declval<typename ContinuesOnTest::scheduler_t>())));

    static_assert(std::same_as<
                  stdexec::__demangle_t<sndr_schedule_from_t>,
                  Tests::Utils::basic_sender_t<
                      stdexec::schedule_from_t,
                      stdexec::__,
                      typename ContinuesOnTest::schedule_sender_t
                  >
    >);

    static_assert(stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Execution::ExecutionSpaceImpl::Domain,
                  stdexec::set_value_t,
                  sndr_schedule_from_t&&,
                  stdexec::env<>
    >);

    return true;
}
static_assert(test_sndr_nothrow_transformable());

} // namespace Tests::ExecutionSpaceImpl
