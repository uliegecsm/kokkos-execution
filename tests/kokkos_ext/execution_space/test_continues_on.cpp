#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/Utils.hpp"
#include "tests/kokkos_ext/Helpers.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"

/**
 * @addtogroup unittests
 *
 * Customization of @c continues_on by @c Kokkos::Experimental::ExecutionSpaceContext
 * ----------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext properly customizes
 * @c continues_on.
 *
 * The tests can be found in @ref tests/kokkos_ext/execution_space/test_continues_on.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;
using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace tests::kokkos_ext {

using namespace Kokkos::utils::callbacks;

class ContinuesOnTest
    : public impl::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

//! @test Check traits of the sender created by the customized @c continues_on.
TEST_F(ContinuesOnTest, traits) {
    static_assert(::utils::check_continues_on<decltype(context_t{exec}.get_scheduler())>());
}

template <char ID>
struct DummyFunctor {
    KOKKOS_FUNCTION void operator()() const {
    }
};

//! @test Check that the @ref Kokkos::Experimental::details::execution_space::get_exec_t query is forwarded as expected.
TEST_F(ContinuesOnTest, queryable_get_exec) {
    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const host_execution_space exec_h{};

    const Kokkos::Experimental::ExecutionSpaceContext esc_h{exec_h};
    const context_t esc_A{exec_A}, esc_B{exec_B};

    auto schs_A = ::stdexec::schedule(esc_A.get_scheduler());

    //! The schedule sender environment is not queryable with @ref Kokkos::Experimental::details::execution_space::get_exec_t.
    static_assert(!::stdexec::__queryable_with<
                  decltype(::stdexec::get_env(schs_A)),
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);

    auto schs_A_then = std::move(schs_A) | ::stdexec::then(DummyFunctor<'A'>{}); // NOLINT(performance-move-const-arg)

    const auto sch_B = esc_B.get_scheduler();
    auto schs_A_then_con_B = std::move(schs_A_then) // NOLINT(performance-move-const-arg)
                           | ::stdexec::continues_on(sch_B);

    //! The default implementation of @c continues_on has set the completion scheduler to @c sch_B.
    ASSERT_EQ(
        ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(::stdexec::get_env(schs_A_then_con_B)), sch_B);

    auto schs_A_then_con_B_then = std::move(schs_A_then_con_B) // NOLINT(performance-move-const-arg)
                                | ::stdexec::then(DummyFunctor<'B'>{});

    static_assert(std::same_as<
                  ::stdexec::__demangle_t<decltype(schs_A_then_con_B_then)>,
                  ::tests::stdexec::basic_sender<
                      ::stdexec::then_t,
                      tests::kokkos_ext::DummyFunctor<'B'>,
                      ::tests::stdexec::basic_sender<
                          ::stdexec::continues_on_t,
                          Kokkos::Experimental::details::execution_space::Scheduler<execution_space>,
                          ::tests::stdexec::basic_sender<
                              ::stdexec::schedule_from_t,
                              ::stdexec::__,
                              ::tests::stdexec::basic_sender<
                                  ::stdexec::then_t,
                                  tests::kokkos_ext::DummyFunctor<'A'>,
                                  Kokkos::Experimental::details::execution_space::Scheduler<execution_space>::Sender
                              >
                          >
                      >
                  >
    >);

    //! Continue on an execution space instance of a different type.
    auto sch_h = esc_h.get_scheduler();
    auto schs_A_then_con_B_then_con_h_then = std::move(schs_A_then_con_B_then) // NOLINT(performance-move-const-arg)
                                           | ::stdexec::continues_on(sch_h) | ::stdexec::then(DummyFunctor<'h'>{});
    ASSERT_EQ(
        ::stdexec::get_completion_scheduler<::stdexec::set_value_t>(
            ::stdexec::get_env(schs_A_then_con_B_then_con_h_then)),
        sch_h);

    auto op_state = ::stdexec::connect(
        std::move(schs_A_then_con_B_then_con_h_then), // NOLINT(performance-move-const-arg)
        Kokkos::Experimental::details::execution_space::SyncWaitReceiver<host_execution_space>{
            .state = std::addressof(esc_h.m_state), .runloop_state = nullptr});

    //! Helper to check that an environment is as expected.
    constexpr auto check_env = []<typename T, Kokkos::ExecutionSpace Exec>() constexpr -> bool {
        return std::same_as<
            ::stdexec::env_of_t<T>,
            ::stdexec::__env::__fwd<::stdexec::env<
                ::stdexec::prop<
                    Kokkos::Experimental::details::execution_space::get_exec_t,
                    Kokkos::Experimental::details::execution_space::ExecutionSpaceRef<Exec>
                >,
                ::stdexec::__env::__fwd<Kokkos::Experimental::details::impl::env>
            >>
        >;
    };

    /**
     * The environment of the receiver created by the customization of the most downstream @c then
     * is queryable with @ref Kokkos::Experimental::details::execution_space::get_exec_t.
     */
    using then_rcvr_t = Kokkos::Experimental::details::execution_space::OpStateReceiver<
        Kokkos::Experimental::details::execution_space::OpStateBase<
            Kokkos::Experimental::details::execution_space::SyncWaitReceiver<host_execution_space>,
            Kokkos::Experimental::details::execution_space::ParallelForClosure<
                Kokkos::Experimental::details::execution_space::ThenWrapper<tests::kokkos_ext::DummyFunctor<'h'>>,
                Kokkos::RangePolicy<host_execution_space, Kokkos::LaunchBounds<1>>
            >
        >
    >;
    static_assert(std::same_as<decltype(op_state.inner_opstate.rcvr.rcvr.rcvr), then_rcvr_t>);
    static_assert(::stdexec::__queryable_with<
                  ::stdexec::env_of_t<then_rcvr_t>,
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);
    static_assert(check_env.template operator()<then_rcvr_t, host_execution_space>());
    ASSERT_EQ(
        Kokkos::Experimental::details::execution_space::get_exec(
            ::stdexec::get_env(op_state.inner_opstate.rcvr.rcvr.rcvr))
            .get(),
        exec_h);

    //! Our customization of @c continues_on forwards the environment.
    using con_h_then_rcvr_t = Kokkos::Experimental::details::execution_space::ContinuesOnReceiver<then_rcvr_t>;
    static_assert(std::same_as<decltype(op_state.inner_opstate.rcvr.rcvr), con_h_then_rcvr_t>);
    static_assert(::stdexec::__queryable_with<
                  ::stdexec::env_of_t<con_h_then_rcvr_t>,
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);
    static_assert(check_env.template operator()<con_h_then_rcvr_t, host_execution_space>());
    ASSERT_EQ(
        Kokkos::Experimental::details::execution_space::get_exec(::stdexec::get_env(op_state.inner_opstate.rcvr.rcvr))
            .get(),
        exec_h);

    //! @ref Kokkos::Experimental::details::execution_space::ScheduleFromReceiver updates the @ref Kokkos::Experimental::details::execution_space::get_exec_t query.
    using sfrom_con_h_then_rcvr_t = Kokkos::Experimental::details::execution_space::ScheduleFromReceiver<
        Kokkos::Experimental::details::execution_space::Scheduler<execution_space>,
        con_h_then_rcvr_t
    >;
    static_assert(std::same_as<decltype(op_state.inner_opstate.rcvr), sfrom_con_h_then_rcvr_t>);
    static_assert(::stdexec::__queryable_with<
                  ::stdexec::env_of_t<sfrom_con_h_then_rcvr_t>,
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);
    static_assert(check_env.template operator()<sfrom_con_h_then_rcvr_t, execution_space>());
    ASSERT_EQ(
        Kokkos::Experimental::details::execution_space::get_exec(::stdexec::get_env(op_state.inner_opstate.rcvr)).get(),
        exec_B);

    using then_sfrom_con_h_then_rcvr_t = Kokkos::Experimental::details::execution_space::OpStateReceiver<
        Kokkos::Experimental::details::execution_space::OpStateBase<
            sfrom_con_h_then_rcvr_t,
            Kokkos::Experimental::details::execution_space::ParallelForClosure<
                Kokkos::Experimental::details::execution_space::ThenWrapper<tests::kokkos_ext::DummyFunctor<'B'>>,
                Kokkos::RangePolicy<execution_space, Kokkos::LaunchBounds<1>>
            >
        >
    >;
    static_assert(
        std::same_as<decltype(op_state.inner_opstate.inner_opstate.rcvr.rcvr.rcvr), then_sfrom_con_h_then_rcvr_t>);
    static_assert(::stdexec::__queryable_with<
                  ::stdexec::env_of_t<then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);
    static_assert(check_env.template operator()<then_sfrom_con_h_then_rcvr_t, execution_space>());
    ASSERT_EQ(
        Kokkos::Experimental::details::execution_space::get_exec(
            ::stdexec::get_env(op_state.inner_opstate.inner_opstate.rcvr.rcvr.rcvr))
            .get(),
        exec_B);

    using con_B_then_sfrom_con_h_then_rcvr_t =
        Kokkos::Experimental::details::execution_space::ContinuesOnReceiver<then_sfrom_con_h_then_rcvr_t>;
    static_assert(
        std::same_as<decltype(op_state.inner_opstate.inner_opstate.rcvr.rcvr), con_B_then_sfrom_con_h_then_rcvr_t>);
    static_assert(::stdexec::__queryable_with<
                  ::stdexec::env_of_t<con_B_then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);
    static_assert(check_env.template operator()<con_B_then_sfrom_con_h_then_rcvr_t, execution_space>());
    ASSERT_EQ(
        Kokkos::Experimental::details::execution_space::get_exec(
            ::stdexec::get_env(op_state.inner_opstate.inner_opstate.rcvr.rcvr))
            .get(),
        exec_B);

    using sfrom_con_B_then_sfrom_con_h_then_rcvr_t =
        Kokkos::Experimental::details::execution_space::ScheduleFromReceiver<
            Kokkos::Experimental::details::execution_space::Scheduler<execution_space>,
            con_B_then_sfrom_con_h_then_rcvr_t
        >;
    static_assert(
        std::same_as<decltype(op_state.inner_opstate.inner_opstate.rcvr), sfrom_con_B_then_sfrom_con_h_then_rcvr_t>);
    static_assert(::stdexec::__queryable_with<
                  ::stdexec::env_of_t<sfrom_con_B_then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);
    static_assert(check_env.template operator()<sfrom_con_B_then_sfrom_con_h_then_rcvr_t, execution_space>());
    ASSERT_EQ(
        Kokkos::Experimental::details::execution_space::get_exec(
            ::stdexec::get_env(op_state.inner_opstate.inner_opstate.rcvr))
            .get(),
        exec_A);

    using then_sfrom_con_B_then_sfrom_con_h_then_rcvr_t =
        Kokkos::Experimental::details::execution_space::OpStateReceiver<
            Kokkos::Experimental::details::execution_space::OpStateBase<
                sfrom_con_B_then_sfrom_con_h_then_rcvr_t,
                Kokkos::Experimental::details::execution_space::ParallelForClosure<
                    Kokkos::Experimental::details::execution_space::ThenWrapper<tests::kokkos_ext::DummyFunctor<'A'>>,
                    Kokkos::RangePolicy<execution_space, Kokkos::LaunchBounds<1>>
                >
            >
        >;
    static_assert(std::same_as<
                  decltype(op_state.inner_opstate.inner_opstate.inner_opstate.rcvr),
                  then_sfrom_con_B_then_sfrom_con_h_then_rcvr_t
    >);
    static_assert(::stdexec::__queryable_with<
                  ::stdexec::env_of_t<then_sfrom_con_B_then_sfrom_con_h_then_rcvr_t>,
                  Kokkos::Experimental::details::execution_space::get_exec_t
    >);
    static_assert(check_env.template operator()<then_sfrom_con_B_then_sfrom_con_h_then_rcvr_t, execution_space>());
    ASSERT_EQ(
        Kokkos::Experimental::details::execution_space::get_exec(
            ::stdexec::get_env(op_state.inner_opstate.inner_opstate.inner_opstate.rcvr))
            .get(),
        exec_A);
}

//! @test A @c then and a @c sync_wait following a @c continues_on properly use the execution space instance.
TEST_F(ContinuesOnTest, then_sync_wait) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    ::stdexec::sender auto chain = ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
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

    auto chain = ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN
               | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN | ::stdexec::continues_on(esc.get_scheduler())
               | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); }),
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
    SHOW_EXEC_SPACE_ID(exec_A)
    const context_t esc_B{exec_B};
    SHOW_EXEC_SPACE_ID(exec_B)

    auto chain = ::stdexec::just() | ::stdexec::continues_on(esc_A.get_scheduler()) | ADD_THEN
               | ::stdexec::continues_on(esc_B.get_scheduler()) | ADD_THEN
               | ::stdexec::continues_on(esc_A.get_scheduler()) | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); });

    if (are_same_instances(exec_A, exec_B)) {
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

    const Kokkos::Experimental::ExecutionSpaceContext esc_h{exec_h};
    const context_t esc{exec};

    SHOW_EXEC_SPACE_ID(exec)
    SHOW_EXEC_SPACE_ID(exec_h)

    auto chain = ::stdexec::just() | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN
               | ::stdexec::continues_on(esc_h.get_scheduler()) | ADD_THEN
               | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN;

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); });

    if (are_same_instances(exec, exec_h)) {
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

//! @test No kernel launch happens, but @c stdexec::sync_wait fences when starting with @c stdexec::just_stopped.
TEST_F(ContinuesOnTest, just_stopped) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};
    SHOW_EXEC_SPACE_ID(exec)

    auto chain = ::stdexec::just_stopped() | ::stdexec::continues_on(esc.get_scheduler()) | ADD_THEN;

    static_assert(
        ::tests::stdexec::has_completion_signatures<decltype(chain), ::stdexec::__mset<::stdexec::set_stopped_t()>>);

    static_assert(::tests::stdexec::has_completion_signatures<
                  decltype(chain),
                  ::stdexec::__mset<
                      ::stdexec::set_stopped_t(),
                      ::stdexec::set_value_t(),
                      ::stdexec::set_error_t(std::exception_ptr)
                  >,
                  ::stdexec::env<>
    >);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable { ::stdexec::sync_wait(std::move(chain)); });

    ASSERT_THAT(
        recorded_events, ::testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 0) << "It should not execute on 'set_error'.";
}

//! @test Check @c noexcept specification of sender transformation.
consteval bool test_sndr_no_throw_transformable() {
    using sndr_continues_on_t =
        decltype(::stdexec::just() | ::stdexec::continues_on(std::declval<typename ContinuesOnTest::scheduler_t>()));

    static_assert(std::same_as<
                  ::stdexec::__demangle_t<sndr_continues_on_t>,
                  ::tests::stdexec::basic_sender<
                      ::stdexec::continues_on_t,
                      typename ContinuesOnTest::scheduler_t,
                      ::tests::stdexec::basic_sender<
                          ::stdexec::schedule_from_t,
                          ::stdexec::__,
                          ::tests::stdexec::basic_sender<::stdexec::just_t, ::stdexec::__tuple<>>
                      >
                  >
    >);

    static_assert(::stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Experimental::details::execution_space::Domain,
                  ::stdexec::set_value_t,
                  sndr_continues_on_t&&,
                  ::stdexec::env<>
    >);

    using sndr_schedule_from_t = decltype(::stdexec::schedule_from(
        ::stdexec::schedule(std::declval<typename ContinuesOnTest::scheduler_t>())));

    static_assert(std::same_as<
                  ::stdexec::__demangle_t<sndr_schedule_from_t>,
                  ::tests::stdexec::basic_sender<
                      ::stdexec::schedule_from_t,
                      ::stdexec::__,
                      typename ContinuesOnTest::schedule_sender_t
                  >
    >);

    static_assert(::stdexec::__detail::__has_nothrow_transform_sender<
                  Kokkos::Experimental::details::execution_space::Domain,
                  ::stdexec::set_value_t,
                  sndr_schedule_from_t&&,
                  ::stdexec::env<>
    >);

    return true;
}
static_assert(test_sndr_no_throw_transformable());

} // namespace tests::kokkos_ext
