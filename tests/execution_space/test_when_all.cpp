#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/execution_space.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_rcvr_env_queryable_with.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/labeled.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/kokkos.hpp"
#include "tests/utils/sink_receiver.hpp"
#include "tests/utils/stdexec.hpp"
#include "tests/utils/sync_wait.hpp"

/**
 * @addtogroup unittests
 *
 * Interaction of @c Kokkos::Execution::ExecutionSpaceContext with @c stdexec::when_all
 * ------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly interacts with
 * @c stdexec::when_all.
 *
 * The tests can be found in @ref tests/execution_space/test_when_all.cpp.
 */

using host_execution_space = Kokkos::DefaultHostExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class WhenAllTest
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

//! @test Check traits of @c Kokkos::Execution::ExecutionSpaceImpl::WhenAllSender with two branches.
consteval bool test_when_all_two_branches_traits() {
    //! Schedule sender.
    using schd_sndr_t = typename WhenAllTest::schedule_sender_t;

    //! Parallel for sender.
    using functor_t = Tests::Utils::Functors::SumIndices<typename WhenAllTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using pfor_sndr_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<
        Kokkos::Execution::parallel_for_t,
        schd_sndr_t,
        std::string,
        functor_t,
        policy_t
    >;

    //! When all sender of two parallel for senders.
    using when_all_sndr_t = Kokkos::Execution::ExecutionSpaceImpl::WhenAllSender<pfor_sndr_t, pfor_sndr_t>;

    //! Has the expected completion signatures.
    using completion_signatures_t = stdexec::__completion_signatures_of_t<when_all_sndr_t, stdexec::env<>>;

    static_assert(stdexec::__mset_eq<
                  stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>,
                  completion_signatures_t
    >);

    //! Has the expected completion domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, when_all_sndr_t, stdexec::env<>>,
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! Does not advertise a completion scheduler.
    static_assert(!Tests::Utils::has_completion_scheduler_for<when_all_sndr_t, stdexec::set_value_t, stdexec::env<>>);

    //! Is connectable.
    static_assert(stdexec::sender_to<when_all_sndr_t, Tests::Utils::SinkReceiver>);

    //! It is nothrow connectable.
    static_assert(stdexec::__nothrow_connectable<when_all_sndr_t, Tests::Utils::SinkReceiver>);

    //! The branches are connected to a receiver that supports submitted depend on, but not submitted order on.
    using connect_result_t = stdexec::connect_result_t<when_all_sndr_t, Tests::Utils::SinkReceiver>;
    using when_all_child_receiver_0_t = Kokkos::Execution::ExecutionSpaceImpl::WhenAllChildReceiver<
        TEST_EXECUTION_SPACE,
        0,
        connect_result_t,
        stdexec::env<>
    >;
    static_assert(
        Kokkos::Execution::Impl::supports_submitted_depend_on<when_all_child_receiver_0_t, TEST_EXECUTION_SPACE>);
    static_assert(!Kokkos::Execution::Impl::supports_submitted_order_on<when_all_child_receiver_0_t>);

    return true;
}
static_assert(test_when_all_two_branches_traits());

/**
 * @test Check traits of @c Kokkos::Execution::ExecutionSpaceImpl::WhenAllSender with a mixed branch
 *       in an environment that contains a stop token.
 */
consteval bool test_when_all_mixed_branch_traits() {
    //! Schedule sender.
    using stc_schd_sndr_t = decltype(stdexec::schedule(
        std::declval<experimental::execution::single_thread_context>().get_scheduler()));

    //! Mixed branch with continues on and parallel for sender.
    using con_sndr_t =
        Kokkos::Execution::ExecutionSpaceImpl::ContinuesOnSender<typename WhenAllTest::scheduler_t, stc_schd_sndr_t>;
    using functor_t = Tests::Utils::Functors::SumIndices<typename WhenAllTest::view_s_t>;
    using policy_t = Kokkos::RangePolicy<TEST_EXECUTION_SPACE>;
    using pfor_sndr_t = Kokkos::Execution::ExecutionSpaceImpl::ParallelForSender<
        Kokkos::Execution::parallel_for_t,
        con_sndr_t,
        std::string,
        functor_t,
        policy_t
    >;
    using when_all_sndr_t = Kokkos::Execution::ExecutionSpaceImpl::WhenAllSender<pfor_sndr_t>;

    //! Has the expected completion signatures.
    using env_with_stop_token_t = stdexec::prop<stdexec::get_stop_token_t, stdexec::inplace_stop_token>;
    static_assert(stdexec::sends_stopped<stc_schd_sndr_t, env_with_stop_token_t>);

    using completion_signatures_t = stdexec::__completion_signatures_of_t<when_all_sndr_t, env_with_stop_token_t>;

    static_assert(
        stdexec::__mset_eq<
            stdexec::__mset<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr), stdexec::set_stopped_t()>,
            completion_signatures_t
        >);

    //! Has the expected completion domain.
    static_assert(std::same_as<
                  stdexec::__completion_domain_of_t<stdexec::set_value_t, when_all_sndr_t, env_with_stop_token_t>,
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    //! Does not advertise a completion scheduler.
    static_assert(
        !Tests::Utils::has_completion_scheduler_for<when_all_sndr_t, stdexec::set_value_t, env_with_stop_token_t>);

    return true;
}
static_assert(test_when_all_mixed_branch_traits());

/**
 * @test A @c stdexec::when_all with a single branch on @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then -- when_all --> sync_wait
 * @endverbatim
 */
TEST_F(WhenAllTest, single_branch) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto sndr = stdexec::when_all(stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data));

    static_assert(std::same_as<
                  decltype(stdexec::get_completion_domain<stdexec::set_value_t>(stdexec::get_env(sndr))),
                  Kokkos::Execution::ExecutionSpaceImpl::Domain
    >);

    /// Even though the sender returned by @c stdexec::when_all is in the customized domain,
    /// it does not have a completion scheduler and so it is not an execution space completing sender.
    static_assert(!Kokkos::Execution::ExecutionSpaceImpl::execution_space_completing_sender<decltype(sndr)>);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    /// Because the sender returned by @c stdexec::when_all is not an execution space completing sender,
    /// the default implementation of @c stdexec::sync_wait is used.
    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_RECORD_EVENT(exec),
            MATCHER_FOR_WAIT_EVENT(recorded_events.at(1))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test A @c stdexec::when_all with a single branch on @ref Kokkos::Execution::ExecutionSpaceContext,
 *       followed by work on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then -- when_all --> continues_on(esc) | then
 * @endverbatim
 */
TEST_F(WhenAllTest, single_branch_followed_by_self) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto sndr = stdexec::when_all(stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data))
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 2);
}

/**
 * @test A @c stdexec::when_all with a single branch with a segment on @ref Kokkos::Execution::ExecutionSpaceContext
 *       followed by a segment on another context. The @c stdexec::when_all is followed by work on the same
 *       @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then --> continues_on(stc) | then -- when_all --> continues_on(esc) | then
 * @endverbatim
 */
TEST_F(WhenAllTest, single_mixed_branch_followed_by_self) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data)
                    | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT(data))
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test A @c stdexec::when_all with a single branch on @ref Kokkos::Execution::ExecutionSpaceContext, followed by work
 *       on another context, followed by work on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then -- when_all --> continues_on(stc) | then --> continues_on(esc) | then
 * @endverbatim
 */
TEST_F(WhenAllTest, single_branch_followed_by_other_and_finish_on_self) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data))
              | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT(data)
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_RECORD_EVENT(exec),
            MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test A @c stdexec::when_all with two branches, the first on @ref Kokkos::Execution::ExecutionSpaceContext and the second
 *       on another context, followed by work on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then_atomic ------------------------ \
 *                                                       when_all --> continues_on(esc) | then
 * schedule(stc) | then_atomic --> continues_on(esc) -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, two_mixed_branches_followed_by_self) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto w_a = stdexec::when_all(
        stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data),
        stdexec::schedule(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
            | stdexec::continues_on(esc.get_scheduler()));

    static_assert(
        std::same_as<
            decltype(stdexec::get_completion_domain<stdexec::set_value_t>(stdexec::get_env(w_a), stdexec::env<>{})),
            Kokkos::Execution::ExecutionSpaceImpl::Domain
        >);

    auto sndr = std::move(w_a) | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test A @c stdexec::when_all with two branches, the first on @ref Kokkos::Execution::ExecutionSpaceContext instance A
 *       and the second on @ref Kokkos::Execution::ExecutionSpaceContext instance B, followed by work on instance A.
 *
 * @verbatim
 * schedule(esc_A) | then_atomic -- \
 *                                   when_all --> continues_on(esc_A) | then
 * schedule(esc_B) | then_atomic -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, two_branches_followed_by_self) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const context_t esc_A{exec_A}, esc_B{exec_B};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc_A.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data),
                    stdexec::schedule(esc_B.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data))
              | stdexec::continues_on(esc_A.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    if constexpr (Kokkos::Execution::Impl::has_exec_wait_event<TEST_EXECUTION_SPACE>) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                MATCHER_FOR_RECORD_EVENT(exec_B),
                MATCHER_FOR_WAIT_EXEC_EVENT(exec_A, recorded_events.at(2)),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
    } else {
        if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
        } else {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                    MATCHER_FOR_RECORD_EVENT(exec_B),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
        }
    }

    ASSERT_EQ(data(), 3);
}

/**
 * @test A @c stdexec::when_all with two branches, the first on a @ref Kokkos::Execution::ExecutionSpaceContext
 *       instantiated for the test execution space type and the second on a @ref Kokkos::Execution::ExecutionSpaceContext
 *       instantiated for the default host execution space type, followed by work on the first execution context.
 *
 * @verbatim
 * schedule(esc)   | then_atomic -- \
 *                                   when_all --> continues_on(esc) | then
 * schedule(esc_h) | then_atomic -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, two_branches_host_device_followed_by_device) {
    using context_h_t = Kokkos::Execution::ExecutionSpaceContext<host_execution_space>;

    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    const host_execution_space exec_h{};
    const context_h_t esc_h{exec_h};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data),
                    stdexec::schedule(esc_h.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data))
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    if constexpr (
        Kokkos::Execution::Impl::has_exec_wait_event<host_execution_space>
        && std::same_as<host_execution_space, TEST_EXECUTION_SPACE>) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                MATCHER_FOR_RECORD_EVENT(exec_h),
                MATCHER_FOR_WAIT_EXEC_EVENT(exec, recorded_events.at(2)),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    } else {
        if (Tests::Utils::are_same_instances(exec, exec_h)) {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
        } else {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                    MATCHER_FOR_RECORD_EVENT(exec_h),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
        }
    }

    ASSERT_EQ(data(), 3);
}

/**
 * @test A @c stdexec::when_all with two branches, the first on @ref Kokkos::Execution::ExecutionSpaceContext
 *       and the second on another context, followed by work on the same other context, followed by work
 *       on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then_atomic --> continues_on(stc) -- \
 *                                                       when_all --> continues_on(stc) | then --> continues_on(esc) | then
 * schedule(stc) | then_atomic ------------------------ /
 * @endverbatim
 */
TEST_F(WhenAllTest, two_mixed_branches_followed_by_other_and_finish_on_self) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                        | stdexec::continues_on(stc.get_scheduler()),
                    stdexec::schedule(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data))
              | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT(data)
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 4);
}

/**
 * @test A nested @c stdexec::when_all. There are two branches. The first branch is on @ref Kokkos::Execution::ExecutionSpaceContext.
 *       The second branch is itself a @c stdexec::when_all with a single branch with a segment on the same @ref Kokkos::Execution::ExecutionSpaceContext
 *       instance, followed by work on another context.
 *
 * @todo There is a missing fence. Our @c stdexec::continues_on puts the execution space instance in the environment
 *       because its preceding @c stdexec::when_all has at least one branch on @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then_atomic ------------------------------------------------------------------------ \
 *                                                                                                       when_all --> continues_on(esc) | then
 * schedule(esc) | then_atomic -- when_all --> continues_on(stc) | then_atomic --> continues_on(esc) -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, nested_with_inner_followed_by_other) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data),
                    stdexec::when_all(stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data))
                        | Tests::Utils::check_rcvr_env_not_queryable_with<Kokkos::Execution::Impl::get_exec_t>()
                        | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                        | stdexec::continues_on(esc.get_scheduler()))
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_RECORD_EVENT(exec),
            MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 4);
}

/**
 * @test Verify that an independent branch can overlap with a nested @c stdexec::when_all.
 *
 * @verbatim
 * schedule(esc_A) | then('A') -- \
 *                                 when_all --> continues_on(esc_A) | then('D') -- \
 * schedule(esc_B) | then('B') -- /                                                 when_all
 *                                                                                 /
 * schedule(esc_C) | then('C') ---------------------------------------------------
 * @endverbatim
 */
TEST_F(WhenAllTest, nested_when_all_with_independent_branch) {
    const auto [exec_A, exec_B, exec_C] = Kokkos::Experimental::partition_space(exec, 1, 1, 1);

    const context_t esc_A{exec_A}, esc_B{exec_B}, esc_C{exec_C};

    auto br_A = ::stdexec::schedule(esc_A.get_scheduler()) | THEN_LABELED_PFOR(TEST_EXECUTION_SPACE, 'A');
    auto br_B = ::stdexec::schedule(esc_B.get_scheduler()) | THEN_LABELED_PFOR(TEST_EXECUTION_SPACE, 'B');
    auto br_C = ::stdexec::schedule(esc_C.get_scheduler()) | THEN_LABELED_PFOR(TEST_EXECUTION_SPACE, 'C');

    auto when_AB_then_D = ::stdexec::when_all(std::move(br_A), std::move(br_B))
                        | ::stdexec::continues_on(esc_A.get_scheduler()) | THEN_LABELED_PFOR(TEST_EXECUTION_SPACE, 'D');

    auto sndr = ::stdexec::when_all(std::move(when_AB_then_D), std::move(br_C));

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    if constexpr (Kokkos::Execution::Impl::has_exec_wait_event<TEST_EXECUTION_SPACE>) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                MATCHER_FOR_RECORD_EVENT(exec_B),
                MATCHER_FOR_WAIT_EXEC_EVENT(exec_A, recorded_events.at(2)),
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                MATCHER_FOR_RECORD_EVENT(exec_A),
                MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                MATCHER_FOR_RECORD_EVENT(exec_C),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(5)),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(7))));
    } else {
        if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                    MATCHER_FOR_RECORD_EVENT(exec_A),
                    MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                    MATCHER_FOR_RECORD_EVENT(exec_C),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(3)),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(5))));
        } else {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                    MATCHER_FOR_RECORD_EVENT(exec_B),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                    MATCHER_FOR_RECORD_EVENT(exec_A),
                    MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                    MATCHER_FOR_RECORD_EVENT(exec_C),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(5)),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(7))));
        }
    }
}

/**
 * @test Stress-test the @c stdexec::when_all customization by creating many branches, each starting in
 *       a dedicated @c experimental::execution::single_thread_context, using its own @ref Kokkos::Execution::ExecutionSpaceContext.
 */
TEST_F(WhenAllTest, many_concurrent_branches) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    constexpr size_t num_branches = 6;

    unsigned int counter_start = 0, counter_after_stc = 0;
    std::array<unsigned int, num_branches> order_start{}, order_after_stc{};

    const auto [exec_A, exec_B, exec_C, exec_D, exec_E, exec_F] =
        Kokkos::Experimental::partition_space(exec, 1, 1, 1, 1, 1, 1);

    using view_um_h_t = Kokkos::View<unsigned int, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using increment_and_memorize_t = Tests::Utils::Functors::FetchIncrement<view_um_h_t, false>;

#define DEFINE_ONE_BRANCH(_letter_, _id_)                                                                              \
    const context_t esc_##_letter_{exec_##_letter_};                                                                   \
    experimental::execution::single_thread_context stc_##_letter_{};                                                   \
    auto br_##_letter_ = stdexec::just()                                                                               \
                       | stdexec::then(                                                                                \
                             increment_and_memorize_t{                                                                 \
                                 .counter = view_um_h_t{std::addressof(counter_start)},                                \
                                 .value = view_um_h_t{std::addressof(order_start.at(_id_))}})                          \
                       | stdexec::continues_on(stc_##_letter_.get_scheduler())                                         \
                       | stdexec::then(                                                                                \
                             increment_and_memorize_t{                                                                 \
                                 .counter = view_um_h_t{std::addressof(counter_after_stc)},                            \
                                 .value = view_um_h_t{std::addressof(order_after_stc.at(_id_))}})                      \
                       | stdexec::continues_on(esc_##_letter_.get_scheduler()) | THEN_INCREMENT_ATOMIC(Device, data);

    DEFINE_ONE_BRANCH(A, 0)
    DEFINE_ONE_BRANCH(B, 1)
    DEFINE_ONE_BRANCH(C, 2)
    DEFINE_ONE_BRANCH(D, 3)
    DEFINE_ONE_BRANCH(E, 4)
    DEFINE_ONE_BRANCH(F, 5)

    auto sndr = stdexec::when_all(
        std::move(br_A), std::move(br_B), std::move(br_C), std::move(br_D), std::move(br_E), std::move(br_F));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    stdexec::sync_wait(std::move(sndr));

    ASSERT_EQ(counter_start, num_branches);
    ASSERT_EQ(counter_after_stc, num_branches);

    ASSERT_EQ(data(), num_branches);

    //! Branches are started in order.
    ASSERT_THAT(order_start, testing::ElementsAre(0, 1, 2, 3, 4, 5));

    /// However, since the branch uses @c experimental::execution::single_thread_context,
    /// the order in which the branches terminate is not deterministic.
    // NOLINTBEGIN(modernize-use-std-print)
#define SHOW_ONE_BRANCH_ORDER_AFTER_STC(_letter_, _id_)                                                                \
    SCOPED_TRACE(testing::Message() << "Branch " #_letter_ " order after 'stc' is " << order_after_stc.at(_id_));

    SHOW_ONE_BRANCH_ORDER_AFTER_STC(A, 0)
    SHOW_ONE_BRANCH_ORDER_AFTER_STC(B, 1)
    SHOW_ONE_BRANCH_ORDER_AFTER_STC(C, 2)
    SHOW_ONE_BRANCH_ORDER_AFTER_STC(D, 3)
    SHOW_ONE_BRANCH_ORDER_AFTER_STC(E, 4)
    SHOW_ONE_BRANCH_ORDER_AFTER_STC(F, 5)
    // NOLINTEND(modernize-use-std-print)

    ASSERT_THAT(order_after_stc, testing::UnorderedElementsAre(0, 1, 2, 3, 4, 5));
}

} // namespace Tests::ExecutionSpaceImpl
