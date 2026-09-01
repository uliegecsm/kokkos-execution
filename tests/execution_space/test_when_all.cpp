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
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/kokkos.hpp"
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

//! @test A @c stdexec::when_all with no branch is equivalent to @c stdexec::just().
TEST(WhenAll, no_branch) {
    auto sndr = stdexec::when_all();

    static_assert(std::same_as<stdexec::tag_of_t<decltype(sndr)>, stdexec::just_t>);

    static_assert(!stdexec::dependent_sender<decltype(sndr)>);

    static_assert(
        stdexec::get_completion_signatures<decltype(sndr)>()
        == stdexec::completion_signatures<stdexec::set_value_t()>{});

    ASSERT_TRUE(stdexec::sync_wait(std::move(sndr)).has_value()); // NOLINT(performance-move-const-arg)
}

//! @test A @c stdexec::when_all(sndr) with a single branch is equivalent to @c auto(sndr).
TEST(WhenAll, single_branch) {
    auto sndr = stdexec::just(42);

    static_assert(std::same_as<decltype(stdexec::when_all(sndr)), decltype(sndr)>);
    static_assert(std::same_as<
                  decltype(stdexec::when_all(std::move(sndr))), // NOLINT(performance-move-const-arg)
                  decltype(sndr)
    >);

    auto&& w_a = stdexec::when_all(sndr);
    ASSERT_NE(std::addressof(w_a), std::addressof(sndr)); // w_a is a new copy, not a reference to sndr.
    sndr = stdexec::just(56);
    const auto result = stdexec::sync_wait(std::move(w_a)); // NOLINT(performance-move-const-arg)
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(std::get<0>(*result), 42);
    ASSERT_EQ(
        std::get<0>(*stdexec::sync_wait(stdexec::when_all(std::move(sndr)))), 56); // NOLINT(performance-move-const-arg)

    // lvalues must be copyable; rvalues move.
    using just_not_copy_constructible_t = decltype(stdexec::just(std::make_unique<int>(0)));
    static_assert(!std::copy_constructible<just_not_copy_constructible_t>);
    static_assert(!std::invocable<const stdexec::when_all_t&, just_not_copy_constructible_t&>);
    static_assert(std::invocable<const stdexec::when_all_t&, just_not_copy_constructible_t&&>);
}

/**
 * @test A @c stdexec::when_all with a single branch on @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @note After the implementation of P4269R0 in https://github.com/NVIDIA/stdexec/pull/2124,
 *       @c when_all(sndr) with a single sender is expression-equivalent to @c auto(sndr). Hence, the sender
 *       returned by @c when_all(sndr) may have a completion scheduler. Notably, for an execution space completing
 *       sender @c sndr, @c when_all(sndr) returns an execution space completing sender.
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

    static_assert(std::same_as<stdexec::tag_of_t<decltype(sndr)>, stdexec::then_t>);
    static_assert(Kokkos::Execution::ExecutionSpaceImpl::execution_space_completing_sender<decltype(sndr)>);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr)),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 1);
}

/**
 * @test A @c stdexec::when_all with a schedule sender in one branch and a single other branch
 *       on @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) --------- \
 *                          when_all
 * schedule(esc) | then -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, schedule_sender_and_single_branch) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto sndr = stdexec::when_all(
        stdexec::schedule(esc.get_scheduler()), stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data));

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
    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")));
        }
    }());

    ASSERT_EQ(data(), 1);
}

/**
 * @test A @c stdexec::when_all with a schedule sender in one branch and a single other branch
 *       on @ref Kokkos::Execution::ExecutionSpaceContext, followed by work
 *       on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) --------- \
 *                          when_all --> continues_on(esc) | then
 * schedule(esc) | then -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, schedule_sender_and_single_branch_followed_by_self) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()),
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data))
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
 * @test A @c stdexec::when_all with a schedule sender in one branch and a single other branch
 *       with a segment on @ref Kokkos::Execution::ExecutionSpaceContext followed by a segment
 *       on another context. The @c stdexec::when_all is followed by work on the same
 *       @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(stc) ----------------------------------- \
 *                                                    when_all --> continues_on(esc) | then
 * schedule(esc) | then --> continues_on(stc) | then /
 * @endverbatim
 */
TEST_F(WhenAllTest, schedule_sender_and_single_mixed_branch_followed_by_self) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(stc.get_scheduler()),
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
 * @test A @c stdexec::when_all with a schedule sender in one branch and a single other branch
 *       on @ref Kokkos::Execution::ExecutionSpaceContext, followed by work on another context,
 *       followed by work on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) ------ \
 *                       when_all --> continues_on(stc) | then --> continues_on(esc) | then
 * schedule(esc) | then /
 * @endverbatim
 */
TEST_F(WhenAllTest, schedule_sender_and_single_branch_followed_by_other_and_finish_on_self) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()),
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT(data))
              | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT(data)
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        }
    }());

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

    if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
    } else {
        ASSERT_THAT(recorded_events, [&]() {
            if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec, "then")),
                    MATCHER_FOR_RECORD_EVENT(exec_B),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec, "sync_wait")));

            } else {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, dispatch_label(exec_B, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "after dispatch")),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait")));
            }
        }());
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

    if (Tests::Utils::are_same_instances(exec, exec_h)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));
    } else {
        ASSERT_THAT(recorded_events, [&]() {
            if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<host_execution_space>) {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                    MATCHER_FOR_RECORD_EVENT(exec_h),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
            } else {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_PFOR(exec_h, dispatch_label(exec_h, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec_h, dispatch_label(exec_h, "after dispatch")),
                    MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                    MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
            }
        }());
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
 * schedule(esc) | then_atomic ---------------------------------------------------------------------- \
 *                                                                                                     \
 *                                                                                                      when_all --> continues_on(esc) | then
 * schedule(esc) ------------- \                                                                       /
 *                              when_all --> continues_on(stc) | then_atomic --> continues_on(esc) -- /
 * schedule(esc) | then_atomic /
 * @endverbatim
 */
TEST_F(WhenAllTest, nested_with_inner_followed_by_other) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data),
                    stdexec::when_all(
                        stdexec::schedule(esc.get_scheduler()),
                        stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data))
                        | Tests::Utils::check_rcvr_env_not_queryable_with<Kokkos::Execution::Impl::get_exec_t>()
                        | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(System, data)
                        | stdexec::continues_on(esc.get_scheduler()))
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(sndr)

    const auto recorded_events = Tests::Utils::record_sync_wait<recorder_listener_t>(std::move(sndr));

    ASSERT_THAT(recorded_events, [&]() {
        if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_RECORD_EVENT(exec),
                MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        } else {
            return testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "after dispatch")),
                MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait")));
        }
    }());

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

    if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "after dispatch")),
                MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                MATCHER_FOR_BEGIN_FENCE(exec_C, dispatch_label(exec_C, "after dispatch"))));
    } else {
        ASSERT_THAT(recorded_events, [&]() {
            if constexpr (Kokkos::Execution::Impl::has_non_blocking_dispatch<TEST_EXECUTION_SPACE>) {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                    MATCHER_FOR_RECORD_EVENT(exec_B),
                    MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                    MATCHER_FOR_RECORD_EVENT(exec_C),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                    MATCHER_FOR_RECORD_EVENT(exec_A),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(4)),
                    MATCHER_FOR_WAIT_EVENT(recorded_events.at(7)));
            } else {
                return testing::ElementsAre(
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                    MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                    MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "after dispatch")),
                    MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                    MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "after dispatch")),
                    MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                    MATCHER_FOR_BEGIN_FENCE(exec_C, dispatch_label(exec_C, "after dispatch")));
            }
        }());
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
