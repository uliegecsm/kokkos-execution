#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/single_thread_context.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_rcvr_env_queryable_with.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/labeled.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/kokkos.hpp"
#include "tests/utils/stdexec.hpp"

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

using execution_space = Kokkos::DefaultExecutionSpace;

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class WhenAllTest
    : public Tests::Utils::ExecutionSpaceContextTest<execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
};

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

    /// Because the sender returned by @c stdexec::when_all is not an execution space completing sender,
    /// the default implementation of @c stdexec::sync_wait is used.
    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation"))));

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
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); }),
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

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); }),
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

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test A @c stdexec::when_all with two branches, the first on @ref Kokkos::Execution::ExecutionSpaceContext and the second
 *       on another context, followed by work on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then_atomic -- \
 *                                 when_all --> continues_on(esc) | then
 * schedule(stc) | then_atomic -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, two_mixed_branches_followed_by_self) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto w_a = stdexec::when_all(
        stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data),
        stdexec::schedule(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data));

    static_assert(
        std::same_as<
            decltype(stdexec::get_completion_domain<stdexec::set_value_t>(stdexec::get_env(w_a), stdexec::env<>{})),
            stdexec::default_domain
        >);

    auto sndr = std::move(w_a) | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); }),
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
                    stdexec::schedule(esc_A.get_scheduler()) | THEN_INCREMENT_ATOMIC(data),
                    stdexec::schedule(esc_B.get_scheduler()) | THEN_INCREMENT_ATOMIC(data))
              | stdexec::continues_on(esc_A.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); });

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
                MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "continuation")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, dispatch_label(exec_A, "then")),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "sync_wait"))));
    }

    ASSERT_EQ(data(), 3);
}

/**
 * @test A @c stdexec::when_all with two branches, the first on @ref Kokkos::Execution::ExecutionSpaceContext
 *       and the second on another context, followed by work on the same other context, followed by work
 *       on the same @ref Kokkos::Execution::ExecutionSpaceContext.
 *
 * @verbatim
 * schedule(esc) | then_atomic -- \
 *                                 when_all --> continues_on(stc) | then --> continues_on(esc) | then
 * schedule(stc) | then_atomic -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, two_mixed_branches_followed_by_other_and_finish_on_self) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr = stdexec::when_all(
                    stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data),
                    stdexec::schedule(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data))
              | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT(data)
              | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation")),
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
 * schedule(esc) | then_atomic -------------------------------------------------- \
 *                                                                                 when_all --> continues_on(esc) | then
 * schedule(esc) | then_atomic -- when_all --> continues_on(stc) | then_atomic -- /
 * @endverbatim
 */
TEST_F(WhenAllTest, nested_with_inner_followed_by_other) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};
    experimental::execution::single_thread_context stc{};

    auto sndr =
        stdexec::when_all(
            stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data),
            stdexec::when_all(stdexec::schedule(esc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data))
                | Tests::Utils::check_rcvr_env_queryable_with<Kokkos::Execution::ExecutionSpaceImpl::get_exec_t>()
                | stdexec::continues_on(stc.get_scheduler()) | THEN_INCREMENT_ATOMIC(data))
        | stdexec::continues_on(esc.get_scheduler()) | THEN_INCREMENT(data);

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            // MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "continuation")),
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
 *
 * @todo C cannot currently overlap with either of A, B or D.
 */
TEST_F(WhenAllTest, nested_when_all_with_independent_branch) {
    const auto [exec_A, exec_B, exec_C] = Kokkos::Experimental::partition_space(exec, 1, 1, 1);

    const context_t esc_A{exec_A}, esc_B{exec_B}, esc_C{exec_C};

    auto br_A = ::stdexec::schedule(esc_A.get_scheduler()) | THEN_LABELED_PFOR('A');
    auto br_B = ::stdexec::schedule(esc_B.get_scheduler()) | THEN_LABELED_PFOR('B');
    auto br_C = ::stdexec::schedule(esc_C.get_scheduler()) | THEN_LABELED_PFOR('C');

    auto when_AB_then_D = ::stdexec::when_all(std::move(br_A), std::move(br_B))
                        | ::stdexec::continues_on(esc_A.get_scheduler()) | THEN_LABELED_PFOR('D');

    auto sndr = ::stdexec::when_all(std::move(when_AB_then_D), std::move(br_C));

    const auto recorded_events = recorder_listener_t::record(
        [sndr = std::move(sndr)]() mutable { stdexec::sync_wait(std::move(sndr)); });

    if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "continuation")),
                MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                MATCHER_FOR_BEGIN_FENCE(exec_C, dispatch_label(exec_C, "continuation"))));
    } else {
        ASSERT_THAT(
            recorded_events,
            testing::ElementsAre(
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'A'"),
                MATCHER_FOR_BEGIN_PFOR(exec_B, "'B'"),
                MATCHER_FOR_BEGIN_FENCE(exec_B, dispatch_label(exec_B, "continuation")),
                MATCHER_FOR_BEGIN_PFOR(exec_A, "'D'"),
                MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "continuation")),
                MATCHER_FOR_BEGIN_PFOR(exec_C, "'C'"),
                MATCHER_FOR_BEGIN_FENCE(exec_C, dispatch_label(exec_C, "continuation"))));
    }
}

} // namespace Tests::ExecutionSpaceImpl
