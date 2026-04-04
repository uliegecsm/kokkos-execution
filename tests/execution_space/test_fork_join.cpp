#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/fork_join.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/check_scheduler_type.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/increment.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/functors/sum_indices.hpp"
#include "tests/utils/kokkos.hpp"

/**
 * @addtogroup unittests
 *
 * Use @c experimental::execution::fork_join with @c Kokkos::Execution::ExecutionSpaceContext
 * ------------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext properly works
 * with @c experimental::execution::fork_join.
 *
 * The tests can be found in @ref tests/execution_space/test_fork_join.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class ForkJoinTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<EventDiscardMatcher<TEST_EXECUTION_SPACE>, BeginFenceEvent, BeginParallelForEvent>;

    static constexpr bool on_device = Tests::Utils::on_device<TEST_EXECUTION_SPACE>();
};

//! @test Use @c experimental::execution::fork_join to produce a diamond-like topology.
TEST_F(ForkJoinTest, diamond) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    experimental::execution::static_thread_pool pool{4};
    const context_t esc{exec};

    auto chain =
        stdexec::schedule(pool.get_scheduler())
        | stdexec::then(Tests::Utils::Functors::LoadCheckAdd<int, false>{.prev = 0, .value = 4, .data = data.data()})
        | experimental::execution::fork_join(
            stdexec::continues_on(esc.get_scheduler())
                | Tests::Utils::check_scheduler_type<stdexec::set_value_t, scheduler_t>() | THEN_INCREMENT_ATOMIC(data),
            stdexec::continues_on(pool.get_scheduler()) | THEN_INCREMENT_ATOMIC(data))
        | stdexec::continues_on(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<int, on_device>{.prev = 6, .value = 3, .data = data.data()})
        | stdexec::continues_on(stdexec::inline_scheduler{})
        | stdexec::then(Tests::Utils::Functors::LoadCheckAdd<int, false>{.prev = 9, .value = 5, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(chain)

    ASSERT_THAT(
        recorder_listener_t::record([chain = std::move(chain)]() mutable { stdexec::sync_wait(std::move(chain)); }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 14);
}

/**
 * @test Use @c experimental::execution::fork_join after a @c stdexec::continues_on.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/issues/1823.
 */
TEST_F(ForkJoinTest, continues_on) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto sndr =
        stdexec::just() | stdexec::continues_on(esc.get_scheduler())
        | experimental::execution::fork_join(
            stdexec::continues_on(esc.get_scheduler())
            | stdexec::then(
                Tests::Utils::Functors::LoadCheckAdd<int, on_device>{.prev = 0, .value = 3, .data = data.data()}));

    static_assert(stdexec::__sender_for<decltype(sndr), experimental::execution::fork_join_t>);

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(sndr));                         // NOLINT(performance-move-const-arg)
        }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

/**
 * @test Use @c experimental::execution::fork_join after a @c stdexec::continues_on and a @c stdexec::bulk.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/issues/1823.
 */
TEST_F(ForkJoinTest, continues_on_bulk) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    auto sndr =
        stdexec::just() | stdexec::continues_on(esc.get_scheduler()) | BULK_SUM_INDICES(2, data)
        | experimental::execution::fork_join(
            stdexec::continues_on(esc.get_scheduler())
            | stdexec::then(
                Tests::Utils::Functors::LoadCheckAdd<int, on_device>{.prev = 1, .value = 2, .data = data.data()}));

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    ASSERT_THAT(
        recorder_listener_t::record([sndr = std::move(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(sndr));                         // NOLINT(performance-move-const-arg)
        }),
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "bulk")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 3);
}

} // namespace Tests::ExecutionSpaceImpl
