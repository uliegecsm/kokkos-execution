#include "gtest/gtest.h"

#include <thread>

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/store_thread_id.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::get_start_scheduler
 * -----------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::get_start_scheduler.
 *
 * The tests can be found in @ref tests/stdexec/test_get_start_scheduler.cpp.
 */

namespace Tests {

consteval bool test_is_forwarding_query() {
    return stdexec::forwarding_query(stdexec::get_start_scheduler);
}
static_assert(test_is_forwarding_query());

/**
 * @test Check that the start scheduler that @c stdexec::sync_wait sets in the receiver environment
 *       is a @c run_loop scheduler on the thread on which it starts the operation state.
 */
TEST(get_start_scheduler, sync_wait) {
    std::thread::id tid;

    auto sndr = stdexec::read_env(stdexec::get_start_scheduler) | stdexec::let_value([&](auto schd) {
                    static_assert(std::same_as<decltype(schd), stdexec::run_loop::scheduler>);
                    return stdexec::schedule(schd) | THEN_STORE_THREAD_ID(&tid);
                });

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, std::this_thread::get_id());
}

/**
 * @test Check that the start scheduler that @c stdexec::let_value sets in the receiver environment
 *       of the sender returned by the closure is the completion scheduler of the predecessor.
 *
 * Indeed, @c stdexec::let_value starts the successor from the completion of the predecessor.
 *
 * See also:
 * - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__let.hpp#L180
 * - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__schedulers.hpp#L639-L655
 */
TEST(get_start_scheduler, let_value) {
    std::thread::id pool_tid, tid;

    experimental::execution::static_thread_pool pool{1};

    auto sndr = stdexec::schedule(pool.get_scheduler()) | THEN_STORE_THREAD_ID(&pool_tid) | stdexec::let_value([&]() {
                    return stdexec::read_env(stdexec::get_start_scheduler) | stdexec::let_value([&](auto schd) {
                               static_assert(std::same_as<decltype(schd), decltype(pool.get_scheduler())>);
                               return stdexec::schedule(schd) | THEN_STORE_THREAD_ID(&tid);
                           });
                });

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, pool_tid);
    ASSERT_NE(tid, std::this_thread::get_id());
}

/**
 * @test Show that @c stdexec::continues_on onto an inline scheduler behaves as expected at run time. However, the compile-time
 *       scheduler queries are not correct: they indicate a hop onto the the start scheduler set in the outer receiver environment
 *       rather than continuing on the completion scheduler of the predecessor.
 *
 * Indeed, although @c stdexec::continues_on starts an operation state from the completion of the predecessor, it does not set the start scheduler
 * accordingly through a secondary environment.
 *
 * See also:
 * - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__continues_on.hpp#L193
 * - https://github.com/NVIDIA/stdexec/blob/5f94dbac91de3c4869fe695b7fe4d0ed66c0612d/include/stdexec/__detail/__continues_on.hpp#L127
 * - https://github.com/NVIDIA/stdexec/issues/2268
 */
TEST(get_start_scheduler, continues_on) {
    std::thread::id pool_tid, tid;

    experimental::execution::static_thread_pool pool{1};

    auto sndr = stdexec::schedule(pool.get_scheduler()) | THEN_STORE_THREAD_ID(&pool_tid)
              | stdexec::continues_on(stdexec::inline_scheduler{}) | THEN_STORE_THREAD_ID(&tid);

    static_assert(
        std::same_as<
            stdexec::__completion_scheduler_of_t<
                stdexec::set_value_t,
                decltype(sndr),
                stdexec::prop<stdexec::get_start_scheduler_t, stdexec::run_loop::scheduler>
            >,
            stdexec::run_loop::scheduler // does not match run-time behavior; should be decltype(pool.get_scheduler())
        >);

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, pool_tid); // run-time behavior is as expected
    ASSERT_NE(tid, std::this_thread::get_id());
}

} // namespace Tests
