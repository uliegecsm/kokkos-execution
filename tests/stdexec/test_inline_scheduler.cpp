#include "gtest/gtest.h"

#include <thread>

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/completion_behavior.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/store_thread_id.hpp"
#include "tests/utils/return_thread_id.hpp"
#include "tests/utils/stdexec.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c stdexec::inline_scheduler
 * --------------------------------------
 *
 * This group of tests check the behavior of @c stdexec::inline_scheduler.
 *
 * The tests can be found in @ref tests/stdexec/test_inline_scheduler.cpp.
 */

namespace Tests {

//! @test Check that the completion behavior of @c stdexec::inline_scheduler is @c exec::completion_behavior::inline_completion.
consteval bool test_completion_behavior() {
    static_assert(
        exec::get_completion_behavior<stdexec::set_value_t, stdexec::schedule_result_t<stdexec::inline_scheduler>>()
        == exec::completion_behavior::inline_completion);

    return true;
}

static_assert(test_completion_behavior());

/**
 * @test The schedule sender of @c stdexec::inline_scheduler has no completion scheduler.
 *
 * However, if the environment provides a @c stdexec::get_start_scheduler_t, it is served as the completion scheduler.
 *
 * See also https://github.com/NVIDIA/stdexec/blob/4781797199f20ef742b39ea00785a1503badd218/include/stdexec/__detail/__inline_scheduler.hpp#L37-L45.
 */
consteval bool test_has_no_completion_scheduler() {
    using sndr_t = stdexec::schedule_result_t<stdexec::inline_scheduler>;

    static_assert(stdexec::__sends<stdexec::set_value_t, sndr_t, stdexec::env<>>);

    using attrs_t = stdexec::env_of_t<sndr_t>;

    static_assert(std::same_as<attrs_t, stdexec::__inline_attrs>);

    static_assert(!Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_value_t, stdexec::env<>>);

    using start_scheduler_env_t = stdexec::prop<stdexec::get_start_scheduler_t, stdexec::run_loop::scheduler>;

    static_assert(Tests::Utils::has_completion_scheduler_for<sndr_t, stdexec::set_value_t, start_scheduler_env_t>);

    return true;
}

static_assert(test_has_no_completion_scheduler());

//! @test It uses the main thread when nothing else is provided.
TEST(inline_scheduler, main_thread) {
    std::thread::id tid;

    stdexec::sync_wait(stdexec::schedule(stdexec::inline_scheduler{}) | THEN_STORE_THREAD_ID(&tid));

    ASSERT_EQ(tid, std::this_thread::get_id());
}

//! @test It uses the previous scheduler thread when using @c stdexec::continues_on.
TEST(inline_scheduler, pool_thread) {
    experimental::execution::static_thread_pool pool{1};

    const auto [tid_pool] = stdexec::sync_wait(stdexec::schedule(pool.get_scheduler()) | THEN_RETURN_THREAD_ID).value();

    ASSERT_NE(tid_pool, std::this_thread::get_id());

    std::thread::id tid;

    stdexec::sync_wait(
        stdexec::schedule(pool.get_scheduler()) | stdexec::continues_on(stdexec::inline_scheduler{})
        | THEN_STORE_THREAD_ID(&tid));

    ASSERT_EQ(tid, tid_pool);
}

} // namespace Tests
