#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/split.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/check_completion_scheduler_type.hpp"
#include "tests/utils/functors/show_thread_id.hpp"
#include "tests/utils/functors/store_thread_id.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Tests::Utils::check_completion_scheduler_type
 * ----------------------------------------------------------
 *
 * This group of tests check the behavior of @ref Tests::Utils::check_completion_scheduler_type.
 *
 * The tests can be found in @ref tests/utils/test_check_completion_scheduler_type.cpp.
 */

namespace Tests {

using run_loop_scheduler_t = stdexec::run_loop::scheduler;
using static_thread_pool_scheduler_t = experimental::execution::_pool_::_static_thread_pool::scheduler;

//! @test Start scheduler.
TEST(check_completion_scheduler, default) {
    std::thread::id tid;

    stdexec::sync_wait(
        stdexec::just() | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>()
        | THEN_STORE_THREAD_ID(&tid));

    ASSERT_EQ(tid, std::this_thread::get_id());
}

//! @test @c experimental::execution::static_thread_pool scheduler.
TEST(check_completion_scheduler, static_thread_pool) {
    std::thread::id tid;

    experimental::execution::static_thread_pool pool{1};

    auto chain = stdexec::schedule(pool.get_scheduler())
               | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
               | THEN_STORE_THREAD_ID(&tid);

    stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_NE(tid, std::this_thread::get_id());
}

} // namespace Tests
