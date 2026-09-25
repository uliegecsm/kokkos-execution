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
TEST(check_scheduler, default) {
    std::thread::id tid;

    stdexec::sync_wait(
        stdexec::just() | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>()
        | THEN_STORE_THREAD_ID(&tid));

    ASSERT_EQ(tid, std::this_thread::get_id());
}

//! @test @c experimental::execution::static_thread_pool scheduler.
TEST(check_scheduler, static_thread_pool) {
    std::thread::id tid;

    experimental::execution::static_thread_pool pool{1};

    auto chain = stdexec::schedule(pool.get_scheduler())
               | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
               | THEN_STORE_THREAD_ID(&tid);

    stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)

    ASSERT_NE(tid, std::this_thread::get_id());
}

/**
 * @todo Diagnose what queries may advertise in such scenarios with @c stdexec::when_all
 *       and @c experimental::execution::split.
 *
 * See also https://github.com/NVIDIA/stdexec/issues/1736#issuecomment-3720622409.
 */
TEST(check_scheduler, split_when_all_no_forward) {
    experimental::execution::static_thread_pool pool{1};

    auto fork = stdexec::schedule(pool.get_scheduler())
              | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
              | THEN_SHOW_THREAD_ID | experimental::execution::split();

    static_assert(!Tests::Utils::has_completion_scheduler_for<
                  decltype(fork),
                  stdexec::set_value_t,
                  stdexec::prop<stdexec::get_start_scheduler_t, stdexec::run_loop::scheduler>
    >);

    auto chain = stdexec::when_all(fork | THEN_SHOW_THREAD_ID, fork | THEN_SHOW_THREAD_ID);

    stdexec::sync_wait(std::move(chain));
}

//! @todo Scenario with @c stdexec::transfer_when_all.
TEST(check_scheduler, split_transfer_when_all_no_forward) {
    ::exec::static_thread_pool pool{1};

    auto fork = stdexec::schedule(pool.get_scheduler()) | THEN_SHOW_THREAD_ID | experimental::execution::split();

    auto chain =
        stdexec::transfer_when_all(pool.get_scheduler(), fork | THEN_SHOW_THREAD_ID, fork | THEN_SHOW_THREAD_ID);

    stdexec::sync_wait(
        std::move(chain)
        | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>());
}

//! @todo Scenario with multiple splits.
TEST(check_scheduler, multiple_splits) {
    ::exec::static_thread_pool pool{1};

    auto fork_A = stdexec::schedule(pool.get_scheduler()) | THEN_SHOW_THREAD_ID | experimental::execution::split();

    auto chain_A_branch_a = fork_A | THEN_SHOW_THREAD_ID;
    auto chain_A_branch_b = std::move(fork_A) | THEN_SHOW_THREAD_ID;

    auto chain_A = stdexec::when_all(std::move(chain_A_branch_a), std::move(chain_A_branch_b)) | THEN_SHOW_THREAD_ID;

    auto fork_B = std::move(chain_A) | stdexec::continues_on(pool.get_scheduler()) | experimental::execution::split();

    auto chain_B_branch_a = fork_B | THEN_SHOW_THREAD_ID;
    auto chain_B_branch_b = std::move(fork_B) | THEN_SHOW_THREAD_ID;

    auto chain_B = stdexec::when_all(std::move(chain_B_branch_a), std::move(chain_B_branch_b)) | THEN_SHOW_THREAD_ID;

    stdexec::sync_wait(std::move(chain_B));
}

} // namespace Tests
