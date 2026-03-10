#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include "exec/split.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/check_scheduler_type.hpp"
#include "tests/utils/functors/show_thread_id.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c Tests::Utils::check_scheduler_type
 * -----------------------------------------------
 *
 * This group of tests check the behavior of @ref Tests::Utils::check_scheduler_type.
 *
 * The tests can be found in @ref tests/utils/test_check_scheduler_type.cpp.
 */

namespace Tests {

using run_loop_scheduler_t = stdexec::run_loop::scheduler;
using static_thread_pool_scheduler_t = experimental::execution::_pool_::_static_thread_pool::scheduler;

//! @test Default scheduler.
TEST(check_scheduler, default) {
    stdexec::sync_wait(
        stdexec::just() | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>()
        | THEN_SHOW_THREAD_ID);
}

//! @test @c experimental::execution::static_thread_pool scheduler.
TEST(check_scheduler, static_thread_pool) {
    experimental::execution::static_thread_pool pool{1};

    auto chain = stdexec::schedule(pool.get_scheduler())
               | Tests::Utils::check_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
               | THEN_SHOW_THREAD_ID;

    stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
}

/**
 * @test @c stdexec::when_all does not forward the scheduler to its input senders.
 *
 * See also https://github.com/NVIDIA/stdexec/issues/1736#issuecomment-3720622409.
 */
TEST(check_scheduler, split_when_all_no_forward) {
    experimental::execution::static_thread_pool pool{1};

    auto fork = stdexec::schedule(pool.get_scheduler())
              | Tests::Utils::check_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
              | THEN_SHOW_THREAD_ID | experimental::execution::split();

    auto chain = stdexec::when_all(
        fork | THEN_SHOW_THREAD_ID | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>(),
        fork | THEN_SHOW_THREAD_ID | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>());

    stdexec::sync_wait(
        std::move(chain) | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>());
}

/**
 * @test @c stdexec::transfer_when_all does not forward the scheduler to its input senders.
 *
 * See also https://github.com/NVIDIA/stdexec/issues/1736#issuecomment-3720622409.
 */
TEST(check_scheduler, split_transfer_when_all_no_forward) {
    ::exec::static_thread_pool pool{1};

    auto fork = stdexec::schedule(pool.get_scheduler())
              | Tests::Utils::check_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
              | THEN_SHOW_THREAD_ID | experimental::execution::split();

    auto chain = stdexec::transfer_when_all(
        pool.get_scheduler(),
        fork | THEN_SHOW_THREAD_ID | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>(),
        fork | THEN_SHOW_THREAD_ID | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>());

    stdexec::sync_wait(
        std::move(chain) | Tests::Utils::check_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>());
}

//! @test Multiple splits.
TEST(check_scheduler, multiple_splits) {
    ::exec::static_thread_pool pool{1};

    auto fork_A = stdexec::schedule(pool.get_scheduler())
                | Tests::Utils::check_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
                | THEN_SHOW_THREAD_ID | experimental::execution::split();

    auto chain_A_branch_a = fork_A | THEN_SHOW_THREAD_ID;
    auto chain_A_branch_b = std::move(fork_A) | THEN_SHOW_THREAD_ID;

    auto chain_A = stdexec::when_all(std::move(chain_A_branch_a), std::move(chain_A_branch_b)) | THEN_SHOW_THREAD_ID;

    auto fork_B = std::move(chain_A) | stdexec::continues_on(pool.get_scheduler())
                | Tests::Utils::check_scheduler_type<stdexec::set_value_t, static_thread_pool_scheduler_t>()
                | experimental::execution::split();

    auto chain_B_branch_a = fork_B | THEN_SHOW_THREAD_ID
                          | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>();
    auto chain_B_branch_b = std::move(fork_B) | THEN_SHOW_THREAD_ID
                          | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>();

    auto chain_B = stdexec::when_all(std::move(chain_B_branch_a), std::move(chain_B_branch_b)) | THEN_SHOW_THREAD_ID
                 | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>();

    stdexec::sync_wait(
        std::move(chain_B) | Tests::Utils::check_scheduler_type<stdexec::set_value_t, run_loop_scheduler_t>());
}

} // namespace Tests
