#include "gtest/gtest.h"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/sequence.hpp"
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/functors/store_thread_id.hpp"
#include "tests/utils/probe_opstate_start_thread.hpp"

/**
 * @addtogroup unittests
 *
 * Tests for @c exec::sequence
 * ---------------------------
 *
 * This group of tests check the behavior of @c exec::sequence.
 *
 * See also @cite P4320R1.
 *
 * The tests can be found in @ref tests/stdexec/test_sequence.cpp.
 */

namespace Tests {

//! @test The second sender to @c exec::sequence has no scheduler, check it runs on the previous sender's scheduler.
TEST(sequence, second_has_no_scheduler) {
    exec::static_thread_pool pool{1};

    std::thread::id tid_A, tid_B;

    stdexec::sync_wait(
        exec::sequence(
            stdexec::schedule(pool.get_scheduler()) | THEN_STORE_THREAD_ID(&tid_A),
            stdexec::just() | THEN_STORE_THREAD_ID(&tid_B)));

    ASSERT_EQ(tid_A, tid_B);
    ASSERT_NE(tid_A, std::this_thread::get_id());
}

//! @test The second sender to @c exec::sequence starts with @c stdexec::schedule, and thus uses the proper scheduler.
TEST(sequence, second_has_scheduler) {
    exec::static_thread_pool pool_A{1}, pool_B{1};

    std::thread::id tid_A, tid_B;

    stdexec::sync_wait(
        exec::sequence(
            stdexec::schedule(pool_A.get_scheduler()) | THEN_STORE_THREAD_ID(&tid_A),
            stdexec::schedule(pool_B.get_scheduler()) | THEN_STORE_THREAD_ID(&tid_B)));

    ASSERT_NE(tid_A, tid_B);
    ASSERT_NE(tid_A, std::this_thread::get_id());
    ASSERT_NE(tid_B, std::this_thread::get_id());
}

/**
 * @test Check that the thread on which the second sender to @c exec::sequence operation state is @c start() ed
 *       is the same thread as the one on which the first sender ran, and is not the main thread.
 */
TEST(sequence, second_probe_where_its_opstate_is_started) {
    exec::static_thread_pool pool{1};

    std::thread::id tid_A, tid_start_opstate_B;

    stdexec::sync_wait(
        exec::sequence(
            stdexec::schedule(pool.get_scheduler()) | THEN_STORE_THREAD_ID(&tid_A),
            Tests::Utils::ProbeOpStateStartThreadSender{&tid_start_opstate_B}));

    ASSERT_EQ(tid_A, tid_start_opstate_B);
    ASSERT_NE(tid_A, std::this_thread::get_id());
}

//! @test The second sender to @c exec::sequence has the first sender's scheduler as the @c stdexec::get_start_scheduler query result.
TEST(sequence, t) {
    exec::static_thread_pool pool{1};

    std::thread::id tid_A{};

    auto schd = pool.get_scheduler();

    stdexec::sender auto second =
        stdexec::read_env(stdexec::get_start_scheduler) | stdexec::let_value([&](auto schd_from_env) {
            static_assert(
                std::same_as<decltype(schd_from_env), experimental::execution::_pool_::_static_thread_pool::scheduler>);
            return stdexec::just(schd_from_env == schd);
        });

    const auto [res] = stdexec::sync_wait(
                           exec::sequence(stdexec::schedule(schd) | THEN_STORE_THREAD_ID(&tid_A), std::move(second)))
                           .value();

    ASSERT_NE(tid_A, std::this_thread::get_id());
    ASSERT_TRUE(res);
}

} // namespace Tests
