#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"

#include "tests/utils/check_completion_scheduler_type.hpp"
#include "tests/utils/functors/store_thread_id.hpp"
#include "tests/utils/stdexec.hpp"

namespace Tests::Utils {

//! @test Check whether the sender can be nothrow applied for @c stdexec::sync_wait.
template <typename Domain, stdexec::sender Sndr>
consteval bool check_nothrow_apply_sender() {
    static_assert(stdexec::__never_sends<stdexec::set_error_t, Sndr>);
    static_assert(Tests::Utils::has_nothrow_apply_sender<Domain, stdexec::sync_wait_t, Sndr>);
    static_assert(!Tests::Utils::has_nothrow_apply_sender<stdexec::sync_wait_t, Sndr>);
    return true;
}

template <typename ContextType>
struct SyncWaitTest
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<typename ContextType::execution_space> {
    using execution_space = typename ContextType::execution_space;

    const ContextType ctx{this->exec};
};

TYPED_TEST_SUITE_P(SyncWaitTest);

/**
 * @test Check that the start scheduler that @c stdexec::sync_wait sets in the receiver environment
 *       is a @c run_loop scheduler on the thread on which it starts the operation state.
 */
TYPED_TEST_P(SyncWaitTest, get_start_scheduler) {
    std::thread::id tid;

    auto sndr = stdexec::read_env(stdexec::get_start_scheduler)
              | stdexec::let_value([&](auto schd) { return stdexec::schedule(schd) | THEN_STORE_THREAD_ID(&tid); })
              | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, stdexec::run_loop::scheduler>()
              | stdexec::continues_on(this->ctx.get_scheduler());

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, std::this_thread::get_id());
}

/**
 * @test Check that the completion-scheduler query for inline work resolves to
 *       the receiver's start scheduler, and that the work runs on the calling thread.
 */
TYPED_TEST_P(SyncWaitTest, just) {
    std::thread::id tid;

    auto sndr = stdexec::just() | THEN_STORE_THREAD_ID(&tid)
              | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, stdexec::run_loop::scheduler>()
              | stdexec::continues_on(this->ctx.get_scheduler());

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, std::this_thread::get_id());
}

REGISTER_TYPED_TEST_SUITE_P(SyncWaitTest, get_start_scheduler, just);

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP
