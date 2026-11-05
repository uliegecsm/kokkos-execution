#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP

#include "gtest/gtest.h"

#include "Kokkos_Core.hpp"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph/events.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/category.hpp"
#include "tests/utils/check_completion_scheduler_type.hpp"
#include "tests/utils/functors/store_thread_id.hpp"
#include "tests/utils/functors/throws_when_copied.hpp"
#include "tests/utils/schedulers/stopped.hpp"
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
struct TEST_CATEGORY(SyncWaitTest)
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<typename ContextType::execution_space>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
    using execution_space = typename ContextType::execution_space;

    using recorder_listener_t = Kokkos::utils::callbacks::RecorderListener<
        EventDiscardMatcher<execution_space>,
        Kokkos::utils::callbacks::BeginFenceEvent,
        Kokkos::utils::callbacks::BeginSingleEvent,
        Kokkos::utils::callbacks::BeginParallelForEvent,
        Kokkos::utils::callbacks::BeginParallelReduceEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::GraphImpl::GraphCreateEvent
    >;

    const ContextType ctx{this->exec};
};

TYPED_TEST_SUITE_P(TEST_CATEGORY(SyncWaitTest));

/**
 * @test Check that the start scheduler that @c stdexec::sync_wait sets in the receiver environment
 *       is a @c run_loop scheduler on the thread on which it starts the operation state.
 */
TYPED_TEST_P(TEST_CATEGORY(SyncWaitTest), get_start_scheduler) {
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
TYPED_TEST_P(TEST_CATEGORY(SyncWaitTest), just) {
    std::thread::id tid;

    auto sndr = stdexec::just() | THEN_STORE_THREAD_ID(&tid)
              | Tests::Utils::check_completion_scheduler_type<stdexec::set_value_t, stdexec::run_loop::scheduler>()
              | stdexec::continues_on(this->ctx.get_scheduler());

    stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)

    ASSERT_EQ(tid, std::this_thread::get_id());
}

//! @test Check that calling @c stdexec::sync_wait on a sender that does not have any operation in it will not result in a spurious fence.
TYPED_TEST_P(TEST_CATEGORY(SyncWaitTest), set_value_no_spurious_fence) {
    auto sndr = stdexec::schedule(this->ctx.get_scheduler());

    ASSERT_THAT(
        TestFixture::recorder_listener_t::record(
            [sndr = std::move(sndr)]() mutable {                        // NOLINT(performance-move-const-arg)
                const auto value = stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)
                static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
                ASSERT_TRUE(value.has_value());
            }),
        testing::IsEmpty());
}

//! @test Check that @c stdexec::sync_wait does not induce a fence when there is no outstanding operation.
TYPED_TEST_P(TEST_CATEGORY(SyncWaitTest), set_stopped_no_spurious_fence) {
    auto sndr = stdexec::schedule(Tests::Utils::Schedulers::Stopped{})
              | stdexec::continues_on(this->ctx.get_scheduler());

    ASSERT_THAT(
        TestFixture::recorder_listener_t::record(
            [sndr = std::move(sndr)]() mutable {                        // NOLINT(performance-move-const-arg)
                const auto value = stdexec::sync_wait(std::move(sndr)); // NOLINT(performance-move-const-arg)
                static_assert(std::same_as<decltype(value), const std::optional<std::tuple<>>>);
                ASSERT_FALSE(value.has_value());
            }),
        testing::IsEmpty());
}

//! @test Check that @c stdexec::sync_wait properly rethrows if needed.
TYPED_TEST_P(TEST_CATEGORY(SyncWaitTest), rethrows) {
    auto sndr = stdexec::schedule(this->ctx.get_scheduler())
              | stdexec::then(Tests::Utils::Functors::ThrowsWhenCopied{});

    ASSERT_THAT(
        Tests::Utils::Functors::MutableMoveToSyncWait{.sndr = std::move(sndr)},
        testing::ThrowsMessage<std::runtime_error>(testing::StrEq("ThrowsWhenCopied: Throwing in copy constructor!")));
}

REGISTER_TYPED_TEST_SUITE_P(
    TEST_CATEGORY(SyncWaitTest),
    get_start_scheduler,
    just,
    set_value_no_spurious_fence,
    set_stopped_no_spurious_fence,
    rethrows);

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SYNC_WAIT_HPP
