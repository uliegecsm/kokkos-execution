#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/static_thread_pool.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/load_check_add.hpp"
#include "tests/utils/kokkos.hpp"

/**
 * @addtogroup unittests
 *
 * Interoperability of @c Kokkos::Execution::ExecutionSpaceContext with other schedulers
 * -------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Execution::ExecutionSpaceContext can be used in
 * conjunction with other schedulers like @c experimental::execution::static_thread_pool.
 *
 * The tests can be found in @ref tests/execution_space/test_inter_op.cpp.
 */

namespace Tests::ExecutionSpaceImpl {

using namespace Kokkos::utils::callbacks;

class InterOpTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t =
        RecorderListener<EventDiscardMatcher<TEST_EXECUTION_SPACE>, BeginFenceEvent, BeginParallelForEvent>;

    static constexpr bool on_device = Tests::Utils::on_device<TEST_EXECUTION_SPACE>();
};

//! @test Transition from @ref Kokkos::Execution::ExecutionSpaceContext to @c stdexec::inline_scheduler.
TEST_F(InterOpTest, transition_to_inline_scheduler) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    Tests::Utils::show_exec_space_id(exec, "exec");

    auto chain =
        stdexec::schedule(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | stdexec::continues_on(stdexec::inline_scheduler{})
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable {    // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
        });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 8);
}

//! @test Transition from @c stdexec::inline_scheduler to @ref Kokkos::Execution::ExecutionSpaceContext.
TEST_F(InterOpTest, transition_from_inline_scheduler) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    Tests::Utils::show_exec_space_id(exec, "exec");

    auto chain =
        stdexec::schedule(stdexec::inline_scheduler{})
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 0, .value = 4, .data = data.data()})
        | stdexec::continues_on(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable {    // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
        });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 8);
}

//! @test Transition from @c stdexec::inline_scheduler to @ref Kokkos::Execution::ExecutionSpaceContext and back.
TEST_F(InterOpTest, transition_from_inline_scheduler_and_back) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    Tests::Utils::show_exec_space_id(exec, "exec");

    auto chain =
        stdexec::schedule(stdexec::inline_scheduler{})
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 0, .value = 4, .data = data.data()})
        | stdexec::continues_on(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 4, .value = 4, .data = data.data()})
        | stdexec::continues_on(stdexec::inline_scheduler{})
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 8, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable {    // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
        });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 12);
}

//! @test Transition from @ref Kokkos::Execution::ExecutionSpaceContext to @c experimental::execution::static_thread_pool.
TEST_F(InterOpTest, transition_to_static_thread_pool) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    experimental::execution::static_thread_pool pool{1};

    Tests::Utils::show_exec_space_id(exec, "exec");

    auto chain =
        stdexec::schedule(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | stdexec::continues_on(pool.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable {    // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
        });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 8);
}

//! @test Transition from @c experimental::execution::static_thread_pool to @ref Kokkos::Execution::ExecutionSpaceContext.
TEST_F(InterOpTest, transition_from_static_thread_pool) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    experimental::execution::static_thread_pool pool{1};

    Tests::Utils::show_exec_space_id(exec, "exec");

    auto chain =
        stdexec::schedule(pool.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 0, .value = 4, .data = data.data()})
        | stdexec::continues_on(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(chain)

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable {    // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
        });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 8);
}

//! @test Transition from @c experimtnal::execution::static_thread_pool to @ref Kokkos::Execution::ExecutionSpaceContext and back.
TEST_F(InterOpTest, transition_from_static_thread_pool_and_back) {
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    experimental::execution::static_thread_pool pool{1};

    Tests::Utils::show_exec_space_id(exec, "exec");

    auto chain =
        stdexec::schedule(pool.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 0, .value = 4, .data = data.data()})
        | stdexec::continues_on(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 4, .value = 4, .data = data.data()})
        | stdexec::continues_on(pool.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 8, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(chain)

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable {    // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
        });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))));

    ASSERT_EQ(data(), 12);
}

//! @test Transition from @ref Kokkos::Execution::ExecutionSpaceContext to @c experimental::execution::static_thread_pool and back.
TEST_F(InterOpTest, transition_to_static_thread_pool_and_back) {
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    experimental::execution::static_thread_pool pool{1};

    Tests::Utils::show_exec_space_id(exec, "exec");

    auto chain =
        stdexec::schedule(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | stdexec::continues_on(pool.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, false>{.prev = 4, .value = 4, .data = data.data()})
        | stdexec::continues_on(esc.get_scheduler())
        | stdexec::then(
            Tests::Utils::Functors::LoadCheckAdd<value_t, on_device>{.prev = 8, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP(chain)

    const auto recorded_events = recorder_listener_t::record(
        [chain = std::move(chain)]() mutable {    // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(chain)); // NOLINT(performance-move-const-arg)
        });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from")),
            MATCHER_FOR_BEGIN_PFOR(exec, dispatch_label(exec, "then")),
            MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))));

    ASSERT_EQ(data(), 12);
}

} // namespace Tests::ExecutionSpaceImpl
