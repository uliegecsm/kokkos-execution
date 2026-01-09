#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
#include "exec/static_thread_pool.hpp"
#include "stdexec/execution.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/CallbackMatchers.hpp"
#include "tests/Utils.hpp"
#include "tests/kokkos_ext/execution_space/Helpers.hpp"
#include "tests/stdexec/Utils.hpp"
#include "tests/utils/LoadCheckAdd.hpp"

/**
 * @addtogroup unittests
 *
 * Interoperability of @c Kokkos::Experimental::ExecutionSpaceContext with other schedulers
 * ----------------------------------------------------------------------------------------
 *
 * This group of tests check that @ref Kokkos::Experimental::ExecutionSpaceContext can be used in
 * conjunction with other schedulers like @c exec::static_thread_pool.
 *
 * The tests can be found in @ref kokkos_ext/execution_space/test_inter_op.cpp.
 */

using execution_space = Kokkos::DefaultExecutionSpace;

namespace tests::kokkos_ext
{

using namespace Kokkos::utils::callbacks;

class InterOpTest : public impl::ExecutionSpaceContextTest<execution_space>,
                    public Kokkos::utils::tests::scoped::callbacks::Manager
{
public:
    using recorder_listener_t = RecorderListener<BeginFenceEvent, BeginParallelForEvent>;
    using variant_t           = std::variant    <BeginFenceEvent, BeginParallelForEvent>;

    using value_t = typename view_s_t::value_type;

#if defined(KOKKOS_ENABLE_CUDA)
    static constexpr bool on_device = std::same_as<execution_space, Kokkos::Cuda>;
#elif defined(KOKKOS_ENABLE_HIP)
    static constexpr bool on_device = std::same_as<execution_space, Kokkos::HIP>;
#else
    static constexpr bool on_device = false;
#endif
};

//! @test Transition from @ref Kokkos::Experimental::ExecutionSpaceContext to @c stdexec::inline_scheduler.
TEST_F(InterOpTest, transition_to_inline_scheduler)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    SHOW_EXEC_SPACE_ID(exec)

    auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(::stdexec::inline_scheduler{})
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, false>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });
    for (const auto& recorded_event : recorded_events) {
        std::visit([] (const auto& arg) { std::cout << "- " << arg << std::endl; }, recorded_event);
    }

    EXPECT_THAT(recorded_events, ::testing::ElementsAre(
        MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
        MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))
    ));

    ASSERT_EQ(data(), 8);
}

//! @test Transition from @c stdexec::inline_scheduler to @ref Kokkos::Experimental::ExecutionSpaceContext.
TEST_F(InterOpTest, transition_from_inline_scheduler)
{
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    SHOW_EXEC_SPACE_ID(exec)

    auto chain = ::stdexec::schedule(::stdexec::inline_scheduler{})
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, false>{.prev = 0, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(esc.get_scheduler())
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });

    EXPECT_THAT(recorded_events, ::testing::ElementsAre(
        MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
        MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
    ));

    ASSERT_EQ(data(), 8);
}

//! @test Transition from @c stdexec::inline_scheduler to @ref Kokkos::Experimental::ExecutionSpaceContext and back.
TEST_F(InterOpTest, transition_from_inline_scheduler_and_back)
{
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    SHOW_EXEC_SPACE_ID(exec)

    auto chain = ::stdexec::schedule(::stdexec::inline_scheduler{})
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, false>{.prev = 0, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(esc.get_scheduler())
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 4, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(::stdexec::inline_scheduler{})
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, false>{.prev = 8, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });

    EXPECT_THAT(recorded_events, ::testing::ElementsAre(
        MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
        MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))
    ));

    ASSERT_EQ(data(), 12);
}

//! @test Transition from @ref Kokkos::Experimental::ExecutionSpaceContext to @c exec::static_thread_pool.
TEST_F(InterOpTest, transition_to_static_thread_pool)
{
    const view_s_t data(Kokkos::view_alloc(exec, "data - shared space"));

    const context_t esc{exec};

    ::exec::static_thread_pool pool{1};

    SHOW_EXEC_SPACE_ID(exec)

    auto chain = ::stdexec::schedule(esc.get_scheduler())
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 0, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(pool.get_scheduler())
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, false>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });

    EXPECT_THAT(recorded_events, ::testing::ElementsAre(
        MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
        MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "schedule_from"))
    ));

    ASSERT_EQ(data(), 8);
}

//! @test Transition from @c exec::static_thread_pool to @ref Kokkos::Experimental::ExecutionSpaceContext.
TEST_F(InterOpTest, transition_from_static_thread_pool)
{
    const view_s_t data(Kokkos::view_alloc("data - shared space"));

    const context_t esc{exec};

    ::exec::static_thread_pool pool{1};

    SHOW_EXEC_SPACE_ID(exec)

    auto chain = ::stdexec::schedule(pool.get_scheduler())
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, false>{.prev = 0, .value = 4, .data = data.data()})
        | ::stdexec::continues_on(esc.get_scheduler())
        | ::stdexec::then(tests::utils::LoadCheckAddFunctor<value_t, on_device>{.prev = 4, .value = 4, .data = data.data()});

    ASSERT_EQ(data(), 0) << "Eager execution is not allowed.";

    const auto recorded_events = recorder_listener_t::record([chain = std::move(chain)] () mutable { ::stdexec::sync_wait(std::move(chain)); });

    EXPECT_THAT(recorded_events, ::testing::ElementsAre(
        MATCHER_FOR_BEGIN_PFOR (exec, dispatch_label(exec, "then")),
        MATCHER_FOR_BEGIN_FENCE(exec, dispatch_label(exec, "sync_wait"))
    ));

    ASSERT_EQ(data(), 8);
}

} // namespace tests::kokkos_ext
