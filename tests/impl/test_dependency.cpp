#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/impl/dependency.hpp"
#include "kokkos-execution/impl/event.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"

/**
 * @addtogroup unittests
 *
 * Dependency
 * ----------
 *
 * This group of tests check @ref Kokkos::Execution::Impl::Dependency.
 *
 * The tests can be found in @ref tests/impl/test_dependency.cpp.
 */

#if !defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
#    error "This is not supported."
#endif

namespace Tests::Impl {

using namespace Kokkos::utils::callbacks;

class DependencyTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        EventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        Kokkos::Execution::Impl::RecordEvent,
        Kokkos::Execution::Impl::WaitEvent
    >;
};

//! @test Check @ref Kokkos::Execution::Impl::Dependency when the underlying execution space type is the same.
TEST_F(DependencyTest, same_type) {
    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const auto recorded_events = recorder_listener_t::record(
        [&exec_A, &exec_B]() { const Kokkos::Execution::Impl::Dependency dependency{exec_B, exec_A}; });

    if constexpr (Kokkos::Execution::Impl::has_exec_wait_event<TEST_EXECUTION_SPACE>) {
        ASSERT_THAT(recorded_events, ::testing::SizeIs(2));
        ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec_A));
        ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EXEC_EVENT(exec_B, recorded_events.at(0)));
    } else {
        if (Tests::Utils::are_same_instances(exec_A, exec_B)) {
            ASSERT_THAT(recorded_events, testing::IsEmpty());
        } else {
            ASSERT_THAT(
                recorded_events,
                testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(exec_A, dispatch_label(exec_A, "dependency"))));
        }
    }
}

//! @test Check @ref Kokkos::Execution::Impl::Dependency when the underlying execution space type is different.
TEST_F(DependencyTest, different_type) {
    if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::DefaultHostExecutionSpace>) {
        GTEST_SKIP() << "The default host execution space is the same type as the test execution space.";
    }

    const Kokkos::DefaultHostExecutionSpace exec_h;

    const auto recorded_events = recorder_listener_t::record(
        [this, &exec_h]() { const Kokkos::Execution::Impl::Dependency dependency{exec_h, this->exec}; });

    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(MATCHER_FOR_BEGIN_FENCE(this->exec, dispatch_label(this->exec, "dependency"))));
}

} // namespace Tests::Impl
