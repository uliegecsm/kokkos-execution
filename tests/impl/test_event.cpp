#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/impl/event.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/no_op.hpp"

/**
 * @addtogroup unittests
 *
 * Events
 * ------
 *
 * This group of tests check the events defined in @ref kokkos-execution/impl/event.hpp.
 *
 * The tests can be found in @ref tests/impl/test_event.cpp.
 */

namespace Tests::Impl {

using namespace Kokkos::utils::callbacks;

class EventTest
    : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<ProfileEvent>;

    /**
     * Events must be supported for:
     *  - @c Kokkos::Cuda
     *  - @c Kokkos::HIP
     *  - @c Kokkos::Experimental::HPX
     */
    template <Kokkos::ExecutionSpace Exec>
    static constexpr bool has_support = []() {
#if defined(KOKKOS_ENABLE_CUDA)
        if constexpr (std::same_as<Exec, Kokkos::Cuda>)
            return true;
#endif
#if defined(KOKKOS_ENABLE_HIP)
        if constexpr (std::same_as<Exec, Kokkos::HIP>)
            return true;
#endif
#if defined(KOKKOS_ENABLE_HPX)
        if constexpr (std::same_as<Exec, Kokkos::Experimental::HPX>)
            return true;
#endif
        return false;
    }();
};

//! @test Check that the specialization of @ref Kokkos::Execution::Impl::Event satisfies @ref Kokkos::Execution::Impl::event.
template <Kokkos::ExecutionSpace Exec>
consteval bool test_models_event() {
    if constexpr (EventTest::has_support<Exec>) {
        return Kokkos::Execution::Impl::event<Kokkos::Execution::Impl::Event<Exec>, Exec>;
    }
    return true;
}
static_assert(test_models_event<TEST_EXECUTION_SPACE>());

//! @test Check @ref Kokkos::Execution::Impl::support_events.
template <Kokkos::ExecutionSpace Exec>
consteval bool test_support_events() {
    return Kokkos::Execution::Impl::support_events<Exec> == EventTest::has_support<Exec>;
}
static_assert(test_support_events<TEST_EXECUTION_SPACE>());

//! @test Record an event and wait for it. Check that it marks both steps.
template <Kokkos::ExecutionSpace Exec>
void test_event_record_and_wait(const Exec& exec) {
    const auto recorded_events = EventTest::recorder_listener_t::record([&exec]() {
        Kokkos::Execution::Impl::Event<Exec> event;
        event.record(exec);
        event.wait();
    });

    ASSERT_EQ(recorded_events.size(), 2);
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_EVENT_RECORD(exec));
    const auto event_id = extract_event_record_id(recorded_events.at(0));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_EVENT_WAIT(TEST_EXECUTION_SPACE, event_id));
}

TEST_F(EventTest, record_and_wait) {
    if constexpr (EventTest::has_support<TEST_EXECUTION_SPACE>) {
        test_event_record_and_wait(exec);
    } else {
        GTEST_SKIP() << Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name() << " does not support events.";
    }
}

//! @test Record an event and wait for it many times. Check that it marks both steps once, and subsequent waits are just "for free".
template <Kokkos::ExecutionSpace Exec>
void test_event_record_and_wait_many_times(const Exec& exec) {
    const auto recorded_events = EventTest::recorder_listener_t::record([&exec]() {
        Kokkos::Execution::Impl::Event<Exec> event;
        event.record(exec);
        event.wait();
        event.wait();
        event.wait();
        event.wait();
        event.wait();
    });

    ASSERT_EQ(recorded_events.size(), 2);
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_EVENT_RECORD(exec));
    const auto event_id = extract_event_record_id(recorded_events.at(0));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_EVENT_WAIT(TEST_EXECUTION_SPACE, event_id));
}

TEST_F(EventTest, record_and_wait_many_times) {
    if constexpr (EventTest::has_support<TEST_EXECUTION_SPACE>) {
        test_event_record_and_wait_many_times(exec);
    } else {
        GTEST_SKIP() << Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name() << " does not support events.";
    }
}

//! @test Record an event and wait for it. Repeat the record/wait steps but reusing the same instance.
template <Kokkos::ExecutionSpace Exec>
void test_event_record_and_wait_and_record_and_wait(const Exec& exec) {
    const auto recorded_events = EventTest::recorder_listener_t::record([&exec]() {
        Kokkos::Execution::Impl::Event<Exec> event;
        event.record(exec);
        event.wait();
        event.record(exec);
        event.wait();
    });

    ASSERT_EQ(recorded_events.size(), 4);

    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_EVENT_RECORD(exec));
    auto event_id = extract_event_record_id(recorded_events.at(0));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_EVENT_WAIT(TEST_EXECUTION_SPACE, event_id));

    ASSERT_THAT(recorded_events.at(2), MATCHER_FOR_EVENT_RECORD(exec));
    event_id = extract_event_record_id(recorded_events.at(2));
    ASSERT_THAT(recorded_events.at(3), MATCHER_FOR_EVENT_WAIT(TEST_EXECUTION_SPACE, event_id));
}

TEST_F(EventTest, record_and_wait_and_record_and_wait) {
    if constexpr (EventTest::has_support<TEST_EXECUTION_SPACE>) {
        test_event_record_and_wait_and_record_and_wait(exec);
    } else {
        GTEST_SKIP() << Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name() << " does not support events.";
    }
}

#if defined(KOKKOS_ENABLE_HPX)
/**
 * @test Before https://github.com/kokkos/kokkos/commit/d71a3b7287e7a4c06b1595332eedc98cf4c76157,
 *       using @ref Kokkos::Execution::Impl::Event with @c Kokkos::Experimental::HPX instances that have
 *       not been created with the "independent" mode always crashes because the @c Kokkos::Experimental::HPX::instance_data
 *       destructor would not fence.
 *       Therefore, the event would create a "split" sender (with a shared state), that the @c Kokkos::Experimental::HPX::instance_data
 *       shares as well. Since @ref Kokkos::Execution::Impl::Event will synchronize only its own split shared state, and since the
 *       @c Kokkos::Experimental::HPX::instance_data would never synchronize its own if no one calls the @c fence on the @c Kokkos::Experimental::HPX
 *       instance, the shared state is alive past the @c Kokkos::finalize and @c Kokkos appropriately shouts.
 */
TEST(EventTestHPX, cleanup_when_using_default_instance) {
    const Kokkos::Experimental::HPX exec{};

    Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, 1), Tests::Utils::Functors::NoOp{});
    Kokkos::Execution::Impl::Event<Kokkos::Experimental::HPX> event{exec};

    event.wait();
}
#endif

} // namespace Tests::Impl
