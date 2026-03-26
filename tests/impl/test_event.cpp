#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/impl/event.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"

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

//! Fixture that does not set any @ref Kokkos::utils::callbacks::Manager callback.
class EventTestNoCallback : public Tests::Utils::ExecutionSpaceContextTest<TEST_EXECUTION_SPACE> {
   public:
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

//! Fixture that enables callbacks with @ref Kokkos::utils::tests::scoped::callbacks::Manager.
class EventTest
    : public EventTestNoCallback
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<ProfileEvent>;
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

#define KOKKOS_EXECUTION_TESTS_IMPL_EVENT(_fixture_, _name_, _statement_)                                              \
    TEST_F(_fixture_, _name_) {                                                                                        \
        if constexpr (EventTest::has_support<TEST_EXECUTION_SPACE>) {                                                  \
            test_event_##_name_ _statement_;                                                                           \
        } else {                                                                                                       \
            GTEST_SKIP() << Kokkos::Impl::TypeInfo<TEST_EXECUTION_SPACE>::name() << " does not support events.";       \
        }                                                                                                              \
    }

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

KOKKOS_EXECUTION_TESTS_IMPL_EVENT(EventTest, record_and_wait, (exec))

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

KOKKOS_EXECUTION_TESTS_IMPL_EVENT(EventTest, record_and_wait_many_times, (exec))

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

KOKKOS_EXECUTION_TESTS_IMPL_EVENT(EventTest, record_and_wait_and_record_and_wait, (exec))

//! @test Similar to @ref test_event_record_and_wait but does not use any event listener.
template <Kokkos::ExecutionSpace Exec>
void test_event_record_and_wait_no_check(const Exec& exec) {
    ASSERT_THAT(Kokkos::utils::callbacks::Manager::is_initialized(), ::testing::IsFalse());

    Kokkos::Execution::Impl::Event<Exec> event;
    event.record(exec);
    event.wait();
}

KOKKOS_EXECUTION_TESTS_IMPL_EVENT(EventTestNoCallback, record_and_wait_no_check, (exec))

} // namespace Tests::Impl
