#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/event.hpp"

#include "tests/utils/callback_matchers.hpp"
#include "tests/utils/execution_space_context.hpp"
#include "tests/utils/functors/no_op.hpp"
#include "tests/utils/sink_receiver.hpp"

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

#if !defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
#    error "This is not supported."
#endif

namespace Tests::Impl {

using namespace Kokkos::utils::callbacks;

//! Fixture that enables callbacks with @ref Kokkos::utils::tests::scoped::callbacks::Manager.
class EventTest
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

//! @test Check that @ref Kokkos::Execution::Impl::Event satisfies @ref Kokkos::Execution::Impl::event.
template <Kokkos::ExecutionSpace Exec>
consteval bool test_models_event() {
    return Kokkos::Execution::Impl::event<Kokkos::Execution::Impl::Event<Exec>, Exec>;
}
static_assert(test_models_event<TEST_EXECUTION_SPACE>());

//! @test Check the stream operator of @ref Kokkos::Execution::Impl::RecordEvent.
TEST(RecordEvent, description) {
    const Kokkos::Execution::Impl::RecordEvent event{.dev_id = 42, .event_id = 1337};

    std::ostringstream oss;
    oss << event;

    ASSERT_EQ(oss.str(), "RecordEvent: {dev_id = 42, event_id = 1337}");
}

//! @test Check the stream operator of @ref Kokkos::Execution::Impl::WaitEvent.
TEST(WaitEvent, description) {
    const Kokkos::Execution::Impl::WaitEvent event{.event_id = 1337};

    std::ostringstream oss;
    oss << event;

    ASSERT_EQ(oss.str(), "WaitEvent: {event_id = 1337}");
}

//! @test Record an event and wait for it. Check that it marks both steps.
TEST_F(EventTest, record_and_wait) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event;
        event.record(exec);
        event.wait();
    });

    ASSERT_THAT(recorded_events, testing::SizeIs(2));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
}

//! @test Record an event but don't wait for it.
TEST_F(EventTest, record_but_dont_wait) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event;
        Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, 1), Tests::Utils::Functors::NoOp{});
        event.record(exec);

        exec.fence("some-label");
    });

    ASSERT_THAT(recorded_events, testing::SizeIs(2));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_BEGIN_FENCE(exec, "some-label"));
}

//! @test Record an event and wait for it many times. It marks all wait events.
TEST_F(EventTest, record_and_wait_many_times) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event;
        event.record(exec);
        event.wait();
        event.wait();
        event.wait();
        event.wait();
        event.wait();
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(6));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(2), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(3), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(4), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(5), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
}

//! @test Record an event and wait for it. Repeat the record/wait steps but reusing the same instance.
TEST_F(EventTest, record_and_wait_and_record_and_wait) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event;
        event.record(exec);
        event.wait();
        event.record(exec);
        event.wait();
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(4));

    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));

    ASSERT_THAT(recorded_events.at(2), MATCHER_FOR_RECORD_EVENT(exec));
    ASSERT_THAT(recorded_events.at(3), MATCHER_FOR_WAIT_EVENT(recorded_events.at(2)));
}

//! @test Events created before or after a work dispatch have distinct identifiers.
TEST_F(EventTest, uniqueness) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event_before, event_after;
        event_before.record(exec);
        Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, 1), Tests::Utils::Functors::NoOp{});
        event_after.record(exec);

        exec.fence("some-label");
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(3));

    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec));

    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_RECORD_EVENT(exec));

    ASSERT_NE(
        std::get<Kokkos::Execution::Impl::RecordEvent>(recorded_events.at(0)).event_id,
        std::get<Kokkos::Execution::Impl::RecordEvent>(recorded_events.at(1)).event_id);

    ASSERT_THAT(recorded_events.back(), MATCHER_FOR_BEGIN_FENCE(exec, "some-label"));
}

//! @test Check that event record/wait works for the default instance.
TEST_F(EventTest, default_instance) {
    const TEST_EXECUTION_SPACE default_exec{};

    const auto recorded_events = recorder_listener_t::record([&default_exec]() {
        Kokkos::parallel_for(Kokkos::RangePolicy(default_exec, 0, 1), Tests::Utils::Functors::NoOp{});
        const Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event{default_exec};
        event.wait();
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(2));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(default_exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
}

/**
 * @test Check that event record/wait works.
 *
 * This test reflects the intended usage pattern in @ref Kokkos::Execution. But on backends with blocking dispatch, it may not really check
 * the visibility effect of the wait because the enqueueing of the run loop may itself involve a release-acquire operation.
 */
TEST_F(EventTest, works_intended_usage_pattern) {
    constexpr size_t size = 128;

    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const Kokkos::View<int, TEST_EXECUTION_SPACE> data(Kokkos::view_alloc(exec_A, "data"));
    const auto data_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, data);

    std::optional<Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE>> event;

    stdexec::run_loop loop;
    std::thread consumer([&] { loop.run(); });

    Kokkos::parallel_for(
        Kokkos::RangePolicy(exec_A, 0, size), KOKKOS_LAMBDA(const auto idx) { Kokkos::atomic_add(&data(), idx); });
    event.emplace(exec_A);

    bool upon_error = false;

    auto opstate = stdexec::connect(
        stdexec::schedule(loop.get_scheduler()) | stdexec::then([&] {
            event->wait();
            event.reset();
            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec_B, 0, 1),
                KOKKOS_LAMBDA(const auto) { Kokkos::atomic_compare_exchange(&data(), size / 2 * (size - 1), 1); });
            Kokkos::deep_copy(exec_B, data_h, data);
            exec_B.fence("wait for the deep copy to complete");
            loop.finish();
        }) | stdexec::upon_error([&](const auto& eptr) {
            upon_error = true;
            try {
                std::rethrow_exception(eptr);
            } catch (const std::exception& exc) {
                Kokkos::printf("%s\n", exc.what()); // NOLINT(modernize-use-std-print)
            }
            loop.finish();
        }),
        Tests::Utils::SinkReceiver{});

    opstate.start();

    consumer.join();

    ASSERT_FALSE(event.has_value());

//! See @ref KOKKOS_EXECUTION_THREADS_THROWS_ON_SYNC_WAIT_ASSERT_AND_SKIP for an explanation.
#if defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::Threads>) {
        ASSERT_EQ(data_h(), size / 2 * (size - 1));
        ASSERT_TRUE(upon_error);
    } else
#endif
    {
        ASSERT_EQ(data_h(), 1);
        ASSERT_FALSE(upon_error);
    }
}

} // namespace Tests::Impl
