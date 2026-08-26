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
    static_assert(Kokkos::Execution::Impl::event<Kokkos::Execution::Impl::Event<Exec>, Exec>);

    return true;
}
static_assert(test_models_event<TEST_EXECUTION_SPACE>());

//! @test Check @ref Kokkos::Execution::Impl::has_exec_wait_event.
template <Kokkos::ExecutionSpace Exec>
consteval bool test_has_exec_wait_event() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    if constexpr (std::same_as<Exec, Kokkos::DefaultExecutionSpace>) {
        static_assert(Kokkos::Execution::Impl::has_exec_wait_event<Exec>);
        return true;
    } else
#endif
#if defined(KOKKOS_ENABLE_HPX)
        if constexpr (std::same_as<Exec, Kokkos::Experimental::HPX>) {
        static_assert(Kokkos::Execution::Impl::has_exec_wait_event<Exec>);
        return true;
    } else
#endif
    {
        static_assert(!Kokkos::Execution::Impl::has_exec_wait_event<Exec>);
        return true;
    }
}
static_assert(test_has_exec_wait_event<TEST_EXECUTION_SPACE>());

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

    ASSERT_EQ(oss.str(), "WaitEvent: {dev_id = 0, event_id = 1337}");
}

//! @test Record an event and wait for it. Check that it marks both steps.
TEST_F(EventTest, record_and_wait) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event;
        Kokkos::Execution::Impl::record(event, exec);
        Kokkos::Execution::Impl::wait(event);
    });

    ASSERT_THAT(recorded_events, testing::SizeIs(2));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
}

//! @test Record an event and wait for it. Repeat the record/wait steps but reusing the same instance.
TEST_F(EventTest, record_and_wait_and_record_and_wait) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event;
        Kokkos::Execution::Impl::record(event, exec);
        Kokkos::Execution::Impl::wait(event);
        Kokkos::Execution::Impl::record(event, exec);
        Kokkos::Execution::Impl::wait(event);
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
        Kokkos::Execution::Impl::record(event_before, exec);
        Kokkos::parallel_for(Kokkos::RangePolicy(exec, 0, 1), Tests::Utils::Functors::NoOp{});
        Kokkos::Execution::Impl::record(event_after, exec);

        Kokkos::Execution::Impl::wait(event_before);
        Kokkos::Execution::Impl::wait(event_after);
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(4));

    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec));

    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_RECORD_EVENT(exec));

    ASSERT_NE(
        std::get<Kokkos::Execution::Impl::RecordEvent>(recorded_events.at(0)).event_id,
        std::get<Kokkos::Execution::Impl::RecordEvent>(recorded_events.at(1)).event_id);

    ASSERT_THAT(recorded_events.at(2), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(3), MATCHER_FOR_WAIT_EVENT(recorded_events.at(1)));
}

//! @test Check that event record/wait works for the default instance.
TEST_F(EventTest, default_instance) {
    const TEST_EXECUTION_SPACE default_exec{};

    const auto recorded_events = recorder_listener_t::record([&default_exec]() {
        Kokkos::parallel_for(Kokkos::RangePolicy(default_exec, 0, 1), Tests::Utils::Functors::NoOp{});
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event;
        Kokkos::Execution::Impl::record(event, default_exec);
        Kokkos::Execution::Impl::wait(event);
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(2));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(default_exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EVENT(recorded_events.at(0)));
}

//! @test Check @ref Kokkos::Execution::Impl::wait when the underlying execution space type is the same.
TEST_F(EventTest, wait_exec_event_for_same_type) {
    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const auto recorded_events = recorder_listener_t::record([&exec_A, &exec_B]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event_A;
        Kokkos::Execution::Impl::record(event_A, exec_A);
        Kokkos::Execution::Impl::wait(exec_B, event_A);

        //! Ensure the event has been waited for.
        exec_B.fence("some-label");
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(3));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec_A));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EXEC_EVENT(exec_B, recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(2), MATCHER_FOR_BEGIN_FENCE(exec_B, "some-label"));
}

//! @test Check @ref Kokkos::Execution::Impl::wait when the underlying execution space type is the same and there are many events to wait for.
TEST_F(EventTest, wait_exec_events_for_same_type) {
    const auto [exec_A, exec_B] = Kokkos::Experimental::partition_space(exec, 1, 1);

    const auto recorded_events = recorder_listener_t::record([&exec_A, &exec_B]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event_A_0, event_A_1, event_A_2;
        Kokkos::Execution::Impl::record(event_A_0, exec_A);
        Kokkos::Execution::Impl::record(event_A_1, exec_A);
        Kokkos::Execution::Impl::record(event_A_2, exec_A);
        Kokkos::Execution::Impl::wait(exec_B, event_A_0, event_A_1, event_A_2);

        //! Ensure the events have been waited for.
        exec_B.fence("some-label");
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(7));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(exec_A));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_RECORD_EVENT(exec_A));
    ASSERT_THAT(recorded_events.at(2), MATCHER_FOR_RECORD_EVENT(exec_A));
    ASSERT_THAT(recorded_events.at(3), MATCHER_FOR_WAIT_EXEC_EVENT(exec_B, recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(4), MATCHER_FOR_WAIT_EXEC_EVENT(exec_B, recorded_events.at(1)));
    ASSERT_THAT(recorded_events.at(5), MATCHER_FOR_WAIT_EXEC_EVENT(exec_B, recorded_events.at(2)));
    ASSERT_THAT(recorded_events.at(6), MATCHER_FOR_BEGIN_FENCE(exec_B, "some-label"));
}

//! @test Check @ref Kokkos::Execution::Impl::wait when the underlying execution space type is different.
TEST_F(EventTest, wait_exec_event_different_type) {
    if constexpr (std::same_as<TEST_EXECUTION_SPACE, Kokkos::DefaultHostExecutionSpace>) {
        GTEST_SKIP() << "The default host execution space is the same type as the test execution space.";
    }

    const Kokkos::DefaultHostExecutionSpace exec_h;

    const auto recorded_events = recorder_listener_t::record([this, &exec_h]() {
        Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event_d;
        Kokkos::Execution::Impl::record(event_d, this->exec);
        Kokkos::Execution::Impl::wait(exec_h, event_d);

        //! Ensure the event has been waited for.
        exec_h.fence("some-label");
    });

    ASSERT_THAT(recorded_events, ::testing::SizeIs(3));
    ASSERT_THAT(recorded_events.at(0), MATCHER_FOR_RECORD_EVENT(this->exec));
    ASSERT_THAT(recorded_events.at(1), MATCHER_FOR_WAIT_EXEC_EVENT(exec_h, recorded_events.at(0)));
    ASSERT_THAT(recorded_events.at(2), MATCHER_FOR_BEGIN_FENCE(exec_h, "some-label"));
}

/**
 * @test Check that event record/wait works.
 *
 * This test reflects the intended usage pattern in @ref Kokkos::Execution. But on backends with blocking dispatch, it may not really check
 * the visibility effect of the wait because the enqueueing of the run loop may itself involve a release-acquire operation.
 */
TEST_F(EventTest, works_intended_usage_pattern) {
    constexpr size_t size = 128;

    const auto [exec_A, exec_B, exec_C] = Kokkos::Experimental::partition_space(exec, 1, 1, 1);

    const Kokkos::View<int, TEST_EXECUTION_SPACE> data(Kokkos::view_alloc(exec_A, "data"));
    const auto data_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, data);

    std::optional<Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE>> event_A;

    stdexec::run_loop loop;
    std::thread consumer([&] { loop.run(); });

    Kokkos::parallel_for(
        Kokkos::RangePolicy(exec_A, 0, size), KOKKOS_LAMBDA(const auto idx) { Kokkos::atomic_add(&data(), idx); });
    event_A.emplace();
    Kokkos::Execution::Impl::record(*event_A, exec_A);

    bool upon_error = false;

    auto op_state = stdexec::connect(
        stdexec::schedule(loop.get_scheduler()) | stdexec::then([&] {
            event_A->wait();
            event_A.reset();
            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec_B, 0, size),
                KOKKOS_LAMBDA(const auto idx) { Kokkos::atomic_add(&data(), idx); });
            Kokkos::Execution::Impl::Event<TEST_EXECUTION_SPACE> event_B;
            Kokkos::Execution::Impl::record(event_B, exec_B);
            Kokkos::Execution::Impl::wait(exec_C, event_B);
            Kokkos::parallel_for(
                Kokkos::RangePolicy(exec_C, 0, 1),
                KOKKOS_LAMBDA(const auto) { Kokkos::atomic_compare_exchange(&data(), size * (size - 1), 1); });
            Kokkos::deep_copy(exec_C, data_h, data);
            exec_C.fence("wait for the deep copy to complete");
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

    op_state.start();

    consumer.join();

    ASSERT_FALSE(event_A.has_value());

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
