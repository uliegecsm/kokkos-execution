#ifndef KOKKOS_EXECUTION_IMPL_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_EVENT_HPP

#include <concepts>

#include "Kokkos_Core.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
#    include "kokkos-utils/callbacks/Manager.hpp"
#endif

/**
 * @file
 *
 * Backends that support an event-like API specialize @ref Kokkos::Execution::Impl::Event for their execution space type.
 * The specialization must satisfy the @ref Kokkos::Execution::Impl::event concept.
 */

namespace Kokkos::Execution::Impl {

//! Constrain an @p EventType type to be a valid event type for @p Exec execution space type.
template <typename EventType, typename Exec>
concept event = Kokkos::ExecutionSpace<Exec> && std::default_initializable<EventType> && !std::copyable<EventType>
             && !std::movable<EventType> && requires(EventType& obj, const Exec& exec) {
                    { obj.record(exec) } -> std::same_as<void>;
                } && requires(const EventType& obj) {
                    { obj.wait() } -> std::same_as<void>;
                    { obj.m_event_id } -> std::same_as<const uint64_t&>;
                };

//! An event that can be recorded on an execution space instance.
template <Kokkos::ExecutionSpace Exec>
struct Event;

/**
 * @brief Determine if the @c Kokkos backend has non-blocking dispatch.
 *
 * A backend is said to have non-blocking dispatch if a dispatch (*e.g.* @c Kokkos::parallel_for)
 * is potentially asynchronous, *i.e.*, may return to the caller before the dispatched work completes.
 */
template <Kokkos::ExecutionSpace Exec>
struct HasNonBlockingDispatch : std::false_type { };

template <typename Exec>
concept has_non_blocking_dispatch = HasNonBlockingDispatch<Exec>::value;

static constexpr auto invalid_dev_id = Kokkos::Experimental::finite_max_v<uint32_t>;
static constexpr auto invalid_event_id = Kokkos::Experimental::finite_max_v<uint64_t>;

//! Event to be sent to @ref Kokkos::utils::callbacks::dispatch when calling @ref record.
struct RecordEvent {
    uint32_t dev_id = 0;
    uint64_t event_id = 0;

    constexpr auto operator<=>(const RecordEvent&) const = default;

    friend std::ostream& operator<<(std::ostream& out, const RecordEvent& event) {
        return out << "RecordEvent: {dev_id = " << event.dev_id << ", event_id = " << event.event_id << '}';
    }
};

template <Kokkos::ExecutionSpace Exec>
void record_event(const Exec& exec, uint64_t& event_id) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    event_id = Kokkos::utils::callbacks::get_next_event_id();
    Kokkos::utils::callbacks::dispatch(
        RecordEvent{.dev_id = Kokkos::Tools::Experimental::device_id(exec), .event_id = event_id});
#endif
}

//! Event to be sent to @ref Kokkos::utils::callbacks::dispatch when calling @ref wait.
struct WaitEvent {
    uint32_t dev_id = 0;
    uint64_t event_id = 0;

    constexpr auto operator<=>(const WaitEvent&) const = default;

    friend std::ostream& operator<<(std::ostream& out, const WaitEvent& event) {
        return out << "WaitEvent: {dev_id = " << event.dev_id << ", event_id = " << event.event_id << '}';
    }
};

template <Kokkos::ExecutionSpace Exec>
void wait_event(const Event<Exec>& event) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    Kokkos::utils::callbacks::dispatch(WaitEvent{.dev_id = invalid_dev_id, .event_id = event.m_event_id});
#endif
}

template <Kokkos::ExecutionSpace ExecTo, Kokkos::ExecutionSpace ExecFrom>
void wait_event(const ExecTo& exec, const Event<ExecFrom>& event) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    Kokkos::utils::callbacks::dispatch(
        WaitEvent{.dev_id = Kokkos::Tools::Experimental::device_id(exec), .event_id = event.m_event_id});
#endif
}

/**
 * @brief This is the default implementation. It assumes the backend has blocking dispatch.
 *
 * It is assumed that the thread calling @ref record does so after submitting the operation
 * on the execution space instance. Since the backend is assumed blocking, this means that
 * the operation must have completed before the call to @ref record, so that making
 * the effects visible after the call to @ref wait on the waiting thread is all that is needed.
 */
template <Kokkos::ExecutionSpace Exec>
struct Event {
    static_assert(!has_non_blocking_dispatch<Exec>);

    std::atomic<bool> m_recorded{false};

    uint64_t m_event_id = invalid_event_id;

    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&&) noexcept = delete;
    Event& operator=(Event&&) noexcept = delete;
    ~Event() = default;

    void record(const Exec&) {
        m_recorded.store(true, std::memory_order_release);
    }

    void wait() const {
        /// The while loop is needed because the promise that memory writes that happened before the atomic store from
        /// the point of view of the recording thread become visible in the waiting thread only holds if the waiting thread
        /// actually reads the value that the recording thread stored.
        /// See https://en.cppreference.com/cpp/atomic/memory_order.
        while (!m_recorded.load(std::memory_order_acquire)) {
        }
    }
};

//! Record @p event on @p exec.
template <Kokkos::ExecutionSpace Exec>
void record(Event<Exec>& event, const Exec& exec) {
    record_event(exec, event.m_event_id);
    event.record(exec);
}

//! Wait for @p event to complete.
template <Kokkos::ExecutionSpace Exec>
void wait(const Event<Exec>& event) {
    wait_event(event);
    event.wait();
}

template <Kokkos::ExecutionSpace ExecFrom, Kokkos::ExecutionSpace ExecTo>
void impl_wait(const ExecTo&, const Event<ExecFrom>& event) {
    event.wait();
}

/**
 * @brief The operations enqueued from the calling thread into @p exec cannot start before the event @p event completes.
 *
 * @note Backends should specialize @ref impl_wait.
 */
template <Kokkos::ExecutionSpace ExecFrom, Kokkos::ExecutionSpace ExecTo>
void wait(const ExecTo& exec, const Event<ExecFrom>& event) {
    wait_event(exec, event);
    impl_wait(exec, event);
}

} // namespace Kokkos::Execution::Impl

#if defined(KOKKOS_ENABLE_CUDA)
#    include "kokkos-execution/impl/Cuda/event.hpp"
#endif
#if defined(KOKKOS_ENABLE_HIP)
#    include "kokkos-execution/impl/HIP/event.hpp"
#endif
#if defined(KOKKOS_ENABLE_HPX)
#    include "kokkos-execution/impl/HPX/event.hpp"
#endif
#if defined(KOKKOS_ENABLE_SYCL)
#    include "kokkos-execution/impl/SYCL/event.hpp"
#endif

#endif // KOKKOS_EXECUTION_IMPL_EVENT_HPP
