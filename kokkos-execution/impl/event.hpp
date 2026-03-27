#ifndef KOKKOS_EXECUTION_IMPL_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_EVENT_HPP

#include <concepts>
#include <format>

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
concept event = Kokkos::ExecutionSpace<Exec> && std::default_initializable<EventType>
             && std::constructible_from<EventType, const Exec&> && std::move_constructible<EventType>
             && requires(EventType& obj, const Exec& exec) {
                    { obj.record(exec) } -> std::same_as<void>;
                } && requires(const EventType& obj) {
                    { obj.wait() } -> std::same_as<void>;
                };

//! An event that can be recorded on an execution space instance.
template <Kokkos::ExecutionSpace Exec>
struct Event;

//! Determine if events are supported.
template <Kokkos::ExecutionSpace Exec>
struct SupportEvents : std::false_type { };

template <typename Exec>
concept support_events = SupportEvents<Exec>::value;

static constexpr auto invalid_event_id = Kokkos::Experimental::finite_max_v<uint64_t>;

//! Event to be sent to @ref Kokkos::utils::callbacks::dispatch when an event is recorded on an execution space instance.
struct RecordEvent {
    uint32_t dev_id = 0;
    uint64_t event_id = 0;

    constexpr auto operator<=>(const RecordEvent&) const = default;
};

template <Kokkos::ExecutionSpace Exec>
void record_event(const Exec& exec, uint64_t& event_id) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    event_id = Kokkos::utils::callbacks::get_next_event_id();
    Kokkos::utils::callbacks::dispatch(
        RecordEvent{.dev_id = Kokkos::Tools::Experimental::device_id(exec), .event_id = event_id});
#endif
}

//! Event to be sent to @ref Kokkos::utils::callbacks::dispatch when an event is being waited for.
struct WaitEvent {
    uint64_t event_id = 0;

    constexpr auto operator<=>(const WaitEvent&) const = default;
};

inline void wait_event(const uint64_t event_id) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    Kokkos::utils::callbacks::dispatch(WaitEvent{.event_id = event_id});
#endif
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

#endif // KOKKOS_EXECUTION_IMPL_EVENT_HPP
