#ifndef KOKKOS_EXECUTION_IMPL_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_EVENT_HPP

#include <concepts>
#include <format>

#include "Kokkos_Core.hpp"

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

//! Helper for marking event record or wait.
template <Kokkos::ExecutionSpace Exec>
struct MarkEvent {
    static constexpr auto name = Kokkos::Impl::TypeInfo<Exec>::name();

    //! Mark that an event has been recorded in @p exec.
    template <typename EventIDType>
    static void record(const EventIDType& event_id, const Exec& exec) {
        Kokkos::Profiling::markEvent(
            std::format("{}: event {} recorded on {}", name, event_id, Kokkos::Tools::Experimental::device_id(exec)));
    }

    //! Mark that an event is being waited for.
    template <typename EventIDType>
    static void wait(const EventIDType& event_id) {
        Kokkos::Profiling::markEvent(std::format("{}: waiting for event {}", name, event_id));
    }
};

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
