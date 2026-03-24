#ifndef KOKKOS_EXECUTION_IMPL_HIP_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_HIP_EVENT_HPP

#include "Kokkos_Core.hpp"

/**
 * @file
 *
 * Specialization of @ref Kokkos::Execution::Impl::Event for @c Kokkos::HIP.
 */

namespace Kokkos::Execution::Impl {

template <>
struct SupportEvents<Kokkos::HIP> : std::true_type { };

template <>
struct Event<Kokkos::HIP> {
    using mark_event_t = MarkEvent<Kokkos::HIP>;

    hipEvent_t event = nullptr;
    mutablebool arrived = false;

    Event() = default;
    
    explicit Event(const Kokkos::HIP& exec) {
        record(exec);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept
        : event(std::exchange(other.event, nullptr))
        , arrived(std::exchange(other.arrived, false)) {
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (event != nullptr) {
                KOKKOS_IMPL_HIP_SAFE_CALL(hipEventDestroy(event));
            }
            event = std::exchange(other.event, nullptr);
            arrived = std::exchange(other.arrived, false);
        }
        return *this;
    }

    ~Event() {
        if (event != nullptr)
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventDestroy(event));
    }

    void record(const Kokkos::HIP& exec) {
        if (event == nullptr) {
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        }
        KOKKOS_IMPL_HIP_SAFE_CALL(hipEventRecord(event, exec.hip_stream()));
        arrived = false;
        mark_event_t::record(static_cast<void*>(event), exec);
    }

    void wait() const {
        if (!arrived) {
            mark_event_t::wait(static_cast<void*>(event));
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventSynchronize(event));
            arrived = true;
        }
    }
};

Event(const Kokkos::HIP&) -> Event<Kokkos::HIP>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HIP_EVENT_HPP
