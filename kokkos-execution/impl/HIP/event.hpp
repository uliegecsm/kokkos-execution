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

    hipEvent_t m_event = nullptr;

    Event() = default;

    explicit Event(const Kokkos::HIP& exec) {
        record(exec);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept
        : m_event(std::exchange(other.m_event, nullptr)) {
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (m_event != nullptr) {
                KOKKOS_IMPL_HIP_SAFE_CALL(hipEventDestroy(m_event));
            }
            m_event = std::exchange(other.m_event, nullptr);
        }
        return *this;
    }

    ~Event() {
        if (m_event != nullptr)
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventDestroy(m_event));
    }

    void record(const Kokkos::HIP& exec) {
        if (m_event == nullptr) {
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventCreateWithFlags(&m_event, hipEventDisableTiming));
        }
        KOKKOS_IMPL_HIP_SAFE_CALL(hipEventRecord(m_event, exec.hip_stream()));
        mark_event_t::record(static_cast<void*>(m_event), exec);
    }

    void wait() const {
        mark_event_t::wait(static_cast<void*>(m_event));
        if (hipEventQuery(m_event) != hipSuccess) {
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventSynchronize(m_event));
        }
    }
};

Event(const Kokkos::HIP&) -> Event<Kokkos::HIP>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HIP_EVENT_HPP
