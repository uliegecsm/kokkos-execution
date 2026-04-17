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
struct HasNonBlockingDispatch<Kokkos::HIP> : std::true_type { };

template <>
struct Event<Kokkos::HIP> {
    hipEvent_t m_event = nullptr;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;

    explicit Event(const Kokkos::HIP& exec) {
        record(exec);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&&) noexcept = delete;
    Event& operator=(Event&&) noexcept = delete;

    ~Event() {
        if (m_event != nullptr)
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventDestroy(m_event));
    }

    void record(const Kokkos::HIP& exec) {
        if (m_event == nullptr) {
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventCreateWithFlags(&m_event, hipEventDisableTiming));
        }
        KOKKOS_IMPL_HIP_SAFE_CALL(hipEventRecord(m_event, exec.hip_stream()));
        record_event(exec, m_event_id);
    }

    void wait() const {
        wait_event(m_event_id);
        if (hipEventQuery(m_event) != hipSuccess) {
            KOKKOS_IMPL_HIP_SAFE_CALL(hipEventSynchronize(m_event));
        }
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HIP_EVENT_HPP
