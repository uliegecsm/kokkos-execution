#ifndef KOKKOS_EXECUTION_IMPL_CUDA_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_CUDA_EVENT_HPP

#include "Kokkos_Core.hpp"

/**
 * @file
 *
 * Specialization of @ref Kokkos::Execution::Impl::Event for @c Kokkos::Cuda.
 */

namespace Kokkos::Execution::Impl {

template <>
struct SupportEvents<Kokkos::Cuda> : std::true_type { };

template <>
struct Event<Kokkos::Cuda> {
    cudaEvent_t m_event = nullptr;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;

    explicit Event(const Kokkos::Cuda& exec) {
        record(exec);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept
        : m_event(std::exchange(other.m_event, nullptr))
        , m_event_id(std::exchange(other.m_event_id, invalid_event_id)) {
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (m_event != nullptr) {
                KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(m_event));
            }
            m_event = std::exchange(other.m_event, nullptr);
            m_event_id = std::exchange(other.m_event_id, invalid_event_id);
        }
        return *this;
    }

    ~Event() {
        if (m_event != nullptr)
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(m_event));
    }

    void record(const Kokkos::Cuda& exec) {
        if (m_event == nullptr) {
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
        }
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(m_event, exec.cuda_stream()));
        record_event(exec, m_event_id);
    }

    void wait() const {
        wait_event(m_event_id);
        if (cudaEventQuery(m_event) != cudaSuccess) {
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(m_event));
        }
    }
};

Event(const Kokkos::Cuda&) -> Event<Kokkos::Cuda>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_CUDA_EVENT_HPP
