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
struct HasNonBlockingDispatch<Kokkos::Cuda> : std::true_type { };

template <>
struct Event<Kokkos::Cuda> {
    cudaEvent_t m_event = nullptr;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&&) noexcept = delete;
    Event& operator=(Event&&) noexcept = delete;

    ~Event() {
        if (m_event != nullptr)
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(m_event));
    }

    void record(const Kokkos::Cuda& exec) {
        if (m_event == nullptr) {
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
        }
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(m_event, exec.cuda_stream()));
    }

    void wait() const {
        if (cudaEventQuery(m_event) != cudaSuccess) {
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(m_event));
        }
    }

    [[nodiscard]]
    constexpr cudaEvent_t cuda_event() const noexcept {
        return m_event;
    }
};

template <>
void impl_wait(const Kokkos::Cuda& exec, const Event<Kokkos::Cuda>& event) {
    KOKKOS_EXPECTS(bool(event.cuda_event()));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamWaitEvent(exec.cuda_stream(), event.cuda_event()));
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_CUDA_EVENT_HPP
