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
    using mark_event_t = MarkEvent<Kokkos::Cuda>;

    cudaEvent_t event = nullptr;
    mutable bool arrived = false;

    Event() = default;

    explicit Event(const Kokkos::Cuda& exec) {
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
                KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(event));
            }
            event = std::exchange(other.event, nullptr);
            arrived = std::exchange(other.arrived, false);
        }
        return *this;
    }

    ~Event() {
        if (event != nullptr)
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventDestroy(event));
    }

    void record(const Kokkos::Cuda& exec) {
        if (event == nullptr) {
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(event, exec.cuda_stream()));
        arrived = false;
        mark_event_t::record(static_cast<void*>(event), exec);
    }

    void wait() const {
        if (!arrived) {
            mark_event_t::wait(static_cast<void*>(event));
            KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(event));
            arrived = true;
        }
    }
};

Event(const Kokkos::Cuda&) -> Event<Kokkos::Cuda>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_CUDA_EVENT_HPP
