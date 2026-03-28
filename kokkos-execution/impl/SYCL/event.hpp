#ifndef KOKKOS_EXECUTION_IMPL_SYCL_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_SYCL_EVENT_HPP

#include "Kokkos_Core.hpp"

/**
 * @file
 *
 * Specialization of @ref Kokkos::Execution::Impl::Event for @c Kokkos::SYCL.
 */

namespace Kokkos::Execution::Impl {

template <>
struct SupportEvents<Kokkos::SYCL> : std::true_type { };

template <>
struct Event<Kokkos::SYCL> {
    mutable std::optional<sycl::event> m_event = std::nullopt;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;

    explicit Event(const Kokkos::SYCL& exec) {
        record(exec);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept
        : m_event(std::move(other.m_event))
        , m_event_id(std::exchange(other.m_event_id, invalid_event_id)) {
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            m_event = std::move(other.m_event);
            m_event_id = std::exchange(other.m_event_id, invalid_event_id);
        }
        return *this;
    }

    /**
     * According to https://github.com/intel/llvm/issues/15606, it should semantically be
     * correct, whether the @c Kokkos::SYCL underlying queue is in-order or out-of-order.
     */
    void record(const Kokkos::SYCL& exec) {
        m_event = exec.sycl_queue().ext_oneapi_submit_barrier();
        record_event(exec, m_event_id);
    }

    void wait() const {
        wait_event(m_event_id);
        if (m_event.has_value()) {
            m_event->wait_and_throw();
            m_event = std::nullopt;
        }
    }
};

Event(const Kokkos::SYCL&) -> Event<Kokkos::SYCL>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SYCL_EVENT_HPP
