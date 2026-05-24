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
struct HasNonBlockingDispatch<Kokkos::SYCL> : std::true_type { };

template <>
struct Event<Kokkos::SYCL> {
    mutable std::optional<sycl::event> m_event = std::nullopt;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept = delete;
    Event& operator=(Event&& other) noexcept = delete;

    /**
     * According to https://github.com/intel/llvm/issues/15606, it should semantically be
     * correct, whether the @c Kokkos::SYCL underlying queue is in-order or out-of-order.
     */
    void record(const Kokkos::SYCL& exec) {
        m_event = exec.sycl_queue().ext_oneapi_submit_barrier();
    }

    void wait() const {
        if (m_event.has_value()) {
            m_event->wait_and_throw();
            m_event = std::nullopt;
        }
    }

    [[nodiscard]]
    constexpr const sycl::event& sycl_event() const noexcept {
        KOKKOS_EXPECTS(m_event.has_value());
        return *m_event;
    }
};

template <>
void impl_wait(const Kokkos::SYCL& exec, const Event<Kokkos::SYCL>& event) {
    exec.sycl_queue().submit([&](sycl::handler& cgh) { cgh.depends_on(event.sycl_event()); });
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SYCL_EVENT_HPP
