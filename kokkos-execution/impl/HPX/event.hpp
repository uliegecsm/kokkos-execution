#ifndef KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP

#include "Kokkos_Core.hpp"

/**
 * @file
 *
 * Specialization of @ref Kokkos::Execution::Impl::Event for @c Kokkos::Experimental::HPX.
 */

#if !defined(KOKKOS_ENABLE_IMPL_HPX_ASYNC_DISPATCH)
#    error "This is not supported."
#endif

namespace Kokkos::Execution::Impl {

template <>
struct SupportEvents<Kokkos::Experimental::HPX> : std::true_type { };

template <>
struct Event<Kokkos::Experimental::HPX> {
    std::shared_ptr<hpx::lcos::local::event> m_event = nullptr;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;

    explicit Event(const Kokkos::Experimental::HPX& exec) {
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
    ~Event() = default;

    void record(const Kokkos::Experimental::HPX& exec) {
        /**
         * Always allocate a new event. Otherwise, the sequence:
         * @code
         * event.record(exec_A);
         * event.record(exec_B);
         * event.wait();
         * @endcode
         * will be surprising to the user. Indeed, if the two successive calls to @ref record
         * would use the same @ref m_event, the call to @ref wait would potentially complete
         * due to @c exec_A, while the expectation is that it completes due to @c exec_B.
         */
        m_event = std::make_shared<hpx::lcos::local::event>();
        exec.impl_bulk_plain_erased<int>(
            false, /* force_synchronous */
            true,  /* is_light_weight_policy */
            [event = m_event](const auto) { event->set(); },
            1);
        record_event(exec, m_event_id);
    }

    void wait() const {
        wait_event(m_event_id);
        if (!m_event->occurred()) {
            m_event->wait();
        }
    }
};

Event(const Kokkos::Experimental::HPX&) -> Event<Kokkos::Experimental::HPX>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP
