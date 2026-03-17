#ifndef KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP
#define KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP

#include "Kokkos_Core.hpp"

/**
 * @file
 *
 * Specialization of @ref Kokkos::Execution::Impl::Event for @c Kokkos::Experimental::HPX.
 */

namespace Kokkos::Execution::Impl {

template <>
struct SupportEvents<Kokkos::Experimental::HPX> : std::true_type { };

template <>
struct Event<Kokkos::Experimental::HPX> {
    using mark_event_t = MarkEvent<Kokkos::Experimental::HPX>;

    mutable std::optional<hpx::execution::experimental::any_sender<>> m_sender = std::nullopt;
    void* m_id = nullptr; //! Used to keep a stable event ID across moves.

    Event() = default;

    explicit Event(const Kokkos::Experimental::HPX& exec) {
        record(exec);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept
        : m_sender(std::move(other.m_sender))
        , m_id(std::exchange(other.m_id, nullptr)) {
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            m_sender = std::move(other.m_sender);
            m_id = std::exchange(other.m_id, nullptr);
        }
        return *this;
    }
    ~Event() = default;

    void record(const Kokkos::Experimental::HPX& exec) {
        m_sender = exec.get_sender();
        m_id = (void*) std::addressof(*m_sender);
        mark_event_t::record(m_id, exec);
    }

    //! @note Consumes @ref m_sender.
    void wait() const {
        if (m_sender.has_value()) {
            mark_event_t::wait(m_id);
            hpx::this_thread::experimental::sync_wait(*std::exchange(m_sender, std::nullopt));
        }
    }
};

Event(const Kokkos::Experimental::HPX&) -> Event<Kokkos::Experimental::HPX>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP
