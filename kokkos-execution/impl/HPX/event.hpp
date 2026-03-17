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
    mutable std::optional<Kokkos::Experimental::HPX> m_exec = std::nullopt;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;

    explicit Event(const Kokkos::Experimental::HPX& exec) {
        record(exec);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept
        : m_exec(std::move(other.m_exec))
        , m_event_id(std::exchange(other.m_event_id, invalid_event_id)) {
    }
    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            m_exec = std::move(other.m_exec);
            m_event_id = std::exchange(other.m_event_id, invalid_event_id);
        }
        return *this;
    }
    ~Event() = default;

    void record(const Kokkos::Experimental::HPX& exec) {
        m_exec = exec;
        record_event(exec, m_event_id);
    }

    /**
     * If there is no explicit call to the @c Kokkos::Experimental::HPX instance @c fence anywhere, there will
     * be leaks due to circular dependencies, much like described in https://github.com/kokkos/kokkos/pull/8992.
     *
     * The initial approach from https://github.com/uliegecsm/kokkos-execution/pull/106, based on using
     * @c hpx::lcos::local::event, has been discarded because it would never clean the underlying sender through a
     * @c fence.
     *
     * Yet, it has been decided to mimic
     * https://github.com/kokkos/kokkos/blob/91584fc13aaf09330bc391466dbae0249895291f/core/src/HPX/Kokkos_HPX.hpp#L139-L156
     * so that there is no @c fence event recorded.
     *
     * Even though @ref wait may therefore synchronize even the work pushed to @ref m_exec after the call to @ref record,
     * it's the most reasonable way forward for now.
     */
    void wait() const {
        wait_event(m_event_id);
        if (m_exec.has_value()) {
            auto& instance_data = m_exec->impl_get_instance_data();

            {
                const std::lock_guard<hpx::spinlock> lock(instance_data.m_sender_mutex);

                auto& sndr = instance_data.m_sender;
                hpx::this_thread::experimental::sync_wait(std::move(sndr));
                sndr = hpx::execution::experimental::unique_any_sender<>(hpx::execution::experimental::just());
            }

            m_exec = std::nullopt;
        }
    }
};

Event(const Kokkos::Experimental::HPX&) -> Event<Kokkos::Experimental::HPX>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP
