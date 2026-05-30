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
struct HasNonBlockingDispatch<Kokkos::Experimental::HPX> : std::true_type { };

/**
 * This implementation assumes that once a call to @ref record is made,
 * a call to @ref wait or @ref Kokkos::Execution::Impl::wait follows at some point.
 *
 * Indeed, since @ref record steals the @c Kokkos::Experimental::HPX instance sender, the only way to wait for the
 * completion of the operations enqueued *before* the call to @ref record is to call @ref wait or @ref Kokkos::Execution::Impl::wait,
 * *i.e.* stealing the instance sender has decoupled its operations from the @c Kokkos::Experimental::HPX instance.
 *
 * Therefore, after calling @ref record, the ordering relationship is no longer carried by the execution space instance;
 * subsequently enqueued operations that need to be ordered after the previously submitted operations must depend on the event.
 */
template <>
struct Event<Kokkos::Experimental::HPX> {
    using sndr_t = hpx::execution::experimental::unique_any_sender<>;

    mutable std::optional<sndr_t> m_sender;
    uint64_t m_event_id = invalid_event_id;

    Event() = default;
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&&) noexcept = delete;
    Event& operator=(Event&&) noexcept = delete;
    ~Event() = default;

    //! Steal the head of the underlying sender, and store it in @ref m_sender.
    void record(const Kokkos::Experimental::HPX& exec) {
        KOKKOS_EXPECTS(!m_sender.has_value());

        auto& instance_data = exec.impl_get_instance_data();

        const std::scoped_lock<hpx::spinlock> lock(instance_data.m_sender_mutex);

        m_sender = std::move(instance_data.m_sender);

        instance_data.m_sender = sndr_t(hpx::execution::experimental::just());
    }

    /**
     * If there is no explicit call to the @c Kokkos::Experimental::HPX instance @c fence anywhere, there will
     * be leaks due to circular dependencies, much like described in https://github.com/kokkos/kokkos/pull/8992.
     *
     * The initial approach from https://github.com/uliegecsm/kokkos-execution/pull/106, based on using
     * @c hpx::lcos::local::event, has been discarded because it would never clean the underlying sender through a
     * @c fence.
     *
     * The current implementation sync-waits the value of @ref m_sender,
     * and then resets it to @c std::nullopt so that all the previous
     * operations in the sender get a chance to clean up.
     */
    void wait() const {
        if (m_sender.has_value()) {
            hpx::this_thread::experimental::sync_wait(std::move(*m_sender));
            m_sender = std::nullopt;
        }
    }
};

//! Enqueue a call to @ref Event::wait into the underlying sender chain.
template <>
void impl_wait(const Kokkos::Experimental::HPX& exec, const Event<Kokkos::Experimental::HPX>& event) {
    auto& instance_data = exec.impl_get_instance_data();

    const std::scoped_lock<hpx::spinlock> lock(instance_data.m_sender_mutex);

    auto& sndr = instance_data.m_sender;
    sndr = hpx::execution::experimental::unique_any_sender<>{
        hpx::execution::experimental::when_all(std::move(sndr), std::move(*event.m_sender))};

    //! Mark the event as consumed.
    event.m_sender = std::nullopt;
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HPX_EVENT_HPP
