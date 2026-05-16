#ifndef KOKKOS_EXECUTION_IMPL_SUBMITTED_HPP
#define KOKKOS_EXECUTION_IMPL_SUBMITTED_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/event.hpp"
#include "kokkos-execution/impl/get_exec.hpp"

namespace Kokkos::Execution::Impl {

struct SubmittedReceiverTag : public stdexec::receiver_tag { };

template <typename Rcvr>
concept supports_submitted =
    std::derived_from<typename std::remove_cvref_t<Rcvr>::receiver_concept, SubmittedReceiverTag>;

template <Kokkos::ExecutionSpace Exec>
struct OrderOn {
    ExecutionSpaceRef<Exec> m_exec_ref;

    constexpr explicit OrderOn(const Exec& exec) noexcept
        : m_exec_ref(exec) {
    }

    constexpr auto exec() const & noexcept -> const Exec& {
        return m_exec_ref.get();
    }

    void synchronize(const std::string& label) const {
        exec().fence(label);
    }

    //! Operations submitted after this point on @p ExecTo depend on operations previously submitted on @p Exec.
    // think of how to specialize transitions
    // event creation useful only cuda to cuda, and so on
    // skip if exec_from == exec_to?
    // optimize for case with nothing has to be done, so don't do anything
    // submitted with orderedon is not needed
    template <Kokkos::ExecutionSpace ExecTo>
    OrderOn<ExecTo> transition(const ExecTo& exec_to, event_storage_t<Exec>& event_storage) const {
        auto& event = event_storage.emplace();
        record(event, exec());
        wait(exec_to, event);
        return {exec_to};
    }
};

template <typename Rcvr, typename Exec>
concept supports_submitted_order_on = supports_submitted<Rcvr> && requires(Rcvr& rcvr) {
    { rcvr.submitted(std::declval<OrderOn<Exec>>()) } noexcept;
};

template <stdexec::__is_instance_of<Event> EventType>
struct DependOn {
    EventType const * m_event;

    constexpr explicit DependOn(const EventType& event) noexcept
        : m_event(std::addressof(event)) {
    }

    constexpr auto event() const & noexcept -> const EventType& {
        return *m_event;
    }

    void synchronize() const {
        wait(event());
    }

    template <Kokkos::ExecutionSpace ExecTo>
    OrderOn<ExecTo> transition(const ExecTo& exec_to) const {
        wait(exec_to, event());
        return {exec_to};
    }
};

template <typename Rcvr, typename EventType>
concept supports_submitted_depend_on = supports_submitted<Rcvr> && requires(Rcvr& rcvr) {
    { rcvr.submitted(std::declval<DependOn<EventType>>()) } noexcept;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SUBMITTED_HPP
