#ifndef KOKKOS_EXECUTION_IMPL_SUBMITTED_HPP
#define KOKKOS_EXECUTION_IMPL_SUBMITTED_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/completion_signal.hpp"
#include "kokkos-execution/impl/event.hpp"

namespace Kokkos::Execution::Impl {

struct SubmittedReceiverTag : public stdexec::receiver_tag { };

template <typename Rcvr>
concept supports_submitted =
    stdexec::receiver<Rcvr>
    && std::derived_from<typename std::remove_cvref_t<Rcvr>::receiver_concept, SubmittedReceiverTag>;

template <typename Rcvr>
concept supports_submitted_order_on = supports_submitted<Rcvr> && requires(Rcvr&& rcvr) {
    { std::move(rcvr).submitted() } noexcept;
};

template <typename Rcvr, typename... Execs>
concept supports_submitted_depend_on = supports_submitted<Rcvr> && (Kokkos::ExecutionSpace<Execs> && ...)
                                    && sizeof...(Execs) > 0
                                    && requires(Rcvr&& rcvr, OptionalConstEventRef<Execs>... deps) {
                                           { std::move(rcvr).submitted(deps...) } noexcept;
                                       };

struct SubmittedOperationStateTag : public stdexec::operation_state_tag { };

template <typename Op>
concept signals_submitted =
    stdexec::operation_state<Op>
    && std::derived_from<typename std::remove_cvref_t<Op>::operation_state_concept, SubmittedOperationStateTag>
    && requires { typename std::remove_cvref_t<Op>::completion_signal_policy_t; }
    && submitted_policy<typename std::remove_cvref_t<Op>::completion_signal_policy_t>;

template <typename Op>
concept signals_submitted_order_on =
    signals_submitted<Op>
    && std::same_as<typename std::remove_cvref_t<Op>::completion_signal_policy_t, SubmittedPolicy::OrderOnExec>;

template <typename Op>
concept signals_submitted_depend_on =
    signals_submitted<Op>
    && std::same_as<typename std::remove_cvref_t<Op>::completion_signal_policy_t, SubmittedPolicy::DependOnEvent>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SUBMITTED_HPP
