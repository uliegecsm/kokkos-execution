#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/event.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Receiver for @c continues_on.
template <typename ExecEnvPolicy, typename ParentOp, Kokkos::ExecutionSpace ExecTo, typename Env = stdexec::env_of_t<ParentOp>>
struct ContinuesOnReceiver {
    using receiver_concept = Impl::SubmittedReceiverTag;

    ParentOp* parent_op;

    void set_value() && noexcept {
        parent_op->complete(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& error) && noexcept {
        parent_op->complete(stdexec::set_error, std::forward<Error>(error));
    }

    void set_stopped() && noexcept {
        parent_op->complete(stdexec::set_stopped);
    }

    template <Kokkos::ExecutionSpace ExecFrom>
    void submitted(Impl::OrderOn<ExecFrom> order_on) const & noexcept {
        parent_op->submit(order_on);
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept
        -> join_env_with_exec_t<ExecEnvPolicy, Env, ExecTo> {
        return join_env_with_exec<ExecEnvPolicy>(stdexec::get_env(*parent_op), Impl::get_exec(*parent_op).get());
    }
};

template <typename Sndr, typename Rcvr>
consteval auto select_continues_on_opstate_exec_env_policy() {
    if constexpr (std::same_as<
            stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>>,
            Domain
        >) {
        return WithExecEnvPolicy{};
    } else {
        return WithoutExecEnvPolicy{};
    }
}

template <typename Sndr, typename Rcvr>
using continues_on_opstate_exec_env_policy_t = decltype(select_continues_on_opstate_exec_env_policy<Sndr, Rcvr>());

template <typename ExecTo, typename Rcvr>
consteval auto select_continues_on_opstate_completion_signal_policy() {
    if constexpr (Impl::supports_submitted<Rcvr>) {
        if constexpr (
            stdexec::__is_instance_of<Rcvr, Impl::SyncWait::Receiver>
            || stdexec::__is_instance_of<Rcvr, ScheduleFromReceiver>
            || stdexec::__is_instance_of<Rcvr, Impl::Receiver>) {
            return Impl::SubmittedPolicy::OrderOnExec{};
        } else {
            return Impl::SubmittedPolicy::DependOnEvent{};
        }
    } else {
        if constexpr (
            Impl::has_non_blocking_dispatch<ExecTo>
            && stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, stdexec::get_delegation_scheduler_t>) {
            return Impl::SyncPolicy::ScheduleWaitEvent{};
        } else {
            return Impl::SyncPolicy::InlineFenceExec{};
        }
    }
}

template <typename Exec, typename Rcvr>
using continues_on_opstate_completion_signal_policy_t = decltype(select_continues_on_opstate_completion_signal_policy<Exec, Rcvr>());

template <stdexec::sender Sndr, stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnOpstate {
    using operation_state_concept = stdexec::operation_state_tag;

    using execution_space = typename Schd::execution_space;

    using exec_env_policy_t = continues_on_opstate_exec_env_policy_t<Sndr, Rcvr>;
    using rcvr_t = ContinuesOnReceiver<exec_env_policy_t, ContinuesOnOpstate, execution_space, stdexec::env_of_t<Rcvr>>;

    using completion_signal_policy_t = continues_on_opstate_completion_signal_policy_t<execution_space, Rcvr>;
    using completion_signal_t = Impl::CompletionSignal<completion_signal_policy_t, execution_space, Rcvr>;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    Schd schd;
    Impl::event_storage_t<execution_space> event_storage;
    completion_signal_t completion_signal;
    inner_opstate_t inner_opstate;

    constexpr explicit ContinuesOnOpstate(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Schd&& schd_, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr)
        : schd(std::forward<Schd>(schd_))
        , event_storage()
        , completion_signal(std::move(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    void complete(stdexec::set_value_t) noexcept {
        completion_signal.propagate(this->query(Impl::get_exec).get());
    }

    template <typename Error>
    void complete(stdexec::set_error_t, Error&& error) noexcept {
        stdexec::set_error(std::move(completion_signal.rcvr), std::forward<Error>(error));
    }

    void complete(stdexec::set_stopped_t) noexcept {
        stdexec::set_stopped(std::move(completion_signal.rcvr));
    }

    template <Kokkos::ExecutionSpace ExecFrom>
    void submit(Impl::OrderOn<ExecFrom> order_on) noexcept {
        const auto& exec_to = this->query(Impl::get_exec).get();        
        try {
            order_on.transition(exec_to, this->event_storage);
        } catch (...) {
            this->complete(stdexec::set_error, std::current_exception());
            return;
        }
        completion_signal.propagate(exec_to);
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }

    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> Impl::ExecutionSpaceRef<execution_space> {
        return Impl::ExecutionSpaceRef<execution_space>{schd.state->exec};
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return stdexec::get_env(this->completion_signal.rcvr);
    }
};

//! Sender for @c continues_on.
template <stdexec::sender Sndr, stdexec::scheduler Schd>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <stdexec::receiver Rcvr>
    auto connect(Rcvr rcvr) && -> ContinuesOnOpstate<Sndr, Schd, Rcvr> {
        return ContinuesOnOpstate<Sndr, Schd, Rcvr>(std::forward<Sndr>(sndr), std::forward<Schd>(schd), std::move(rcvr));
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Schd schd;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::continues_on_t> {
    template <typename Env, stdexec::__is_instance_of<Scheduler> Schd, stdexec::sender Sndr>
    auto operator()(const Env&, stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ContinuesOnSender<Sndr, Schd>, Sndr&&, Schd&&>) {
        return ContinuesOnSender<Sndr, Schd>{.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
