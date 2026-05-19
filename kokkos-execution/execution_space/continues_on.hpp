#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Receiver for @c continues_on.
template <typename ExecEnvPolicy, typename ParentOp, typename Env>
struct ContinuesOnReceiver {
    using receiver_concept = Impl::SubmittedReceiverTag;

    ParentOp* parent_op;

    void set_value() && noexcept {
        stdexec::set_value(std::move(parent_op->rcvr));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        stdexec::set_error(std::move(parent_op->rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        stdexec::set_stopped(std::move(parent_op->rcvr));
    }

    void submitted() && noexcept {
        parent_op->submitted();
    }

    [[nodiscard]]
    constexpr auto
        get_env() const noexcept -> join_env_with_exec_t<ExecEnvPolicy, Env, typename ParentOp::execution_space> {
        return join_env_with_exec<ExecEnvPolicy>(stdexec::get_env(*parent_op), parent_op->query(Impl::get_exec).get());
    }
};


template <typename InnerOp, typename ExecTo, typename Rcvr>
concept continues_on_signals_submitted =
    Impl::signals_submitted<InnerOp>
    && std::same_as<typename stdexec::__query_result_t<InnerOp, Impl::get_exec_t>::execution_space, ExecTo>
    && Impl::supports_submitted_order_on<Rcvr>;

template <typename InnerOp, typename ExecTo, typename Rcvr>
consteval auto select_continues_on_opstate_completion_signal_policy() {
    if constexpr (continues_on_signals_submitted<InnerOp, ExecTo, Rcvr>) {
        return Impl::SubmittedPolicy::OrderOnExec{};
    } else {
        return Impl::SyncPolicy::InlineFenceExec{};
    }
}

template <typename InnerOp, typename ExecTo, typename Rcvr>
using continues_on_opstate_completion_signal_policy_t =
    decltype(select_continues_on_opstate_completion_signal_policy<InnerOp, ExecTo, Rcvr>());

template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnOpStateBase {
    using execution_space = typename Schd::execution_space;

    Rcvr rcvr;
    Schd schd;

    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> Impl::ExecutionSpaceRef<execution_space> {
        return Impl::ExecutionSpaceRef<execution_space>{schd.state->exec};
    }

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->rcvr)
};

//! Operation state for @c stdexec::continues_on.
template <stdexec::sender Sndr, stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnOpState
    : public Impl::Immovable
    , public ContinuesOnOpStateBase<Schd, Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = ContinuesOnOpStateBase<Schd, Rcvr>;
    using execution_space = typename base_t::execution_space;

    using exec_env_policy_t = decltype([]() {
        if constexpr (
            std::same_as<
                stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, stdexec::env_of_t<Rcvr>>,
                Domain
            >) {
            return WithExecEnvPolicy{};
        } else {
            return WithoutExecEnvPolicy{};
        }
    }());

    using rcvr_t = ContinuesOnReceiver<exec_env_policy_t, ContinuesOnOpState, stdexec::env_of_t<Rcvr>>;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;
    using completion_signal_policy_t =
        continues_on_opstate_completion_signal_policy_t<inner_opstate_t, execution_space, Rcvr>;

    Impl::event_storage_t<execution_space> event_storage;
    inner_opstate_t inner_opstate;

    constexpr explicit ContinuesOnOpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Schd&& schd, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr)
        noexcept(
            std::is_nothrow_constructible_v<base_t, Rcvr&&, Schd&&> && stdexec::__nothrow_connectable<Sndr&&, rcvr_t>)
        : base_t(std::move(rcvr), std::forward<Schd>(schd))
        , event_storage()
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }

    void submitted() & requires Impl::signals_submitted<inner_opstate_t>
    {
        static_assert(stdexec::__queryable_with<inner_opstate_t, Impl::get_exec_t>);

        const auto& exec_from = inner_opstate.query(Impl::get_exec).get();
        const auto& exec_to = this->schd.state->exec;

        try {
            if constexpr (std::same_as<completion_signal_policy_t, Impl::SubmittedPolicy::OrderOnExec>) {
                if (exec_from != exec_to) {
                    auto& event = event_storage.emplace();
                    Impl::record(event, exec_from);
                    Impl::wait(exec_to, event);
                }
                std::move(this->rcvr).submitted();
            } else if constexpr (std::same_as<completion_signal_policy_t, Impl::SyncPolicy::InlineFenceExec>) {
                exec_from.fence(
                    std::format(
                        "{}: continues_on", Kokkos::Impl::TypeInfo<std::remove_cvref_t<decltype(exec_from)>>::name()));
                stdexec::set_value(std::move(this->rcvr));
            }
        } catch (...) {
            stdexec::set_error(std::move(this->rcvr), std::current_exception());
        }
    }
};

//! Sender for @c continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<ContinuesOnOpState<Sndr, Schd, Rcvr>, Sndr&&, Schd&&, Rcvr&&>)
        -> ContinuesOnOpState<Sndr, Schd, Rcvr> {
        return ContinuesOnOpState<Sndr, Schd, Rcvr>{
            std::forward<Sndr>(sndr), std::forward<Schd>(schd), std::move(rcvr)};
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Schd schd;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::continues_on_t> {
    template <typename Env, stdexec::__is_instance_of<Scheduler> Schd, stdexec::sender Sndr>
    auto operator()(const Env&, stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ContinuesOnSender<Schd, Sndr>, Schd&&, Sndr&&>)
            -> ContinuesOnSender<Schd, Sndr> {
        return {.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
