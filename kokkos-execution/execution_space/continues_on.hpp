#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename ParentOp, typename ParentOpBase = typename ParentOp::base_t>
struct ContinuesOnReceiver {
    using receiver_concept = Impl::SubmittedReceiverTag;

    using parent_op_base_t = ParentOpBase;
    using exec_env_policy_t = typename ParentOp::exec_env_policy_t;

    ParentOp* parent_op;
    ParentOpBase* parent_op_base;

    constexpr explicit ContinuesOnReceiver(ParentOp* parent_op_) noexcept
        : parent_op(parent_op_)
        , parent_op_base(parent_op_) {
    }

    void set_value() && noexcept {
        using policy_t = typename ParentOp::completion_signal_policy_t;

        if constexpr (std::same_as<policy_t, Impl::SyncPolicy::InlineFenceExec>) {
            stdexec::set_value(std::move(parent_op_base->rcvr));
        }
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        stdexec::set_error(std::move(parent_op_base->rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        stdexec::set_stopped(std::move(parent_op_base->rcvr));
    }

    void submitted() && noexcept {
        static_assert(stdexec::__queryable_with<typename ParentOp::inner_opstate_t, Impl::get_exec_t>);
        using policy_t = typename ParentOp::completion_signal_policy_t;

        const auto& exec_from = parent_op->inner_opstate.query(Impl::get_exec).get();
        const auto& exec_to = parent_op_base->query(Impl::get_exec).get();
        try {
            if constexpr (std::same_as<policy_t, Impl::SubmittedPolicy::OrderOnExec>) {
                if (exec_from != exec_to) {
                    auto& event = parent_op->event_storage.emplace();
                    Impl::record(event, exec_from);
                    Impl::wait(exec_to, event);
                }
                std::move(parent_op_base->rcvr).submitted();
            } else if constexpr (std::same_as<policy_t, Impl::SyncPolicy::InlineFenceExec>) {
                exec_from.fence(
                    std::format(
                        "{}: continues_on", Kokkos::Impl::TypeInfo<std::remove_cvref_t<decltype(exec_from)>>::name()));
                stdexec::set_value(std::move(parent_op_base->rcvr));
            }
        } catch (...) {
            stdexec::set_error(std::move(parent_op_base->rcvr), std::current_exception());
        }
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> join_env_with_exec_t<
        exec_env_policy_t,
        stdexec::env_of_t<parent_op_base_t>,
        typename parent_op_base_t::execution_space
    > {
        return join_env_with_exec<exec_env_policy_t>(
            stdexec::get_env(*parent_op_base), parent_op_base->query(Impl::get_exec).get());
    }
};

template <typename Rcvr>
using continues_on_fwd_env_rcvr_t = stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>;

template <typename Sndr, typename Rcvr>
consteval auto select_continues_on_opstate_exec_env_policy() {
    if constexpr (
        std::same_as<
            stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, continues_on_fwd_env_rcvr_t<Rcvr>>,
            Domain
        >) {
        return WithExecEnvPolicy{};
    } else {
        return WithoutExecEnvPolicy{};
    }
}

template <typename Sndr, typename Rcvr>
using continues_on_opstate_exec_env_policy_t = decltype(select_continues_on_opstate_exec_env_policy<Sndr, Rcvr>());

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

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return stdexec::get_env(this->rcvr);
    }
};

template <stdexec::sender Sndr, stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnOpState
    : public Impl::Immovable
    , public ContinuesOnOpStateBase<Schd, Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = ContinuesOnOpStateBase<Schd, Rcvr>;
    using execution_space = typename base_t::execution_space;

    using exec_env_policy_t = continues_on_opstate_exec_env_policy_t<Sndr, Rcvr>;

    using rcvr_t = ContinuesOnReceiver<ContinuesOnOpState>;

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
};

template <stdexec::sender Sndr, stdexec::scheduler Schd>
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
