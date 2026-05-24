#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/dependency.hpp"
#include "kokkos-execution/impl/empty.hpp"
#include "kokkos-execution/impl/receiver.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Receiver for @c continues_on.
template <typename ParentOp, typename Env = stdexec::env_of_t<ParentOp>>
struct ContinuesOnReceiver : public Impl::Receiver<ParentOp, Env> {
    using exec_env_policy_t = typename ParentOp::exec_env_policy_t;

    [[nodiscard]]
    constexpr auto
        get_env() const noexcept -> join_env_with_exec_t<exec_env_policy_t, Env, typename ParentOp::execution_space> {
        return join_env_with_exec<exec_env_policy_t>(
            stdexec::get_env(*this->parent_op), Impl::get_exec(*this->parent_op).get());
    }
};

template <typename InnerOp, typename ExecTo, typename Rcvr>
concept continues_on_signals_submitted = Impl::signals_submitted<InnerOp>
                                      && std::same_as<ExecTo, Impl::exec_of_t<InnerOp>>
                                      && Impl::supports_submitted_order_on<Rcvr>;

template <typename InnerOp, typename ExecTo, typename Rcvr>
struct DependencyFor {
    using type = Impl::Empty;
};

template <typename InnerOp, typename ExecTo, typename Rcvr>
requires Impl::signals_submitted<InnerOp>
struct DependencyFor<InnerOp, ExecTo, Rcvr> {
    using type = Impl::Dependency<ExecTo, Impl::exec_of_t<InnerOp>>;
};

template <typename InnerOp, typename ExecTo, typename Rcvr>
using dependency_for_t = typename DependencyFor<InnerOp, ExecTo, Rcvr>::type;

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

template <stdexec::scheduler Schd, stdexec::sender Sndr, stdexec::receiver Rcvr>
struct ContinuesOnOpState
    : public Impl::Immovable
    , public ContinuesOnOpStateBase<Schd, Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = ContinuesOnOpStateBase<Schd, Rcvr>;
    using execution_space = typename base_t::execution_space;

    using exec_env_policy_t = std::conditional_t<
        std::same_as<stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, stdexec::env_of_t<Rcvr>>, Domain>,
        WithExecEnvPolicy,
        WithoutExecEnvPolicy
    >;

    using rcvr_t = ContinuesOnReceiver<ContinuesOnOpState, stdexec::env_of_t<Rcvr>>;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    using completion_signal_policy_t = std::conditional_t<
        continues_on_signals_submitted<inner_opstate_t, execution_space, Rcvr>,
        Impl::SubmittedPolicy::OrderOnExec,
        Impl::SyncPolicy::InlineFenceExec
    >;

    using dependency_t = dependency_for_t<inner_opstate_t, execution_space, Rcvr>;

    [[no_unique_address]]
    std::optional<dependency_t> dependency{};
    inner_opstate_t inner_opstate;

    constexpr explicit ContinuesOnOpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Schd&& schd, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr)
        noexcept(
            std::is_nothrow_constructible_v<base_t, Rcvr&&, Schd&&> && stdexec::__nothrow_connectable<Sndr&&, rcvr_t>)
        : base_t(std::move(rcvr), std::forward<Schd>(schd))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    void complete(stdexec::set_value_t) noexcept {
        stdexec::set_value(std::move(this->rcvr));
    }

    template <typename Error>
    void complete(stdexec::set_error_t, Error&& error) noexcept {
        stdexec::set_error(std::move(this->rcvr), std::forward<Error>(error));
    }

    void complete(stdexec::set_stopped_t) noexcept {
        stdexec::set_stopped(std::move(this->rcvr));
    }

    void submit() noexcept {
        const auto& exec_from = Impl::get_exec(this->inner_opstate).get();
        const auto& exec_to = Impl::get_exec(*this).get();
        this->dependency.emplace(exec_to, exec_from);
        try {
            if constexpr (std::same_as<completion_signal_policy_t, Impl::SubmittedPolicy::OrderOnExec>) {
                std::move(this->rcvr).submitted();
            } else {
                static_assert(std::same_as<completion_signal_policy_t, Impl::SyncPolicy::InlineFenceExec>);
                stdexec::set_value(std::move(this->rcvr));
            }
        } catch (...) {
            stdexec::set_error(std::move(this->rcvr), std::current_exception());
        }
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }
};

//! Sender for @c continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<ContinuesOnOpState<Schd, Sndr, Rcvr>, Sndr&&, Schd&&, Rcvr&&>)
        -> ContinuesOnOpState<Schd, Sndr, Rcvr> {
        return ContinuesOnOpState<Schd, Sndr, Rcvr>{
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
