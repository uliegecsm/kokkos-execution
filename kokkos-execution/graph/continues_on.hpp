#ifndef KOKKOS_EXECUTION_GRAPH_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_GRAPH_CONTINUES_ON_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/continues_on.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/state.hpp"
#include "kokkos-execution/impl/type_traits.hpp"

namespace Kokkos::Execution::GraphImpl {

//! Receiver for @c continues_on.
template <typename ParentOp, typename Env = stdexec::env_of_t<ParentOp>>
struct ContinuesOnReceiver : public Impl::Receiver<ParentOp, Env> {
    using exec_env_policy_t = typename ParentOp::exec_env_policy_t;

    [[nodiscard]]
    constexpr auto get_env() const noexcept
        -> ExecutionSpaceImpl::join_env_with_exec_t<exec_env_policy_t, Env, typename ParentOp::execution_space> {
        return ExecutionSpaceImpl::join_env_with_exec<exec_env_policy_t>(
            stdexec::get_env(*this->parent_op), Impl::get_exec(*this->parent_op).get());
    }
};

template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnOpStateBase {
    using execution_space = typename std::remove_cvref_t<Schd>::execution_space;

    Impl::State<execution_space>* state;
    Rcvr rcvr;

    constexpr explicit ContinuesOnOpStateBase(const Schd& schd, Rcvr rcvr_)
        noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
        : state(schd.state)
        , rcvr(std::move(rcvr_)) {
    }

    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> Impl::ExecutionSpaceRef<execution_space> {
        return Impl::ExecutionSpaceRef{state->exec};
    }

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->rcvr)
};

//! Operation state for @c stdexec::continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr, stdexec::receiver Rcvr>
struct ContinuesOnOpState
    : public Impl::Immovable
    , public ContinuesOnOpStateBase<Schd, Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = ContinuesOnOpStateBase<Schd, Rcvr>;
    using execution_space = typename base_t::execution_space;

    using exec_env_policy_t = std::conditional_t<
        std::same_as<stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, stdexec::env_of_t<Rcvr>>, Domain>,
        ExecutionSpaceImpl::WithExecEnvPolicy,
        ExecutionSpaceImpl::WithoutExecEnvPolicy
    >;

    using rcvr_t = ContinuesOnReceiver<ContinuesOnOpState, stdexec::env_of_t<Rcvr>>;
    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    using completion_signal_policy_t =
        Impl::ContinuesOn::completion_signal_policy_t<inner_opstate_t, execution_space, Rcvr>;

    using dependency_t = Impl::ContinuesOn::dependency_for_t<inner_opstate_t, execution_space>;

    [[no_unique_address]]
    std::optional<dependency_t> dependency{};
    inner_opstate_t inner_opstate;

    ContinuesOnOpState(Schd&& schd, Sndr&& sndr, Rcvr rcvr) noexcept( // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        std::is_nothrow_constructible_v<base_t, Schd&&, Rcvr&&> && stdexec::__nothrow_connectable<Sndr&&, rcvr_t>)
        : base_t(std::forward<Schd>(schd), std::move(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
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
};

//! Specialization for @ref Kokkos::Execution::GraphImpl::ContinuesOnOpState. @todo To be removed and done properly.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::ContinuesOnOpState>
    && std::same_as<typename OpState::execution_space, Exec>)
struct GraphOperationStateFor<OpState, Exec> : public std::true_type { };

//! Sender for @c stdexec::continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <typename Self, typename Rcvr>
    using connect_result_t =
        ContinuesOnOpState<stdexec::__copy_cvref_t<Self, Schd>, stdexec::__copy_cvref_t<Self, Sndr>, Rcvr>;

    template <typename Self, typename Rcvr>
    static constexpr bool is_nothrow_connectable_v = std::is_nothrow_constructible_v<
        connect_result_t<Self, Rcvr>,
        KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, schd),
        KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr),
        Rcvr&&
    >;

    template <stdexec::__decays_to<ContinuesOnSender> Self, stdexec::receiver Rcvr>
    STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(is_nothrow_connectable_v<Self, Rcvr>) -> connect_result_t<Self, Rcvr> {
        return {
            KOKKOS_EXECUTION_IMPL_FORWARD_THIS(Self, self).schd,
            KOKKOS_EXECUTION_IMPL_FORWARD_THIS(Self, self).sndr,
            std::move(rcvr)};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Schd schd; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::continues_on_t> {
    template <typename Env, stdexec::scheduler Schd, stdexec::sender Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ContinuesOnSender<Schd, Sndr>, Schd&&, Sndr&&>)
            -> ContinuesOnSender<Schd, Sndr> {
        return {.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_CONTINUES_ON_HPP
