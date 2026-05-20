#ifndef KOKKOS_EXECUTION_GRAPH_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_GRAPH_SCHEDULE_FROM_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/schedule_from.hpp"
#include "kokkos-execution/impl/type_traits.hpp"

namespace Kokkos::Execution::GraphImpl {

template <stdexec::receiver Rcvr>
struct ScheduleFromOpStateBase {
    Rcvr rcvr;

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->rcvr)
};

//! Operation state for @c stdexec::schedule_from.
template <Kokkos::ExecutionSpace Exec, stdexec::sender Sndr, stdexec::receiver Rcvr>
struct ScheduleFromOpState
    : public Impl::Immovable
    , public ScheduleFromOpStateBase<Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using execution_space = Exec;

    using base_t = ScheduleFromOpStateBase<Rcvr>;

    using rcvr_t = Impl::Receiver<ScheduleFromOpState, stdexec::env_of_t<Rcvr>>;
    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    using completion_signal_policy_t = Impl::ScheduleFrom::completion_signal_policy_t<inner_opstate_t, Rcvr>;

    inner_opstate_t inner_opstate;

    ScheduleFromOpState(Sndr&& sndr, Rcvr rcvr) noexcept( // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        stdexec::__nothrow_connectable<Sndr&&, rcvr_t>)
        : base_t(std::move(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> decltype(auto)
        requires stdexec::__queryable_with<inner_opstate_t, Impl::get_exec_t>
    {
        return Impl::get_exec(inner_opstate);
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

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }

    //! Stay in the @ref Kokkos::Execution::GraphImpl::Domain.
    void submit() noexcept requires std::same_as<completion_signal_policy_t, Impl::SubmittedPolicy::OrderOnExec>
    {
        std::move(this->rcvr).submitted();
    }

    //! Transition to another domain.
    void submit() noexcept requires std::same_as<completion_signal_policy_t, Impl::SyncPolicy::InlineFenceExec>
    {
        try {
            this->query(Impl::get_exec)
                .get()
                .fence(std::string(Impl::dispatch_label<Impl::exec_of_t<decltype(*this)>, ": schedule_from">()));
            stdexec::set_value(std::move(this->rcvr));
        } catch (...) {
            stdexec::set_error(std::move(this->rcvr), std::current_exception());
        }
    }
};

//! Specialization for @ref Kokkos::Execution::GraphImpl::ScheduleFromOpState. @todo To be removed and done properly.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::ScheduleFromOpState>
    && std::same_as<typename OpState::execution_space, Exec>)
struct GraphOperationStateFor<OpState, Exec> : public std::true_type { };

//! Sender for @c stdexec::schedule_from.
template <typename Exec, typename Sndr>
struct ScheduleFromSender {
    using sender_concept = stdexec::sender_tag;

    using execution_space = Exec;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender)

    template <typename Self, typename Rcvr>
    using connect_result_t = ScheduleFromOpState<execution_space, stdexec::__copy_cvref_t<Self, Sndr>, Rcvr>;

    template <typename Self, typename Rcvr>
    static constexpr bool is_nothrow_connectable_v = std::is_nothrow_constructible_v<
        connect_result_t<Self, Rcvr>,
        KOKKOS_EXECUTION_IMPL_MEMBER_CVREF_T(Self, sndr),
        Rcvr&&
    >;

    template <stdexec::__decays_to<ScheduleFromSender> Self, stdexec::receiver Rcvr>
    STDEXEC_EXPLICIT_THIS_BEGIN(
        auto connect)(this Self&& self, Rcvr rcvr) // NOLINT(cppcoreguidelines-missing-std-forward)
        noexcept(is_nothrow_connectable_v<Self, Rcvr>) -> connect_result_t<Self, Rcvr> {
        return {std::forward<Self>(self).sndr, std::move(rcvr)};
    }
    STDEXEC_EXPLICIT_THIS_END(connect)

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::schedule_from_t> {
    template <typename Env, stdexec::sender Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::schedule_from_t, stdexec::__ignore, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ScheduleFromSender<Impl::exec_of_t<Sndr, Env>, Sndr>, Sndr&&>) {
        if constexpr (graph_completing_sender<Sndr, Env>) {
            return ScheduleFromSender<Impl::exec_of_t<Sndr, Env>, Sndr>{.sndr = std::forward<Sndr>(sndr)};
        } else {
            return no_graph_scheduler_in_env<stdexec::schedule_from_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_SCHEDULE_FROM_HPP
