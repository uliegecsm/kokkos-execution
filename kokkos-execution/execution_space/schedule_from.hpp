#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/dependency.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/schedule_from.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename ParentOp, typename Env = stdexec::env_of_t<ParentOp>>
struct ScheduleFromReceiver : public Impl::Receiver<ParentOp, Env> {
    using exec_env_policy_t = typename ParentOp::exec_env_policy_t;

    using Impl::Receiver<ParentOp, Env>::submitted;

    template <Kokkos::ExecutionSpace... Execs>
    requires(sizeof...(Execs) > 0)
    void submitted(Impl::OptionalConstEventRef<Execs>... deps) && noexcept {
        this->parent_op->submit(deps...);
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> extend_env_with_exec_t<exec_env_policy_t, Env> {
        return extend_env_with_exec<exec_env_policy_t>(stdexec::get_env(*this->parent_op));
    }
};

template <stdexec::receiver Rcvr>
struct ScheduleFromOpStateBase {
    Rcvr rcvr;

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->rcvr)
};

template <stdexec::sender Sndr, stdexec::receiver Rcvr>
struct ScheduleFromOpState
    : public Impl::Immovable
    , public ScheduleFromOpStateBase<Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = ScheduleFromOpStateBase<Rcvr>;

    using exec_env_policy_t = std::conditional_t<
        execution_space_completing_sender<Sndr, stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>>,
        WithoutExecEnvPolicy,
        extend_env_with_exec_policy_t<stdexec::env_of_t<Rcvr>>
    >;

    using rcvr_t = ScheduleFromReceiver<ScheduleFromOpState, stdexec::env_of_t<Rcvr>>;

    using inner_op_state_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    using completion_signal_policy_t = Impl::ScheduleFrom::completion_signal_policy_t<inner_op_state_t, Rcvr>;

    inner_op_state_t inner_op_state;

    constexpr explicit ScheduleFromOpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr)
        noexcept(std::is_nothrow_constructible_v<base_t, Rcvr&&> && stdexec::__nothrow_connectable<Sndr&&, rcvr_t>)
        : base_t(std::move(rcvr))
        , inner_op_state(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
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

    //! Stay in the @ref Kokkos::Execution::ExecutionSpaceImpl::Domain.
    template <typename... Args>
    void submit(Args&&... args) noexcept requires Impl::submitted_policy<completion_signal_policy_t>
    {
        std::move(this->rcvr).submitted(std::forward<Args>(args)...);
    }

    //! Transition to another domain.
    void submit() noexcept requires std::same_as<completion_signal_policy_t, Impl::SyncPolicy::InlineFenceExec>
    {
        try {
            this->query(Impl::get_exec)
                .get()
                .fence(std::string(Impl::dispatch_label<Impl::exec_of_t<decltype(*this)>, ": schedule_from">()));
        } catch (...) {
            stdexec::set_error(std::move(this->rcvr), std::current_exception());
            return;
        }
        stdexec::set_value(std::move(this->rcvr));
    }

    template <Kokkos::ExecutionSpace... Execs>
    void submit(Impl::OptionalConstEventRef<Execs>... deps) noexcept
        requires(sizeof...(Execs) > 0) && std::same_as<completion_signal_policy_t, Impl::SyncPolicy::InlineFenceExec>
    {
        try {
            Impl::wait_on(deps...);
        } catch (...) {
            stdexec::set_error(std::move(this->rcvr), std::current_exception());
            return;
        }
        stdexec::set_value(std::move(this->rcvr));
    }

    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> decltype(auto)
        requires stdexec::__queryable_with<inner_op_state_t, Impl::get_exec_t>
    {
        return Impl::get_exec(inner_op_state);
    }

    void start() & noexcept {
        stdexec::start(inner_op_state);
    }
};

template <stdexec::sender Sndr>
struct ScheduleFromSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender, Sndr)

    template <stdexec::receiver Rcvr>
    constexpr auto
        connect(Rcvr rcvr) && noexcept(std::is_nothrow_constructible_v<ScheduleFromOpState<Sndr, Rcvr>, Sndr&&, Rcvr&&>)
            -> ScheduleFromOpState<Sndr, Rcvr> {
        return ScheduleFromOpState<Sndr, Rcvr>{std::forward<Sndr>(sndr), std::move(rcvr)};
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::schedule_from_t> {
    template <typename Env, typename Sndr>
    auto operator()(const Env&, stdexec::schedule_from_t, stdexec::__ignore, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ScheduleFromSender<Sndr>, Sndr&&>) -> ScheduleFromSender<Sndr> {
        return {.sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
