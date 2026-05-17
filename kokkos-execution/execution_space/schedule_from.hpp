#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename ParentOp, typename ParentOpBase = typename ParentOp::base_t>
struct ScheduleFromReceiver {
    using receiver_concept = Impl::SubmittedReceiverTag;

    using parent_op_base_t = ParentOpBase;
    using exec_env_policy_t = typename parent_op_base_t::exec_env_policy_t;

    ParentOp* parent_op;
    ParentOpBase* parent_op_base;

    constexpr explicit ScheduleFromReceiver(ParentOp* parent_op_) noexcept
        : parent_op(parent_op_)
        , parent_op_base(parent_op_) {
    }

    void set_value() && noexcept {
        stdexec::set_value(std::move(parent_op_base->rcvr));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        stdexec::set_error(std::move(parent_op_base->rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        stdexec::set_stopped(std::move(parent_op_base->rcvr));
    }

    void submitted() && noexcept {
        //! Stay in the @ref Kokkos::Execution::ExecutionSpaceImpl::Domain.
        if constexpr (stdexec::__is_instance_of<decltype(parent_op_base->rcvr), ContinuesOnReceiver>) {
            //static_assert(
            //    std::same_as<typename ParentOp::completion_signal_policy_t, Impl::SubmittedPolicy::OrderOnExec>);
            //std::move(parent_op_base->rcvr).submitted();
            stdexec::set_value(std::move(parent_op_base->rcvr));
        }
        //! Transition to another domain.
        else {
            static_assert(
                std::same_as<typename ParentOp::completion_signal_policy_t, Impl::SyncPolicy::InlineFenceExec>);
            auto exec_ref = parent_op->query(Impl::get_exec);
            using Exec = typename decltype(exec_ref)::execution_space;
            const Exec& exec = exec_ref.get();
            try {
                exec.fence(std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<Exec>::name()));
                stdexec::set_value(std::move(parent_op_base->rcvr));
            } catch (...) {
                stdexec::set_error(std::move(parent_op_base->rcvr), std::current_exception());
            }
        }
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> extend_env_t<exec_env_policy_t, stdexec::env_of_t<parent_op_base_t>> {
        return extend_env<exec_env_policy_t>(stdexec::get_env(*parent_op_base));
    }
};

template <typename Rcvr>
consteval auto select_schedule_from_opstate_completion_signal_policy() {
    if constexpr (Impl::supports_submitted_order_on<Rcvr>) {
        return Impl::SubmittedPolicy::OrderOnExec{};
    } else {
        return Impl::SyncPolicy::InlineFenceExec{};
    }
}

template <typename Rcvr>
using schedule_from_opstate_completion_signal_policy_t =
    decltype(select_schedule_from_opstate_completion_signal_policy<Rcvr>());

template <stdexec::receiver Rcvr>
struct ScheduleFromOpStateBase {
    Rcvr rcvr;

    using exec_env_policy_t = extend_exec_env_policy_t<stdexec::env_of_t<Rcvr>>;
    using completion_signal_policy_t = schedule_from_opstate_completion_signal_policy_t<Rcvr>;

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return stdexec::get_env(this->rcvr);
    }
};

template <stdexec::sender Sndr, stdexec::receiver Rcvr>
struct ScheduleFromOpState
    : public Impl::Immovable
    , public ScheduleFromOpStateBase<Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = ScheduleFromOpStateBase<Rcvr>;

    using rcvr_t = ScheduleFromReceiver<ScheduleFromOpState>;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    inner_opstate_t inner_opstate;

    constexpr explicit ScheduleFromOpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr)
        noexcept(std::is_nothrow_constructible_v<base_t, Rcvr&&> && stdexec::__nothrow_connectable<Sndr&&, rcvr_t>)
        : base_t(std::move(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> decltype(auto)
        requires stdexec::__queryable_with<inner_opstate_t, Impl::get_exec_t>
    {
        return Impl::get_exec(inner_opstate);
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }
};

template <stdexec::sender Sndr>
struct ScheduleFromSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender)

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
        noexcept(std::is_nothrow_constructible_v<ScheduleFromSender<Sndr>, Sndr&&>) {
        return ScheduleFromSender<Sndr>{.sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
