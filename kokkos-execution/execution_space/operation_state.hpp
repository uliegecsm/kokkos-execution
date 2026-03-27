#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/get_exec.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/event.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Clsr>
concept Closure = requires(const Clsr& clsr) {
    typename Clsr::execution_space;

    { clsr.execute() } -> std::same_as<void>;

    { clsr.get_policy() };
    requires Kokkos::ExecutionPolicy<std::remove_cvref_t<decltype(clsr.get_policy())>>;

    requires std::same_as<std::remove_cvref_t<decltype(clsr.get_policy().space())>, typename Clsr::execution_space>;
};

/**
 * @brief Synchronization at the boundary of the work enqueued on an execution space.
 *
 * Under special circumstances, the implementation is allowed to skip any synchronization of asynchronous work.
 * Otherwise, synchronization must occur before invoking the downstream receiver. This situation may arise, for example,
 * when the execution space scheduler is used in a @c stdexec::when_all branch. In the default implementation of @c stdexec::when_all,
 * the branches are not terminated by a @c stdexec::schedule_from, so we'd be missing a synchronization.
 */
template <stdexec::receiver Rcvr, typename OpState>
struct RequiresSynchronization {
    static constexpr bool successor_handles_sync = stdexec::__is_instance_of<Rcvr, ScheduleFromReceiver>
                                                || stdexec::__is_instance_of<Rcvr, SyncWaitReceiver>;

    //! The synchronization will be handled by the successor.
    constexpr bool operator()(const OpState&) const noexcept requires(successor_handles_sync)
    {
        return false;
    }

    /**
     * If the receiver environment can be queried for @ref Kokkos::Execution::ExecutionSpaceImpl::get_exec_t,
     * and if the successor enqueues work on the same execution space instance, no synchronization is needed.
     */
    bool operator()(const OpState& opstate) const noexcept
        requires(!successor_handles_sync && stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>)
    {
        if constexpr (
            std::same_as<
                std::remove_cvref_t<stdexec::__query_result_t<stdexec::env_of_t<Rcvr>, get_exec_t>>,
                stdexec::__query_result_t<OpState, get_exec_t>
            >) {
            return opstate.query(get_exec).get() != get_exec(stdexec::get_env(opstate.rcvr)).get();
        }
        return true;
    }

    //! As a fallback, synchronization is always required.
    constexpr bool operator()(const OpState&) const noexcept
        requires(!successor_handles_sync && !stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>)
    {
        return true;
    }
};

//! The execution space supports events and the receiver is queryable for a delegation scheduler.
template <typename Rcvr, typename Exec>
concept delegate_completion_with_event =
    Kokkos::Execution::Impl::support_events<Exec>
    && stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, stdexec::get_delegation_scheduler_t>;

template <
    stdexec::receiver Rcvr,
    Kokkos::ExecutionSpace Exec,
    bool Delegate = delegate_completion_with_event<Rcvr, Exec>
>
struct MayDelegateCompletionWithEvent;

template <stdexec::receiver Rcvr, Kokkos::ExecutionSpace Exec>
struct WaitEventReceiver {
    using receiver_concept = stdexec::receiver_t;
    using event_t = Impl::Event<Exec>;

    MayDelegateCompletionWithEvent<Rcvr, Exec>* opstate;
    event_t event;

    void set_value() && noexcept {
        try {
            event.wait();
            opstate->storage.__destroy();
            stdexec::set_value(std::move(opstate->rcvr));
        } catch (...) {
            opstate->storage.__destroy();
            stdexec::set_error(std::move(opstate->rcvr), std::current_exception());
        }
    }
};

template <stdexec::receiver Rcvr, Kokkos::ExecutionSpace Exec>
struct MayDelegateCompletionWithEvent<Rcvr, Exec, false> {
    static constexpr auto label = Impl::dispatch_label<Exec, ": after dispatch">();

    Rcvr rcvr;

    template <typename OpState>
    void delegate(OpState* const opstate) noexcept {
        if (RequiresSynchronization<Rcvr, OpState>{}(*opstate)) {
            opstate->query(get_exec).get().fence(std::string(label));
        }
        stdexec::set_value(std::move(this->rcvr));
    }
};

template <stdexec::receiver Rcvr, Kokkos::ExecutionSpace Exec>
struct MayDelegateCompletionWithEvent<Rcvr, Exec, true> {
    using receiver_t = WaitEventReceiver<Rcvr, Exec>;
    using opstate_t = stdexec::connect_result_t<
        stdexec::schedule_result_t<
            stdexec::__query_result_t<stdexec::env_of_t<Rcvr>, stdexec::get_delegation_scheduler_t>
        >,
        receiver_t
    >;

    Rcvr rcvr;
    stdexec::__manual_lifetime<opstate_t> storage{};

    template <typename OpState>
    void delegate(OpState* const opstate) noexcept {
        if (RequiresSynchronization<Rcvr, OpState>{}(*opstate)) {
            storage.__construct_from(
                stdexec::connect,
                stdexec::schedule(stdexec::get_delegation_scheduler(stdexec::get_env(this->rcvr))),
                receiver_t{.opstate = opstate, .event = typename receiver_t::event_t{opstate->query(get_exec).get()}});
            stdexec::start(storage.__get());
        } else {
            stdexec::set_value(std::move(this->rcvr));
        }
    }
};

template <stdexec::receiver Rcvr, Closure Clsr>
struct OpStateBase
    : public stdexec::__immovable
    , public MayDelegateCompletionWithEvent<Rcvr, typename Clsr::execution_space> {
    using execution_space = typename Clsr::execution_space;
    using receiver_t = Rcvr;

    using may_delegate_completion_with_event_t = MayDelegateCompletionWithEvent<Rcvr, execution_space>;

    Clsr clsr;

    constexpr explicit OpStateBase(Rcvr rcvr_, Clsr clsr_) noexcept(
        std::is_nothrow_constructible_v<may_delegate_completion_with_event_t, Rcvr>
        && std::is_nothrow_move_constructible_v<Clsr>)
        : may_delegate_completion_with_event_t{std::move(rcvr_)}
        , clsr(std::move(clsr_)) {
    }

    void propagate_completion_signal(stdexec::set_value_t) noexcept {
        try {
            this->clsr.execute();
        } catch (...) {
            this->propagate_completion_signal(stdexec::set_error, std::current_exception());
            return;
        }

        may_delegate_completion_with_event_t::delegate(this);
    }

    template <typename Error>
    void propagate_completion_signal(stdexec::set_error_t, Error&& error) noexcept {
        stdexec::set_error(std::move(this->rcvr), std::forward<Error>(error));
    }

    void propagate_completion_signal(stdexec::set_stopped_t) noexcept {
        stdexec::set_stopped(std::move(this->rcvr));
    }

    [[nodiscard]]
    constexpr auto query(get_exec_t) const noexcept -> ExecutionSpaceRef<execution_space> {
        return ExecutionSpaceRef<execution_space>{clsr.get_policy().space()};
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, this->rcvr)
};

template <typename ParentOp>
struct OpStateReceiver {
    using receiver_concept = stdexec::receiver_t;

    ParentOp* parent_op;

    void set_value() && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_value);
    }

    template <typename Error>
    void set_error(Error&& error) && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_error, std::forward<Error>(error));
    }

    void set_stopped() && noexcept {
        parent_op->propagate_completion_signal(stdexec::set_stopped);
    }

    KOKKOS_EXECUTION_UPSERT_EXEC(
        typename ParentOp::execution_space,
        parent_op->query(get_exec).get(),
        typename ParentOp::receiver_t,
        parent_op->rcvr)
};

template <stdexec::sender Sndr, stdexec::receiver Rcvr, Closure Clsr>
struct OpState : public OpStateBase<Rcvr, Clsr> {
    using operation_state_concept = stdexec::operation_state_t;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, OpStateReceiver<OpStateBase<Rcvr, Clsr>>>;

    inner_opstate_t inner_opstate;

    constexpr explicit OpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr_,
        Clsr clsr_)
        noexcept(
            std::is_nothrow_constructible_v<OpStateBase<Rcvr, Clsr>, Rcvr&&, Clsr&&>
            && stdexec::__nothrow_connectable<Sndr&&, OpStateReceiver<OpStateBase<Rcvr, Clsr>>>)
        : OpStateBase<Rcvr, Clsr>(std::move(rcvr_), std::move(clsr_))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), OpStateReceiver<OpStateBase<Rcvr, Clsr>>{this})) {
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP
