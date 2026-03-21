#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/get_exec.hpp"
#include "kokkos-execution/impl/env.hpp"
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

template <stdexec::receiver Rcvr, Closure Clsr>
struct OpStateBase : public stdexec::__immovable {
    using execution_space = typename Clsr::execution_space;

    using receiver_t = Rcvr;

    receiver_t rcvr;
    Clsr clsr;

    constexpr explicit OpStateBase(Rcvr rcvr_, Clsr clsr_)
        noexcept(std::is_nothrow_move_constructible_v<Rcvr> && std::is_nothrow_move_constructible_v<Clsr>)
        : rcvr(std::move(rcvr_))
        , clsr(std::move(clsr_)) {
    }

    void propagate_completion_signal(stdexec::set_value_t) noexcept {
        try {
            clsr.execute();
        } catch (...) {
            this->propagate_completion_signal(stdexec::set_error, std::current_exception());
            return;
        }

        /**
         * Sync at the boundary of the work enqueued on the execution space.
         *
         * If the receiver environment can be queried for @ref Kokkos::Execution::ExecutionSpaceImpl::get_exec_t,
         * and if the successor enqueues work on the same execution space instance, no fence is needed.
         *
         * Otherwise, synchronization must occur before invoking the downstream receiver. This situation may arise, for example,
         * when the execution space scheduler is used in a @c stdexec::when_all branch. In the default implementation of @c stdexec::when_all,
         * the branches are not terminated by a @c stdexec::schedule_from, so we'd be missing a synchronization.
         *
         * @todo Explore event-based synchronization for cases in which the successor is still on the device,
         *       but on a different execution space. The objective would be to avoid occupying the current host thread.
         */
        const bool skip = [&]() {
            if constexpr (
                stdexec::__is_instance_of<Rcvr, ScheduleFromReceiver>
                | stdexec::__is_instance_of<Rcvr, SyncWaitReceiver>) {
                return true;
            } else {
                if constexpr (stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>) {
                    if constexpr (
                        std::same_as<
                            std::remove_cvref_t<decltype(get_exec(stdexec::get_env(rcvr)).get())>,
                            execution_space
                        >) {
                        return this->query(get_exec).get() == get_exec(stdexec::get_env(rcvr)).get();
                    }
                }
                return false;
            }
        }();
        if (!skip) {
            this->query(get_exec)
                .get()
                .fence(std::format("{}: continuation", Kokkos::Impl::TypeInfo<execution_space>::name()));
        }
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void propagate_completion_signal(stdexec::set_error_t, Error&& error) noexcept {
        stdexec::set_error(std::move(rcvr), std::forward<Error>(error));
    }

    void propagate_completion_signal(stdexec::set_stopped_t) noexcept {
        stdexec::set_stopped(std::move(rcvr));
    }

    [[nodiscard]]
    constexpr auto query(get_exec_t) const noexcept -> ExecutionSpaceRef<execution_space> {
        return ExecutionSpaceRef<execution_space>{clsr.get_policy().space()};
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, rcvr)
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
