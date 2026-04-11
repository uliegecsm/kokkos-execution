#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "kokkos-execution/execution_space/domain.hpp"
#include "kokkos-execution/execution_space/get_exec.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/event.hpp"
#include "kokkos-execution/impl/immovable.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/sync_wait.hpp"

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
                                                || stdexec::__is_instance_of<Rcvr, Impl::SyncWait::Receiver>;

    //! The synchronization will be handled by the successor.
    constexpr bool operator()(const OpState&) const noexcept requires(successor_handles_sync)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_DEBUG << "The synchronization will be handled by the successor.";
#endif
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
            const auto& src = opstate.query(get_exec).get();
            const auto& dst = get_exec(stdexec::get_env(opstate.rcvr)).get();
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_DEBUG << "The synchronization happens if " << Kokkos::Tools::Experimental::device_id(src)
                       << " is not equal to " << Kokkos::Tools::Experimental::device_id(dst) << '.';
#endif
            return src != dst;
        }
        return true;
    }

    //! As a fallback, synchronization is always required.
    constexpr bool operator()(const OpState&) const noexcept
        requires(!successor_handles_sync && !stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_DEBUG << "Synchronization always required.";
#endif
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
    using receiver_concept = stdexec::receiver_tag;
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

template <stdexec::receiver Rcvr, Closure Clsr, Closure... Clsrs>
requires(std::same_as<typename Clsr::execution_space, typename Clsrs::execution_space> && ...)
struct OpStateBase : public MayDelegateCompletionWithEvent<Rcvr, typename Clsr::execution_space> {
    using execution_space = typename Clsr::execution_space;
    using receiver_t = Rcvr;
    using closures_t = stdexec::__tuple<Clsr, Clsrs...>;

    using may_delegate_completion_with_event_t = MayDelegateCompletionWithEvent<Rcvr, execution_space>;

    closures_t clsrs;

    constexpr explicit OpStateBase(Rcvr rcvr_, Clsr clsr_, Clsrs... clsrs_) noexcept(
        std::is_nothrow_constructible_v<may_delegate_completion_with_event_t, Rcvr>
        && std::is_nothrow_move_constructible_v<Clsr> && (std::is_nothrow_move_constructible_v<Clsrs> && ...))
        : may_delegate_completion_with_event_t{std::move(rcvr_)}
        , clsrs(std::move(clsr_), std::move(clsrs_)...) {
    }

    void propagate_completion_signal(stdexec::set_value_t) noexcept {
        try {
            stdexec::__apply([](auto&... clsr) { (clsr.execute(), ...); }, clsrs);
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

    //! @note All @ref clsrs are assumed to reference the same execution space instance.
    [[nodiscard]]
    constexpr auto query(get_exec_t) const noexcept -> ExecutionSpaceRef<execution_space> {
        return ExecutionSpaceRef<execution_space>{stdexec::__get<0>(clsrs).get_policy().space()};
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, this->rcvr)
};

template <stdexec::sender Sndr, stdexec::receiver Rcvr, Closure... Clsrs>
requires(!Impl::dispatching_sender<Sndr>)
struct OpState
    : public Impl::Immovable
    , public OpStateBase<Rcvr, Clsrs...> {
    using operation_state_concept = stdexec::operation_state_tag;

    using base_t = OpStateBase<Rcvr, Clsrs...>;
    using rcvr_t = Impl::Receiver<base_t>;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;

    static constexpr bool opstate_base_is_nothrow_constructible =
        std::is_nothrow_constructible_v<base_t, Rcvr&&, Clsrs&&...>;

    static constexpr bool inner_opstate_is_nothrow_constructible = stdexec::__nothrow_connectable<Sndr&&, rcvr_t>;

    inner_opstate_t inner_opstate;

    constexpr explicit OpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr_,
        Clsrs... clsrs_) noexcept(opstate_base_is_nothrow_constructible && inner_opstate_is_nothrow_constructible)
        : base_t(std::move(rcvr_), std::move(clsrs_)...)
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this})) {
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }
};

template <typename Sndr, typename Rcvr, typename... Clsrs>
struct MakeOpStateFn {
    using type = OpState<Sndr, Rcvr, Clsrs...>;

    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    constexpr auto operator()(Sndr&& sndr, Rcvr rcvr, Clsrs... clsrs) const
        noexcept(std::is_nothrow_constructible_v<type, Sndr&&, Rcvr&&, Clsrs&&...>) -> type {
        return type(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsrs)...);
    }
};

template <Impl::dispatching_sender Sndr, typename Rcvr, typename... Clsrs>
struct MakeOpStateFn<Sndr, Rcvr, Clsrs...> {
    using child_of_sndr_t = stdexec::__child_of<Sndr>;
    using clsr_of_sndr_t = typename stdexec::transform_sender_result_t<Sndr, stdexec::env_of_t<Rcvr>>::closure_t;

    using make_opstate_fn_t = MakeOpStateFn<child_of_sndr_t, Rcvr, clsr_of_sndr_t, Clsrs...>;
    using type = typename make_opstate_fn_t::type;

    static constexpr bool sndr_has_nothrow_transform_sender = stdexec::__detail::__has_nothrow_transform_sender<
        Kokkos::Execution::ExecutionSpaceImpl::Domain,
        stdexec::set_value_t,
        Sndr&&,
        stdexec::env_of_t<Rcvr>
    >;

    static constexpr bool is_nothrow_make_opstate =
        std::is_nothrow_invocable_v<make_opstate_fn_t, child_of_sndr_t&&, Rcvr&&, clsr_of_sndr_t&&, Clsrs&&...>;

    /**
     * @note @c stdexec::__forward_like is used because @c stdexec propagates the value category
     *       of the parent sender to its child.
     *
     * See https://github.com/NVIDIA/stdexec/blob/0a3afb8de52b4fde8ca7ab62ca09a23a8aa6a30f/include/stdexec/__detail/__sender_introspection.hpp#L244-L246.
     */
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved)
    constexpr auto operator()(Sndr&& sndr, Rcvr&& rcvr, Clsrs... clsrs) const
        noexcept(sndr_has_nothrow_transform_sender && is_nothrow_make_opstate) -> type {
        auto trnsfrmd_sndr = stdexec::transform_sender(std::forward<Sndr>(sndr), stdexec::get_env(rcvr));
        return make_opstate_fn_t{}(
            stdexec::__forward_like<Sndr>(trnsfrmd_sndr.sndr),
            std::forward<Rcvr>(rcvr),
            std::move(trnsfrmd_sndr.clsr),
            std::move(clsrs)...);
    }
};

template <typename Sndr, typename Rcvr, typename... Clsrs>
using opstate_t = typename MakeOpStateFn<Sndr, Rcvr, Clsrs...>::type;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP
