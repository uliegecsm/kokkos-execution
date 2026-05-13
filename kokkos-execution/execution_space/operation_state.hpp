#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/domain.hpp"
#include "kokkos-execution/impl/completion_signal.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/impl/immovable.hpp"
#include "kokkos-execution/impl/make_opstate.hpp"
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

template <typename Exec, typename Rcvr>
consteval auto select_sync_policy() {
    if constexpr (stdexec::__is_instance_of<Rcvr, Impl::SyncWait::Receiver>) {
        return Impl::SyncPolicy::PassThrough{};
    } else if constexpr (Impl::deferred_completion_receiver<Rcvr>) {
        return Impl::SyncPolicy::DeferWaitEvent{};
    } else if constexpr (
        Impl::has_non_blocking_dispatch<Exec>
        && stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, stdexec::get_delegation_scheduler_t>) {
        return Impl::SyncPolicy::ScheduleWaitEvent{};
    } else {
        return Impl::SyncPolicy::InlineFenceExec{};
    }
}

template <typename Exec, typename Rcvr>
using select_sync_policy_t = decltype(select_sync_policy<Exec, Rcvr>());

template <stdexec::receiver Rcvr, Closure Clsr, Closure... Clsrs>
requires(std::same_as<typename Clsr::execution_space, typename Clsrs::execution_space> && ...)
struct OpStateBase {
    using execution_space = typename Clsr::execution_space;

    using sync_policy_t = select_sync_policy_t<execution_space, Rcvr>;
    using completion_signal_t = Impl::CompletionSignal<sync_policy_t, execution_space, Rcvr>;
    using closures_t = stdexec::__tuple<Clsr, Clsrs...>;

    completion_signal_t completion_signal;
    closures_t clsrs;

    constexpr explicit OpStateBase(Rcvr rcvr, Clsr clsr_, Clsrs... clsrs_) noexcept(
        std::is_nothrow_constructible_v<completion_signal_t, Rcvr&&> && std::is_nothrow_move_constructible_v<Clsr>
        && (std::is_nothrow_move_constructible_v<Clsrs> && ...))
        : completion_signal(std::move(rcvr))
        , clsrs(std::move(clsr_), std::move(clsrs_)...) {
    }

    void complete(stdexec::set_value_t) noexcept {
        try {
            stdexec::__apply([](auto&... clsr) { (clsr.execute(), ...); }, clsrs);
        } catch (...) {
            this->complete(stdexec::set_error, std::current_exception());
            return;
        }
        completion_signal.propagate(stdexec::set_value, this->query(Impl::get_exec).get());
    }

    template <typename Error>
    void complete(stdexec::set_error_t, Error&& error) noexcept {
        completion_signal.propagate(stdexec::set_error, std::forward<Error>(error));
    }

    void complete(stdexec::set_stopped_t) noexcept {
        completion_signal.propagate(stdexec::set_stopped);
    }

    //! @note All @ref clsrs are assumed to reference the same execution space instance.
    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> Impl::ExecutionSpaceRef<execution_space> {
        return Impl::ExecutionSpaceRef<execution_space>{stdexec::__get<0>(clsrs).get_policy().space()};
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return stdexec::get_env(this->completion_signal.rcvr);
    }
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

    //! @bug Needed only for @ref test_any_sender.cpp.
#if defined(KOKKOS_EXECUTION_IMPL_OPSTATE_IMMOVABLE_FIX)
    STDEXEC_IMMOVABLE(OpState);
#endif

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
using make_opstate_t = Impl::MakeOpState<Domain, OpState>::Huddle<Sndr, Rcvr, Clsrs...>;

template <typename Sndr, typename Rcvr, typename... Clsrs>
using opstate_t = typename make_opstate_t<Sndr, Rcvr, Clsrs...>::type;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_OPERATION_STATE_HPP
