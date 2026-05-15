#ifndef KOKKOS_EXECUTION_IMPL_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_IMPL_SYNC_WAIT_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/completion_signal.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/event.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/impl/state.hpp"

namespace Kokkos::Execution::Impl::SyncWait {

//! Inspired by https://github.com/NVIDIA/stdexec/blob/16076a81efa4477513e6ede9c2741fd034ecef99/include/stdexec/__detail/__sync_wait.hpp#L45-L65.
struct env {
    stdexec::run_loop::scheduler schd;

    [[nodiscard]]
    auto query(stdexec::get_scheduler_t) const noexcept -> stdexec::run_loop::scheduler {
        return schd;
    }

    [[nodiscard]]
    auto query(stdexec::get_delegation_scheduler_t) const noexcept -> stdexec::run_loop::scheduler {
        return schd;
    }
};

//! Inspired by https://github.com/NVIDIA/stdexec/blob/16076a81efa4477513e6ede9c2741fd034ecef99/include/stdexec/__detail/__sync_wait.hpp#L83-L86.
template <typename HasErrorPtr>
struct State;

template <>
struct State<std::true_type> {
    std::exception_ptr error;
    stdexec::run_loop loop;
};

template <>
struct State<std::false_type> {
    stdexec::run_loop loop;
};

//! Receiver for @c stdexec::sync_wait.
template <Kokkos::ExecutionSpace Exec, typename HasErrorPtr = std::false_type, typename ResultType = std::tuple<>>
struct Receiver {
    using receiver_concept = Impl::SubmittedReceiverTag;

    static constexpr auto label = Impl::dispatch_label<Exec, ": sync_wait">();

    Impl::State<Exec> const * state;
    State<HasErrorPtr>* runloop_state;
    std::optional<ResultType>* result;

    template <typename... Args>
    void set_value(Args&&... args) && noexcept {
        result->emplace(std::forward<Args>(args)...);
        runloop_state->loop.finish();
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept requires HasErrorPtr::value
    {
        runloop_state->error = std::forward<Error>(err);
        state->exec.fence(std::string(label));
        runloop_state->loop.finish();
    }

    void set_stopped() && noexcept {
        state->exec.fence(std::string(label));
        runloop_state->loop.finish();
    }

    void submitted(Impl::OrderOn<Exec> order_on) & noexcept {
        order_on.synchronize(std::string(label));
        result->emplace();
        runloop_state->loop.finish();
    }

    void submitted(Impl::DependOn<Impl::Event<Exec>> depend_on) & noexcept {
        depend_on.synchronize();
        result->emplace();
        runloop_state->loop.finish();
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> env {
        return {.schd = runloop_state->loop.get_scheduler()};
    }
};

struct SyncWait {
    template <typename Sndr>
    using sends_error = std::bool_constant<stdexec::__sends<stdexec::set_error_t, Sndr, env>>;

    template <typename Sndr>
    using result_t = stdexec::__sync_wait::__value_tuple_for_t<Sndr>;

    template <typename Sndr>
    using receiver_t = Receiver<Impl::exec_of_t<Sndr, env>, sends_error<Sndr>, result_t<Sndr>>;

    template <typename Sndr>
    static constexpr bool is_nothrow_connectable = stdexec::__nothrow_connectable<Sndr, receiver_t<Sndr>>;

    /**
     * According to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait,
     * it has to return an engaged optional (on the value channel).
     */
    template <stdexec::sender Sndr>
    auto operator()(Sndr&& sndr) const noexcept(!sends_error<Sndr&&>::value && is_nothrow_connectable<Sndr&&>)
        -> std::optional<result_t<Sndr&&>> {
        //! See https://github.com/NVIDIA/stdexec/blob/45c0f5803c190366a8529833901d1f6340b40d2e/include/stdexec/__detail/__run_loop.hpp#L50.
        static_assert(noexcept(std::declval<State<sends_error<Sndr&&>>>().loop.run()));

        State<sends_error<Sndr&&>> runloop_state;

        std::optional<result_t<Sndr&&>> result{};

        Receiver rcvr{
            .state = stdexec::get_completion_scheduler<stdexec::set_value_t>(
                         stdexec::get_env(sndr), env{.schd = runloop_state.loop.get_scheduler()})
                         .state,
            .runloop_state = std::addressof(runloop_state),
            .result = std::addressof(result)};

        static_assert(std::same_as<decltype(rcvr), receiver_t<Sndr&&>>);

        auto op_state = stdexec::connect(std::forward<Sndr>(sndr), std::move(rcvr));

        stdexec::start(op_state);

        runloop_state.loop.run();

        if constexpr (sends_error<Sndr&&>::value)
            if (runloop_state.error)
                std::rethrow_exception(std::move(runloop_state.error));

        return result;
    }
};

/**
 * @brief Customize @c stdexec::sync_wait.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/blob/e8a6a7b25fbc2463e1dfe0ee20973b1fe622bfcf/include/nvexec/stream_context.cuh#L247-L251
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait
 *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-dispatch
 */
template <template <typename...> typename SndrTrait>
struct ApplySenderFor {
    template <typename Sndr>
    requires SndrTrait<Sndr, env>::value
    auto operator()(Sndr&& sndr) const noexcept(std::is_nothrow_invocable_v<SyncWait, Sndr&&>) {
        return SyncWait{}(std::forward<Sndr>(sndr));
    }
};

} // namespace Kokkos::Execution::Impl::SyncWait

#endif // KOKKOS_EXECUTION_IMPL_SYNC_WAIT_HPP
