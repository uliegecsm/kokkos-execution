#ifndef KOKKOS_EXECUTION_IMPL_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_IMPL_SYNC_WAIT_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/get_exec.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
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
    using receiver_concept = stdexec::receiver_tag;

    static constexpr auto label = Impl::dispatch_label<Exec, ": sync_wait">();

    Impl::State<Exec> const * state;
    State<HasErrorPtr>* runloop_state;
    std::optional<ResultType>* result;

    template <typename... Args>
    void set_value(Args&&... args) && noexcept {
        state->exec.fence(std::string(label));
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

    //! Make others aware of which execution space instance it will synchronize.
    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::__join_env_t<
        stdexec::prop<ExecutionSpaceImpl::get_exec_t, ExecutionSpaceImpl::ExecutionSpaceRef<Exec>>,
        env
    > {
        return stdexec::__env::__join(
            stdexec::prop{ExecutionSpaceImpl::get_exec, ExecutionSpaceImpl::ExecutionSpaceRef{state->exec}},
            env{runloop_state->loop.get_scheduler()});
    }
};

struct SyncWait {
    template <typename Sndr>
    using sends_error = std::bool_constant<stdexec::__sends<stdexec::set_error_t, Sndr, env>>;

    template <typename Sndr>
    using result_t = stdexec::__sync_wait::__value_tuple_for_t<Sndr>;

    template <typename Sndr>
    static constexpr bool is_nothrow_connectable = stdexec::__nothrow_connectable<
        Sndr,
        Receiver<
            typename stdexec::__completion_scheduler_of_t<stdexec::set_value_t, Sndr, env>::execution_space,
            sends_error<Sndr>,
            result_t<Sndr>
        >
    >;

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
            .state = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), stdexec::env{})
                         .state,
            .runloop_state = std::addressof(runloop_state),
            .result = std::addressof(result)};

        auto op_state = stdexec::connect(std::forward<Sndr>(sndr), std::move(rcvr));

        stdexec::start(op_state);

        runloop_state.loop.run();

        if constexpr (sends_error<Sndr&&>::value)
            if (runloop_state.error)
                std::rethrow_exception(std::move(runloop_state.error));

        return result;
    }
};

} // namespace Kokkos::Execution::Impl::SyncWait

#endif // KOKKOS_EXECUTION_IMPL_SYNC_WAIT_HPP
