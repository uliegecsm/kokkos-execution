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
struct State {
    std::exception_ptr error;
    stdexec::run_loop loop;
};

//! Receiver for @c stdexec::sync_wait.
template <Kokkos::ExecutionSpace Exec, typename... Values>
struct Receiver {
    using receiver_concept = stdexec::receiver_tag;

    static constexpr auto label = Impl::dispatch_label<Exec, ": sync_wait">();

    Impl::State<Exec> const * state;
    State* runloop_state;
    std::optional<std::tuple<Values...>>* result;

    template <typename... Args>
    void set_value(Args&&... args) && noexcept {
        state->exec.fence(std::string(label));
        result->emplace(std::forward<Args>(args)...);
        runloop_state->loop.finish();
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
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
    /**
     * According to https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait,
     * it has to return an engaged optional (on the value channel).
     *
     * @todo Make the @c noexcept specifier depend on the completion signatures of @p sndr.
     */
    template <stdexec::sender Sndr>
    auto operator()(Sndr&& sndr) const noexcept(false)
        -> std::optional<stdexec::__sync_wait::__value_tuple_for_t<Sndr>> {
        State runloop_state;

        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), stdexec::env{});

        using result_t = std::optional<stdexec::__sync_wait::__value_tuple_for_t<Sndr>>;

        result_t result{};

        auto op_state = stdexec::connect(
            std::forward<Sndr>(sndr),
            Receiver{
                .state = std::move(schd.state),
                .runloop_state = std::addressof(runloop_state),
                .result = std::addressof(result)});

        stdexec::start(op_state);

        runloop_state.loop.run();

        if (runloop_state.error)
            std::rethrow_exception(std::move(runloop_state.error));

        return result;
    }
};

} // namespace Kokkos::Execution::Impl::SyncWait

#endif // KOKKOS_EXECUTION_IMPL_SYNC_WAIT_HPP
