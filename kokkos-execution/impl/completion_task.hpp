#ifndef KOKKOS_EXECUTION_IMPL_CONTINUATION_TASK_HPP
#define KOKKOS_EXECUTION_IMPL_CONTINUATION_TASK_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/event.hpp"

namespace Kokkos::Execution::Impl {

struct InlineWaitCompletionPolicy {};
struct DelegateWaitCompletionPolicy {};
struct DeferWaitCompletionPolicy {};

template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionTask {
    stdexec::__manual_lifetime<Event<Exec>>& event;
    Rcvr rcvr;

    void execute() && noexcept {
        try {
            event->wait();
            event.__destroy();
            stdexec::set_value(std::move(rcvr));
        } catch (...) {
            auto eptr = std::current_exception();
            event.__destroy();
            stdexec::set_error(std::move(rcvr), std::move(eptr));
        }
    }
};

template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionReceiver {
    using receiver_concept = stdexec::receiver_tag;

    stdexec::__manual_lifetime<Event<Exec>>& event;
    Rcvr rcvr;

    void set_value() && noexcept {
        CompletionTask<Exec, Rcvr>{event, std::move(rcvr)}.execute();
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, rcvr)
};

template <typename CompletionPolicy, Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionOpState;

template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionOpState<InlineWaitCompletionPolicy, Exec, Rcvr> {
    using operation_state_concept = stdexec::operation_state_tag;
    
    stdexec::__manual_lifetime<Event<Exec>>& event;
    Rcvr rcvr;

    void start() & noexcept{
        CompletionTask<Exec, Rcvr>{event, std::move(rcvr)}.execute();
    }
};

template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionOpState<DelegateWaitCompletionPolicy, Exec, Rcvr> {
    using operation_state_concept = stdexec::operation_state_tag;

    using inner_op_t = stdexec::connect_result_t
        stdexec::schedule_result_t
            stdexec::__query_result_t<stdexec::env_of_t<Rcvr>,
                                      stdexec::get_delegation_scheduler_t>
        >,
        CompletionReceiver<Exec, Rcvr>
    >;
    inner_op_t inner_op;

    CompletionOpState(stdexec::__manual_lifetime<Event<Exec>>& event_, Rcvr rcvr)
      , inner_op(stdexec::connect(
            stdexec::schedule(
                stdexec::get_delegation_scheduler(stdexec::get_env(rcvr))),
            CompletionReceiver<Exec, Rcvr>{event_, std::move(rcvr)}))
    {}

    void start() & noexcept {
        stdexec::start(inner_op);
    }
};

template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionOpState<DeferWaitCompletionPolicy, Exec, Rcvr> {
    using operation_state_concept = stdexec::operation_state_tag;
    
    stdexec::__manual_lifetime<Event<Exec>>& event;
    Rcvr rcvr;

    void start() & noexcept {
        std::move(rcvr).set_awaitable_completion(event);
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_CONTINUATION_TASK_HPP
