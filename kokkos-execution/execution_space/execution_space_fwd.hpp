#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_EXECUTION_SPACE_FWD_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_EXECUTION_SPACE_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Tag>
struct ApplySenderFor;

template <typename Tag>
struct TransformSenderFor;

struct Domain;

template <Kokkos::ExecutionSpace Exec>
struct State;

template <Kokkos::ExecutionSpace Exec>
struct Scheduler;

template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ScheduleFromReceiver;

template <Kokkos::ExecutionSpace Exec, typename... Values>
struct SyncWaitReceiver;

//! Concept for a sender whose completion scheduler is @ref Kokkos::Execution::ExecutionSpaceImpl::Scheduler.
template <typename Sndr, typename Env = stdexec::env<>>
concept execution_space_completing_sender =
    stdexec::sender<Sndr>
    && stdexec::__is_instance_of<
        std::invoke_result_t<stdexec::get_completion_scheduler_t<stdexec::set_value_t>, stdexec::env_of_t<Sndr>, Env>,
        Scheduler
    >;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_EXECUTION_SPACE_FWD_HPP
