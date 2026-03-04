#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_CONTEXT_FWD_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_CONTEXT_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Execution::execution_space::impl {

template <typename Tag>
struct apply_sender_for;

template <typename Tag>
struct transform_sender_for;

struct Domain;

template <Kokkos::ExecutionSpace Exec>
struct State;

template <Kokkos::ExecutionSpace Exec>
struct SchedulerEnv;

template <Kokkos::ExecutionSpace Exec>
struct Scheduler;

//! Concept for a sender whose completion scheduler is @ref Kokkos::Execution::execution_space::impl::Scheduler.
template <typename Sndr, typename Env = stdexec::env<>>
concept execution_space_completing_sender =
    stdexec::sender<Sndr>
    && stdexec::__is_instance_of<
        std::invoke_result_t<stdexec::get_completion_scheduler_t<stdexec::set_value_t>, stdexec::env_of_t<Sndr>, Env>,
        Scheduler
    >;

} // namespace Kokkos::Execution::execution_space::impl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_IMPL_CONTEXT_FWD_HPP
