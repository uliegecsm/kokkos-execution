#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Experimental::details::execution_space
{

template <class Tag>
struct apply_sender_for;

template <class Tag, class Env>
struct transform_sender_for;

struct Domain;

template <Kokkos::ExecutionSpace Exec>
struct State;

template <Kokkos::ExecutionSpace Exec>
struct SchedulerEnv;

template <Kokkos::ExecutionSpace Exec>
struct Scheduler;

//! Concept for a sender whose completion scheduler is @ref Kokkos::Experimental::details::execution_space::Scheduler.
template <class Sndr, class Env = ::stdexec::env<>>
concept execution_space_completing_sender = ::stdexec::sender<Sndr>
    && ::stdexec::__is_instance_of<std::invoke_result_t<
        ::stdexec::get_completion_scheduler_t<::stdexec::set_value_t>, ::stdexec::env_of_t<Sndr>, Env>,
        Scheduler
    >;

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
