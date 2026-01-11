#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Experimental::details::execution_space
{

template <class Tag>
struct apply_sender_for;

template <class Tag, class Env>
struct transform_sender_for;

template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceSchedulerEnv;

template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceScheduler;

//! Concept for a sender whose completion scheduler is @ref Kokkos::Experimental::details::execution_space::ExecutionSpaceScheduler.
template <class Sndr, class Env = ::stdexec::env<>>
concept execution_space_completing_sender = ::stdexec::sender<Sndr>
    && ::stdexec::__is_instance_of<std::invoke_result_t<
        ::stdexec::get_completion_scheduler_t<::stdexec::set_value_t>, ::stdexec::env_of_t<Sndr>, Env>,
        ExecutionSpaceScheduler
    >;

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
