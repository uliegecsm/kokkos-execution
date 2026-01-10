#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Experimental::details::execution_space
{

template <class Tag, class Env>
struct transform_sender_for;

template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceSchedulerEnv;

template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceScheduler;

} // namespace Kokkos::Experimental::details::execution_space

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_FWD_HPP
