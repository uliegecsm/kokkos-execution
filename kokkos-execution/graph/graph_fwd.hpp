#ifndef KOKKOS_EXECUTION_GRAPH_GRAPH_FWD_HPP
#define KOKKOS_EXECUTION_GRAPH_GRAPH_FWD_HPP

#include "Kokkos_Concepts.hpp"

namespace Kokkos::Execution {

template <Kokkos::ExecutionSpace Exec>
struct GraphContext;

} // namespace Kokkos::Execution

namespace Kokkos::Execution::GraphImpl {

template <typename Tag>
struct ApplySenderFor;

struct Domain;

template <Kokkos::ExecutionSpace Exec>
struct Scheduler;

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_GRAPH_FWD_HPP
