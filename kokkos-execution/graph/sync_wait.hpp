#ifndef KOKKOS_EXECUTION_GRAPH_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_GRAPH_SYNC_WAIT_HPP

#include "kokkos-execution/graph/graph_fwd.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/sync_wait.hpp"

namespace Kokkos::Execution::GraphImpl {

template <>
struct ApplySenderFor<stdexec::sync_wait_t> : public Impl::SyncWait::ApplySenderFor<GraphCompletingSender> { };

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_SYNC_WAIT_HPP
