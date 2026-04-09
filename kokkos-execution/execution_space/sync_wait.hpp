#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SYNC_WAIT_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/sync_wait.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <>
struct ApplySenderFor<stdexec::sync_wait_t> : public Impl::SyncWait::ApplySenderFor<ExecutionSpaceCompletingSender> { };

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SYNC_WAIT_HPP
