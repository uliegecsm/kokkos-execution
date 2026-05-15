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
struct Scheduler;

template <typename ExecEnvPolicy, stdexec::receiver Rcvr>
struct ScheduleFromReceiver;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_EXECUTION_SPACE_FWD_HPP
