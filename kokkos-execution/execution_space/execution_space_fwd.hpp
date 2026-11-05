#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_EXECUTION_SPACE_FWD_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_EXECUTION_SPACE_FWD_HPP

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Concepts.hpp"
#endif

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <typename Tag>
struct ApplySenderFor;

template <typename Tag>
struct TransformSenderFor;

struct Domain;

template <Kokkos::ExecutionSpace Exec>
struct Scheduler;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_EXECUTION_SPACE_FWD_HPP
