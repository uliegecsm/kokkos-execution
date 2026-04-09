#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SENDER_INTROSPECTION_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SENDER_INTROSPECTION_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Type of the execution space extracted from a sender's completion scheduler.
template <typename Sndr, typename... Env>
using exec_of_t = typename stdexec::__completion_scheduler_of_t<stdexec::set_value_t, Sndr, Env...>::execution_space;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SENDER_INTROSPECTION_HPP
