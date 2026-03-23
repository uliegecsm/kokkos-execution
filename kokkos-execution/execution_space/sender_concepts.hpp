#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SENDER_CONCEPTS_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SENDER_CONCEPTS_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Concept for a sender whose completion scheduler is @ref Kokkos::Execution::ExecutionSpaceImpl::Scheduler.
template <typename Sndr, typename Env = stdexec::env<>>
concept execution_space_completing_sender =
    stdexec::sender<Sndr>
    && stdexec::__is_instance_of<
        std::invoke_result_t<stdexec::get_completion_scheduler_t<stdexec::set_value_t>, stdexec::env_of_t<Sndr>, Env>,
        Scheduler
    >;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SENDER_CONCEPTS_HPP
