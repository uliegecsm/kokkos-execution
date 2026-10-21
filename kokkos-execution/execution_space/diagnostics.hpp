#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_DIAGNOSTICS_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_DIAGNOSTICS_HPP

#include "kokkos-execution/stdexec.hpp"

/**
 * @file
 *
 * This file contains diagnostic utilities related to the @ref Kokkos::Execution::ExecutionSpaceImpl::Domain domain.
 */

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! @name Tags for @ref no_execution_space_scheduler_in_env.
///@{
struct CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_EXECUTION_SPACE_SCHEDULER;
struct BECAUSE_THERE_IS_NO_EXECUTION_SPACE_SCHEDULER_IN_THE_ENVIRONMENT;
///@}

/**
 * @brief Show a better compile diagnostic when there is no @ref Kokkos::Execution::ExecutionSpaceImpl::Scheduler found.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/6b831318ed0a87e464b28c29a01b88695e5d71c6/include/nvexec/stream/common.cuh#L126-L135.
 */
template <typename Tag, typename Sndr, typename... Env>
auto no_execution_space_scheduler_in_env() noexcept {
    return stdexec::__not_a_sender<
        stdexec::_WHAT_(CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_EXECUTION_SPACE_SCHEDULER),
        stdexec::_WHY_(BECAUSE_THERE_IS_NO_EXECUTION_SPACE_SCHEDULER_IN_THE_ENVIRONMENT),
        stdexec::_WHERE_(stdexec::_IN_ALGORITHM_, Tag),
        stdexec::_WITH_SENDER_<Sndr>,
        stdexec::_WITH_ENVIRONMENT_(Env...)
    >{};
}

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_DIAGNOSTICS_HPP
