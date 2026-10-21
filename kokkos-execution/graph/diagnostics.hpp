#ifndef KOKKOS_EXECUTION_GRAPH_DIAGNOSTICS_HPP
#define KOKKOS_EXECUTION_GRAPH_DIAGNOSTICS_HPP

/**
 * @file
 *
 * This file contains diagnostic utilities related to the @ref Kokkos::Execution::GraphImpl::Domain domain.
 */

namespace Kokkos::Execution::GraphImpl {

//! Tags to be used when an operation state does not model @ref Kokkos::Execution::GraphImpl::RemainsOnGraphFor.
///@{
struct OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH;
struct BECAUSE_THE_INNER_OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH;
struct BECAUSE_AT_LEAST_ONE_CHILD_DOES_NOT_REMAIN_ON_GRAPH;
///@}

//! @name Tags to be used in @ref Kokkos::Execution::GraphImpl::no_graph_scheduler_in_env.
///@{
struct CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_GRAPH_SCHEDULER;
struct BECAUSE_THERE_IS_NO_GRAPH_SCHEDULER_IN_THE_ENVIRONMENT;
///@}

/**
 * @brief Show a better compile diagnostic when there is no @ref Kokkos::Execution::GraphImpl::Scheduler found.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/6b831318ed0a87e464b28c29a01b88695e5d71c6/include/nvexec/stream/common.cuh#L126-L135.
 */
template <typename Tag, typename Sndr, typename... Env>
auto no_graph_scheduler_in_env() noexcept {
    return stdexec::__not_a_sender<
        stdexec::_WHAT_(CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_GRAPH_SCHEDULER),
        stdexec::_WHY_(BECAUSE_THERE_IS_NO_GRAPH_SCHEDULER_IN_THE_ENVIRONMENT),
        stdexec::_WHERE_(stdexec::_IN_ALGORITHM_, Tag),
        stdexec::_WITH_SENDER_<Sndr>,
        stdexec::_WITH_ENVIRONMENT_(Env...)
    >{};
}

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_DIAGNOSTICS_HPP
