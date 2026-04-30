#ifndef KOKKOS_EXECUTION_GRAPH_SENDER_CONCEPTS_HPP
#define KOKKOS_EXECUTION_GRAPH_SENDER_CONCEPTS_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"

namespace Kokkos::Execution::GraphImpl {

//! Concept for a sender whose completion scheduler is @ref Kokkos::Execution::GraphImpl::Scheduler.
template <typename Sndr, typename... Env>
concept graph_completing_sender =
    stdexec::sender<Sndr>
    && stdexec::__is_instance_of<
        std::invoke_result_t<stdexec::get_completion_scheduler_t<stdexec::set_value_t>, stdexec::env_of_t<Sndr>, Env...>,
        Scheduler
    >;

template <typename Sndr, typename... Env>
struct GraphCompletingSender : public std::bool_constant<graph_completing_sender<Sndr, Env...>> { };

struct CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_GRAPH_SCHEDULER;
struct BECAUSE_THERE_IS_NO_GRAPH_SCHEDULER_IN_THE_ENVIRONMENT;

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
        stdexec::_WITH_PRETTY_SENDER_<Sndr>,
        stdexec::_WITH_ENVIRONMENT_(Env...)
    >{};
}

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_SENDER_CONCEPTS_HPP
