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

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_SENDER_CONCEPTS_HPP
