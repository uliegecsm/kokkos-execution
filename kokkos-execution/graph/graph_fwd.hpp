#ifndef KOKKOS_EXECUTION_GRAPH_GRAPH_FWD_HPP
#define KOKKOS_EXECUTION_GRAPH_GRAPH_FWD_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Concepts.hpp"

#include "kokkos-execution/impl/diagnostics.hpp"

namespace Kokkos::Execution {

template <Kokkos::ExecutionSpace Exec>
struct GraphContext;

} // namespace Kokkos::Execution

namespace Kokkos::Execution::GraphImpl {

template <typename Tag>
struct ApplySenderFor;

template <typename Tag>
struct TransformSenderFor;

struct Domain;

template <Kokkos::ExecutionSpace Exec>
struct Scheduler;

template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
struct GraphOperationStateFor : public std::false_type { };

template <typename OpState, typename Exec>
concept graph_operation_state_for = stdexec::operation_state<OpState> && Kokkos::ExecutionSpace<Exec>
                                 && GraphOperationStateFor<OpState, Exec>::value;

template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
struct RemainsOnGraphFor : public std::false_type {
    static constexpr void diagnose() {
        static_assert(Impl::always_false<OpState>, "This operation state that not satisfy 'RemainsOnGraphFor'.");
    }
};

template <typename OpState, typename Exec>
concept remains_on_graph_for = stdexec::operation_state<OpState> && Kokkos::ExecutionSpace<Exec>
                            && RemainsOnGraphFor<OpState, Exec>::value;

template <typename OpState, typename Exec>
constexpr bool diagnose_remains_on_graph_for() noexcept {
    if constexpr (!remains_on_graph_for<OpState, Exec>) {
        RemainsOnGraphFor<OpState, Exec>::diagnose();
        return false;
    } else {
        return true;
    }
}

/**
 * Determine whether the operation state resulting from the connection of @p Sndr and @p Rcvr
 * contains only operations that map to @ref Kokkos::Execution::GraphImpl::Domain operation states that create a graph node
 * and whose execution space type is @p Exec.
 */
template <typename Exec, typename Sndr, typename Rcvr>
concept connect_result_remains_on_graph_for = Kokkos::ExecutionSpace<Exec> && stdexec::sender<Sndr>
                                           && stdexec::receiver<Rcvr>
                                           && remains_on_graph_for<stdexec::connect_result_t<Sndr, Rcvr>, Exec>;

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_GRAPH_FWD_HPP
