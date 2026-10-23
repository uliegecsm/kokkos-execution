#ifndef KOKKOS_EXECUTION_TESTS_GRAPH_EVENT_DISCARD_MATCHER_HPP
#define KOKKOS_EXECUTION_TESTS_GRAPH_EVENT_DISCARD_MATCHER_HPP

#include "kokkos-execution/graph/events.hpp"

namespace Tests::GraphImpl {

//! Similar to @ref EventDiscardMatcher, for graph-related events.
template <Kokkos::ExecutionSpace Exec>
struct GraphEventDiscardMatcher {
    /**
     * Filter out @ref Kokkos::utils::callbacks::BeginFenceEvent triggered by:
     *  * https://github.com/kokkos/kokkos/blob/10e9786f862733fc7c7a23d5e998e3d161dd7b70/core/src/impl/Kokkos_Default_Graph_Impl.hpp#L144
     *  * https://github.com/kokkos/kokkos/blob/10e9786f862733fc7c7a23d5e998e3d161dd7b70/core/src/impl/Kokkos_Default_GraphNode_Impl.hpp#L124
     *  * https://github.com/kokkos/kokkos/blob/10e9786f862733fc7c7a23d5e998e3d161dd7b70/core/src/HIP/Kokkos_HIP_Graph_Impl.hpp#L100
     */
    constexpr bool operator()(const Kokkos::utils::callbacks::BeginFenceEvent& event) const {
        return event.name != "Kokkos::DefaultGraph::submit: fencing before launching graph nodes"
            && event.name != "Kokkos::DefaultGraphNode::execute_node: sync with predecessors"
            && event.name != "Kokkos::GraphImpl::~GraphImpl: Graph Destruction";
    }

    template <Kokkos::utils::callbacks::Event EventType>
    constexpr bool operator()(const EventType&) const {
        return true;
    }
};

} // namespace Tests::GraphImpl

#endif // KOKKOS_EXECUTION_TESTS_GRAPH_EVENT_DISCARD_MATCHER_HPP
