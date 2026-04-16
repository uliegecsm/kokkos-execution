#ifndef KOKKOS_EXECUTION_GRAPH_EVENTS_HPP
#define KOKKOS_EXECUTION_GRAPH_EVENTS_HPP

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
#    include "kokkos-utils/callbacks/Manager.hpp"
#endif

namespace Kokkos::Execution::GraphImpl {

//! Event to be sent to @ref Kokkos::utils::callbacks::dispatch when a @c Kokkos graph is created.
struct GraphCreateEvent {
    void* graph = nullptr;
    uint32_t dev_id = 0;
    uint64_t event_id = 0;

    constexpr auto operator<=>(const GraphCreateEvent&) const = default;

    friend std::ostream& operator<<(std::ostream& out, const GraphCreateEvent& event) {
        return out << "GraphCreateEvent: {graph = " << event.graph << ", dev_id = " << event.dev_id
                   << ", event_id = " << event.event_id << '}';
    }
};

/**
 * @brief Event to be sent to @ref Kokkos::utils::callbacks::dispatch when a @c Kokkos graph node is added.
 *
 * @todo Store a label.
 * @todo Store the dispatch tag.
 */
struct GraphAddNodeEvent {
    void* graph = nullptr;
    void* predecessor = nullptr;
    void* node = nullptr;
    uint32_t dev_id = 0;

    constexpr auto operator<=>(const GraphAddNodeEvent&) const = default;

    friend std::ostream& operator<<(std::ostream& out, const GraphAddNodeEvent& event) {
        return out << "GraphAddNodeEvent: {graph = " << event.graph << ", predecessor = " << event.predecessor
                   << ", node = " << event.node << ", dev_id = " << event.dev_id << '}';
    }
};

//! Event to be sent to @ref Kokkos::utils::callbacks::dispatch when a @c Kokkos graph is instantiated.
struct GraphInstantiateEvent {
    void* graph = nullptr;

    constexpr auto operator<=>(const GraphInstantiateEvent&) const = default;

    friend std::ostream& operator<<(std::ostream& out, const GraphInstantiateEvent& event) {
        return out << "GraphInstantiateEvent: {graph = " << event.graph << '}';
    }
};

//! Event to be sent to @ref Kokkos::utils::callbacks::dispatch when a @c Kokkos graph is submitted.
struct GraphSubmitEvent {
    void* graph = nullptr;
    uint32_t dev_id = 0;

    constexpr auto operator<=>(const GraphSubmitEvent&) const = default;

    friend std::ostream& operator<<(std::ostream& out, const GraphSubmitEvent& event) {
        return out << "GraphSubmitEvent: {graph = " << event.graph << ", dev_id = " << event.dev_id << '}';
    }
};

//! Retrieve the raw graph pointer from a node.
template <typename NodeType>
requires Kokkos::Impl::is_specialization_of_v<NodeType, Kokkos::Experimental::GraphNodeRef>
auto* get_graph_impl_ptr(const NodeType& node) noexcept {
    return Kokkos::Impl::GraphAccess::get_graph_weak_ptr(node).lock().get();
}

//! Retrieve the raw node pointer.
template <typename NodeType>
requires Kokkos::Impl::is_specialization_of_v<NodeType, Kokkos::Experimental::GraphNodeRef>
auto* get_node_ptr(const NodeType& node) noexcept {
    return Kokkos::Impl::GraphAccess::get_node_ptr(node).get();
}

//! Record a @ref GraphCreateEvent event.
template <Kokkos::ExecutionSpace Exec>
void graph_create_event(const Kokkos::Experimental::Graph<Exec>& graph) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    Kokkos::utils::callbacks::dispatch(
        GraphCreateEvent{
            .graph = get_graph_impl_ptr(graph.root_node()),
            .dev_id = Kokkos::Tools::Experimental::device_id(graph.get_device_handle().m_exec),
            .event_id = Kokkos::utils::callbacks::get_next_event_id()});
#endif
}

//! Create a graph and record the associated event with @ref graph_create_event.
template <Kokkos::ExecutionSpace Exec, typename... Args>
auto create_graph(const Kokkos::Impl::DeviceHandle<Exec>& device_handle, Args&&... args) {
    Kokkos::Experimental::Graph<Exec> graph{device_handle, std::forward<Args>(args)...};
    graph_create_event(graph);
    return graph;
}

/**
 * @brief Record an event for a @p node added after @p predecessor.
 *
 * @todo There is no way to tell if the predecessor is a root node.
 */
template <bool IsRoot, Kokkos::ExecutionSpace Exec, typename PredecessorRef, typename NodeRef>
void graph_add_node_event(
    const Kokkos::Impl::DeviceHandle<Exec>& handle,
    const PredecessorRef& predecessor,
    const NodeRef& node) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    Kokkos::utils::callbacks::dispatch(
        GraphAddNodeEvent{
            .graph = get_graph_impl_ptr(predecessor),
            .predecessor = IsRoot ? nullptr : get_node_ptr(predecessor),
            .node = get_node_ptr(node),
            .dev_id = Kokkos::Tools::Experimental::device_id(handle.m_exec)});
#endif
}

//! Record a @ref GraphInstantiateEvent event.
template <Kokkos::ExecutionSpace Exec>
void graph_instantiate_event(const Kokkos::Experimental::Graph<Exec>& graph) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    Kokkos::utils::callbacks::dispatch(GraphInstantiateEvent{.graph = get_graph_impl_ptr(graph.root_node())});
#endif
}


//! Record a @ref GraphSubmitEvent event.
template <Kokkos::ExecutionSpace Exec>
void graph_submit_event(const Kokkos::Experimental::Graph<Exec>& graph, const Exec& exec) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    Kokkos::utils::callbacks::dispatch(
        GraphSubmitEvent{
            .graph = get_graph_impl_ptr(graph.root_node()), .dev_id = Kokkos::Tools::Experimental::device_id(exec)});
#endif
}

//! Submit a graph and record the associated event with @ref graph_submit_event.
template <Kokkos::ExecutionSpace Exec>
void submit_graph(const Kokkos::Experimental::Graph<Exec>& graph, const Exec& exec) {
    graph_submit_event(graph, exec);
    graph.submit(exec);
}

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_EVENTS_HPP
