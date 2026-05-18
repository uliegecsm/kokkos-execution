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

//! Constrain a type that is a specialization of @c Kokkos::Experimental::GraphNodeRef.
template <typename T>
concept NodeRef = Kokkos::Impl::is_specialization_of_v<T, Kokkos::Experimental::GraphNodeRef>;

//! Retrieve the raw graph pointer from a node.
template <NodeRef NodeType>
auto* get_graph_impl_ptr(const NodeType& node) noexcept {
    return Kokkos::Impl::GraphAccess::get_graph_weak_ptr(node).lock().get();
}

//! Retrieve the raw node pointer.
template <NodeRef NodeType>
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
 * @todo Use https://github.com/kokkos/kokkos/pull/9170 to determine if the predecessor is a root node.
 * @todo Once https://github.com/kokkos/kokkos/pull/9137 is merged, get rid of the @p device_handle argument.
 */
template <NodeRef Predecessor, NodeRef NodeType, Kokkos::ExecutionSpace Exec>
void graph_add_node_event(
    const Predecessor& predecessor,
    const NodeType& node,
    const Kokkos::Impl::DeviceHandle<Exec>& device_handle) {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
    auto* const node_ptr = get_node_ptr(node);
    auto* const pred_ptr = get_node_ptr(predecessor);
    auto* const graph_ptr = get_graph_impl_ptr(predecessor);

    // NOLINTBEGIN(bugprone-branch-clone)
    constexpr bool is_root = []() {
        using aggregate_t = decltype(graph_ptr->create_aggregate_ptr());

        if constexpr (requires { typename std::remove_cvref_t<decltype(*pred_ptr)>::kernel_type; }) {
            using kernel_t = typename std::remove_cvref_t<decltype(*pred_ptr)>::kernel_type;
            if constexpr (Kokkos::Impl::is_graph_kernel_v<kernel_t>) {
                return false;
            } else if constexpr (Kokkos::Impl::is_graph_capture_v<kernel_t>) {
                return false;
            } else if constexpr (Kokkos::Impl::is_graph_then_host_v<kernel_t>) {
                return false;
            }
        } else if constexpr (std::same_as<Predecessor, aggregate_t>) {
            return false;
        }
        return true;
    }();
    // NOLINTEND(bugprone-branch-clone)

    Kokkos::utils::callbacks::dispatch(
        GraphAddNodeEvent{
            .graph = graph_ptr,
            .predecessor = is_root ? nullptr : pred_ptr,
            .node = node_ptr,
            .dev_id = Kokkos::Tools::Experimental::device_id(device_handle.m_exec)});
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
