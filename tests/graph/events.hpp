#ifndef KOKKOS_EXECUTION_TESTS_GRAPH_EVENTS_HPP
#define KOKKOS_EXECUTION_TESTS_GRAPH_EVENTS_HPP

#include "kokkos-utils/callbacks/Helpers.hpp"

#include "kokkos-execution/graph/events.hpp"

namespace Tests::GraphImpl {

DEFINE_EVENT_MATCHER_IN(Kokkos::Execution::GraphImpl, GraphAddAggregateNodeEvent)
DEFINE_EVENT_MATCHER_IN(Kokkos::Execution::GraphImpl, GraphAddNodeEvent)
DEFINE_EVENT_MATCHER_IN(Kokkos::Execution::GraphImpl, GraphCreateEvent)
DEFINE_EVENT_MATCHER_IN(Kokkos::Execution::GraphImpl, GraphInstantiateEvent)
DEFINE_EVENT_MATCHER_IN(Kokkos::Execution::GraphImpl, GraphSubmitEvent)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_GRAPH_CREATE(_device_handle_)                                                                      \
    AGraphCreateEvent(                                                                                                 \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphCreateEvent::dev_id,                                                   \
            ::testing::Eq(Kokkos::Tools::Experimental::device_id(_device_handle_.m_exec))))

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_GRAPH_ADDNODE(_graph_create_event_variant_, _device_handle_, _predecessor_)                        \
    AGraphAddNodeEvent(                                                                                                \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphAddNodeEvent::dev_id,                                                  \
            ::testing::Eq(Kokkos::Tools::Experimental::device_id(_device_handle_.m_exec))),                            \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphAddNodeEvent::graph,                                                   \
            ::testing::Eq(                                                                                             \
                std::get<Kokkos::Execution::GraphImpl::GraphCreateEvent>(_graph_create_event_variant_).graph)),        \
        ::testing::Field(&Kokkos::Execution::GraphImpl::GraphAddNodeEvent::predecessor, ::testing::Eq(_predecessor_)))

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_GRAPH_ADD_AGGREGATE_NODE(_graph_create_event_variant_, ...)                                        \
    AGraphAddAggregateNodeEvent(                                                                                       \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphAddAggregateNodeEvent::graph,                                          \
            ::testing::Eq(                                                                                             \
                std::get<Kokkos::Execution::GraphImpl::GraphCreateEvent>(_graph_create_event_variant_).graph)),        \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphAddAggregateNodeEvent::predecessors,                                   \
            ::testing::ElementsAre(__VA_ARGS__)))

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_GRAPH_NODE_OF(_graph_add_node_event_)                                                              \
    std::get<Kokkos::Execution::GraphImpl::GraphAddNodeEvent>(_graph_add_node_event_).node

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_GRAPH_AGGREGATE_NODE_OF(_graph_add_aggregate_node_event_)                                          \
    std::get<Kokkos::Execution::GraphImpl::GraphAddAggregateNodeEvent>(_graph_add_aggregate_node_event_).node

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_GRAPH_INSTANTIATE(_graph_create_event_variant_)                                                    \
    AGraphInstantiateEvent(                                                                                            \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphInstantiateEvent::graph,                                               \
            ::testing::Eq(                                                                                             \
                std::get<Kokkos::Execution::GraphImpl::GraphCreateEvent>(_graph_create_event_variant_).graph)))

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define MATCHER_FOR_GRAPH_SUBMIT(_exec_, _graph_create_event_variant_)                                                 \
    AGraphSubmitEvent(                                                                                                 \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphSubmitEvent::dev_id,                                                   \
            ::testing::Eq(Kokkos::Tools::Experimental::device_id(_exec_))),                                            \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphSubmitEvent::graph,                                                    \
            ::testing::Eq(                                                                                             \
                std::get<Kokkos::Execution::GraphImpl::GraphCreateEvent>(_graph_create_event_variant_).graph)))


//! Similar to @ref EventDiscardMatcher, for graph-related events.
template <Kokkos::ExecutionSpace Exec>
struct GraphEventDiscardMatcher {
    /**
     * Filter out @ref Kokkos::utils::callbacks::BeginFenceEvent triggered by:
     *  * https://github.com/kokkos/kokkos/blob/10e9786f862733fc7c7a23d5e998e3d161dd7b70/core/src/impl/Kokkos_Default_Graph_Impl.hpp#L144
     *  * https://github.com/kokkos/kokkos/blob/10e9786f862733fc7c7a23d5e998e3d161dd7b70/core/src/impl/Kokkos_Default_Graph_Impl.hpp#L154
     *  * https://github.com/kokkos/kokkos/blob/10e9786f862733fc7c7a23d5e998e3d161dd7b70/core/src/impl/Kokkos_Default_GraphNode_Impl.hpp#L124
     *  * https://github.com/kokkos/kokkos/blob/10e9786f862733fc7c7a23d5e998e3d161dd7b70/core/src/HIP/Kokkos_HIP_Graph_Impl.hpp#L100
     */
    constexpr bool operator()(const Kokkos::utils::callbacks::BeginFenceEvent& event) const {
        return event.name != "Kokkos::DefaultGraph::submit: fencing before launching graph nodes"
            && event.name != "Kokkos::DefaultGraph::submit: fencing before ending graph submit"
            && event.name != "Kokkos::DefaultGraphNode::execute_node: sync with predecessors"
            && event.name != "Kokkos::GraphImpl::~GraphImpl: Graph Destruction";
    }

    template <Kokkos::utils::callbacks::Event EventType>
    constexpr bool operator()(const EventType&) const {
        return true;
    }
};

} // namespace Tests::GraphImpl

#endif // KOKKOS_EXECUTION_TESTS_GRAPH_EVENTS_HPP
