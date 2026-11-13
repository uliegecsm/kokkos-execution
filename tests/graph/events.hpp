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
#define MATCHER_FOR_GRAPH_ADDNODE(_graph_create_event_variant_, _device_handle_, _predecessor_, ...)                   \
    AGraphAddNodeEvent(                                                                                                \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphAddNodeEvent::dev_id,                                                  \
            ::testing::Eq(Kokkos::Tools::Experimental::device_id(_device_handle_.m_exec))),                            \
        ::testing::Field(                                                                                              \
            &Kokkos::Execution::GraphImpl::GraphAddNodeEvent::graph,                                                   \
            ::testing::Eq(                                                                                             \
                std::get<Kokkos::Execution::GraphImpl::GraphCreateEvent>(_graph_create_event_variant_).graph)),        \
        ::testing::Field(&Kokkos::Execution::GraphImpl::GraphAddNodeEvent::predecessor, ::testing::Eq(_predecessor_))  \
            __VA_OPT__(                                                                                                \
                ,                                                                                                      \
                ::testing::Field(                                                                                      \
                    &Kokkos::Execution::GraphImpl::GraphAddNodeEvent::name, ::testing::Eq(std::string(__VA_ARGS__)))))

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
#define MATCHER_FOR_GRAPH_ROOT_NODE_OF(_graph_create_event_)                                                           \
    std::get<Kokkos::Execution::GraphImpl::GraphCreateEvent>(_graph_create_event_).root_node

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

} // namespace Tests::GraphImpl

#endif // KOKKOS_EXECUTION_TESTS_GRAPH_EVENTS_HPP
