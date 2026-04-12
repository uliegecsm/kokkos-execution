#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/ExecutionSpace.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "tests/graph/events.hpp"

/**
 * @addtogroup unittests
 *
 * Custom graph-related events
 * ---------------------------
 *
 * This group of tests check the events from @ref kokkos-execution/graph/events.hpp.
 *
 * The tests can be found in @ref tests/graph/test_events.cpp.
 */

#if !defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH)
#    error "This is not supported."
#endif

namespace Tests::GraphImpl {

using namespace Kokkos::utils::callbacks;

class EventsTest
    : public virtual testing::Test
    , public Kokkos::utils::tests::scoped::ExecutionSpace<TEST_EXECUTION_SPACE>
    , public Kokkos::utils::tests::scoped::callbacks::Manager {
   public:
    using recorder_listener_t = RecorderListener<
        GraphEventDiscardMatcher<TEST_EXECUTION_SPACE>,
        BeginFenceEvent,
        Kokkos::Execution::GraphImpl::GraphAddNodeEvent,
        Kokkos::Execution::GraphImpl::GraphCreateEvent,
        Kokkos::Execution::GraphImpl::GraphInstantiateEvent,
        Kokkos::Execution::GraphImpl::GraphSubmitEvent
    >;

    using graph_t = Kokkos::Experimental::Graph<TEST_EXECUTION_SPACE>;
};

//! @test Check events recorded for graph creation, instantiation and asubmission.
TEST_F(EventsTest, create_instantiate_and_submit) {
    auto device_handle = Kokkos::Experimental::get_device_handle(exec);

    const auto recorded_events = recorder_listener_t::record([this, &device_handle]() {
        graph_t graph{device_handle};
        Kokkos::Execution::GraphImpl::graph_create_event(graph);

        graph.instantiate();
        Kokkos::Execution::GraphImpl::graph_instantiate_event(graph);

        graph.submit(exec);
        Kokkos::Execution::GraphImpl::graph_submit_event(graph, exec);
    });
    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_INSTANTIATE(recorded_events.at(0)),
            MATCHER_FOR_GRAPH_SUBMIT(exec, recorded_events.at(0))));
}

//! @test Check events recorded for graph nodes.
TEST_F(EventsTest, create_and_add_nodes) {
    auto device_handle = Kokkos::Experimental::get_device_handle(exec);

    const auto recorded_events = recorder_listener_t::record([&device_handle]() {
        const graph_t graph{device_handle};
        Kokkos::Execution::GraphImpl::graph_create_event(graph);

        const auto root = graph.root_node();

        const auto node_A = root.then(Kokkos::Experimental::node_props(device_handle), KOKKOS_LAMBDA(){});
        Kokkos::Execution::GraphImpl::graph_add_node_event<true>(device_handle, root, node_A);

        const auto node_B = node_A.then(Kokkos::Experimental::node_props(device_handle), KOKKOS_LAMBDA(){});
        Kokkos::Execution::GraphImpl::graph_add_node_event<false>(device_handle, node_A, node_B);
    });
    ASSERT_THAT(
        recorded_events,
        testing::ElementsAre(
            MATCHER_FOR_GRAPH_CREATE(device_handle),
            MATCHER_FOR_GRAPH_ADDNODE(recorded_events.at(0), device_handle, nullptr),
            MATCHER_FOR_GRAPH_ADDNODE(
                recorded_events.at(0), device_handle, MATCHER_FOR_GRAPH_NODE_OF(recorded_events.at(1)))));
}

} // namespace Tests::GraphImpl
