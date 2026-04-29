#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "kokkos-utils/callbacks/RecorderListener.hpp"
#include "kokkos-utils/tests/scoped/callbacks/Manager.hpp"

#include "kokkos-execution/graph.hpp"

#include "tests/graph/events.hpp"
#include "tests/utils/graph_context.hpp"

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
    : public Tests::Utils::GraphContextTest<TEST_EXECUTION_SPACE>
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
};

//! @test Test which calls are @c noexcept.
consteval bool test_noexcept() {
    static_assert(
        noexcept(Kokkos::Execution::GraphImpl::get_graph_impl_ptr(std::declval<const EventsTest::graph_t::root_t&>())));
    static_assert(
        noexcept(Kokkos::Execution::GraphImpl::get_node_ptr(std::declval<const EventsTest::graph_t::root_t&>())));

    static_assert(
        !noexcept(Kokkos::Execution::GraphImpl::graph_create_event(std::declval<const EventsTest::graph_t&>())));
    static_assert(
        !noexcept(Kokkos::Execution::GraphImpl::create_graph(std::declval<const EventsTest::device_handle_t&>())));

    static_assert(!noexcept(Kokkos::Execution::GraphImpl::graph_add_node_event<true>(
        std::declval<const EventsTest::graph_t::root_t&>(), std::declval<const EventsTest::graph_t::root_t&>())));

    static_assert(
        !noexcept(Kokkos::Execution::GraphImpl::graph_instantiate_event(std::declval<const EventsTest::graph_t&>())));

    static_assert(!noexcept(Kokkos::Execution::GraphImpl::graph_submit_event(
        std::declval<const EventsTest::graph_t&>(), std::declval<const TEST_EXECUTION_SPACE&>())));

    static_assert(!noexcept(Kokkos::Execution::GraphImpl::submit_graph(
        std::declval<const EventsTest::graph_t&>(), std::declval<const TEST_EXECUTION_SPACE&>())));

    return true;
}
static_assert(test_noexcept());

//! @test Check events recorded for graph creation, instantiation and submission.
TEST_F(EventsTest, create_instantiate_and_submit) {
    const auto recorded_events = recorder_listener_t::record([this]() {
        graph_t graph = Kokkos::Execution::GraphImpl::create_graph(device_handle);

        graph.instantiate();
        Kokkos::Execution::GraphImpl::graph_instantiate_event(graph);

        Kokkos::Execution::GraphImpl::submit_graph(graph, exec);
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
    const auto recorded_events = recorder_listener_t::record([this]() {
        const graph_t graph{device_handle};
        Kokkos::Execution::GraphImpl::graph_create_event(graph);

        const auto root = graph.root_node();

        const auto node_A = root.then(Kokkos::Experimental::node_props(device_handle), KOKKOS_LAMBDA(){});
        Kokkos::Execution::GraphImpl::graph_add_node_event<true>(root, node_A);

        const auto node_B = node_A.then(Kokkos::Experimental::node_props(device_handle), KOKKOS_LAMBDA(){});
        Kokkos::Execution::GraphImpl::graph_add_node_event<false>(node_A, node_B);
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
