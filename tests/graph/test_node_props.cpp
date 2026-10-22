#include "gtest/gtest.h"

#include "kokkos-execution/graph.hpp"

/**
 * @addtogroup unittests
 *
 * Utilities related to node properties
 * ------------------------------------
 *
 * This group of tests check the utilities from @ref kokkos-execution/graph/node_props.hpp.
 *
 * The tests can be found in @ref tests/graph/test_node_props.cpp.
 */


namespace Tests::GraphImpl {

using node_props_t = Kokkos::Impl::NodeCtorProps<Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>, std::string>;

//! @test Check that @ref Kokkos::Execution::GraphImpl::node_props_t works as expected.
consteval bool test_node_props_t() {
    static_assert(
        std::same_as<Kokkos::Execution::GraphImpl::node_props_t<TEST_EXECUTION_SPACE, std::string>, node_props_t>);

    return true;
}

static_assert(test_node_props_t());

//! @test Check that @ref Kokkos::Execution::GraphImpl::device_handle_of_t works as expected.
consteval bool test_device_handle_of_t() {
    static_assert(std::same_as<
                  Kokkos::Execution::GraphImpl::device_handle_of_t<node_props_t>,
                  Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>
    >);

    return true;
}

static_assert(test_device_handle_of_t());

//! @test Check that @ref Kokkos::Execution::GraphImpl::exec_of_t works as expected.
consteval bool test_exec_of_t() {
    static_assert(std::same_as<
                  Kokkos::Execution::GraphImpl::exec_of_t<Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>>,
                  TEST_EXECUTION_SPACE
    >);

    return true;
}

static_assert(test_exec_of_t());

//! @test Check that @ref Kokkos::Execution::GraphImpl::label_of_t works as expected.
consteval bool test_label_of_t() {
    static_assert(std::same_as<Kokkos::Execution::GraphImpl::label_of_t<node_props_t>, std::string>);

    return true;
}

static_assert(test_label_of_t());

//! Because we need at least one discoverable runtime test per unit.
TEST(is, useless) {
}

} // namespace Tests::GraphImpl
