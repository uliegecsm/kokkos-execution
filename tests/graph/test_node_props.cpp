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

//! @test Check @ref Kokkos::Execution::GraphImpl::get_properties.
TEST(NodeProps, get_properties) {
    auto props = Kokkos::Experimental::node_props(Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>{}, "my label");

    //! An lvalue props container.
    const auto [device_handle_lvalue, label_lvalue] =
        Kokkos::Execution::GraphImpl::get_properties<Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>, std::string>(
            props);

    static_assert(std::same_as<decltype(device_handle_lvalue), Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>&>);
    static_assert(std::same_as<decltype(label_lvalue), std::string&>);

    Kokkos::Impl::get_property<std::string>(props) = "newer label";

    ASSERT_EQ(device_handle_lvalue, Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>{});
    ASSERT_EQ(label_lvalue, "newer label");

    //! A const lvalue props container.
    const auto [device_handle_const_lvalue, label_const_lvalue] =
        Kokkos::Execution::GraphImpl::get_properties<Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>, std::string>(
            std::as_const(props));

    static_assert(
        std::same_as<decltype(device_handle_const_lvalue), const Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>&>);
    static_assert(std::same_as<decltype(label_const_lvalue), const std::string&>);

    //! A prvalue props container.
    const auto [device_handle_prvalue, label_prvalue] =
        Kokkos::Execution::GraphImpl::get_properties<Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>, std::string>(
            std::move(props));

    static_assert(
        std::same_as<decltype(device_handle_prvalue), const Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>>);
    static_assert(std::same_as<decltype(label_prvalue), const std::string>);

    ASSERT_EQ(device_handle_prvalue, Kokkos::Impl::DeviceHandle<TEST_EXECUTION_SPACE>{});
    ASSERT_EQ(label_prvalue, "newer label");
}

} // namespace Tests::GraphImpl
