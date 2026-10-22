#ifndef KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP
#define KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP

#include "kokkos-utils/impl/type_list.hpp"

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::GraphImpl {

template <Kokkos::ExecutionSpace Exec, typename Label>
using node_props_t = typename Kokkos::Impl::NodeCtorProps<Kokkos::Impl::DeviceHandle<Exec>, Label>::uniform_type;

//! Retrieve the @c Kokkos::Impl::DeviceHandle type from a node properties type.
template <Kokkos::Impl::NodeProperties NodeProps>
using device_handle_of_t = Kokkos::utils::impl::type_list_at_t<
    0,
    Kokkos::Impl::filter_type_list_t<Kokkos::Impl::is_device_handle, typename NodeProps::properties_value_type_list_t>
>;

template <typename>
struct ExecOf;

/**
 * @brief Retrieve the execution space type from a @c Kokkos::Impl::DeviceHandle.
 *
 * @todo Enrich the @c Kokkos::Impl::DeviceHandle type with a @c execution_space alias.
 */
template <Kokkos::ExecutionSpace Exec>
struct ExecOf<Kokkos::Impl::DeviceHandle<Exec>> : std::type_identity<Exec> { };

template <typename T>
using exec_of_t = typename ExecOf<T>::type;

//! Retrieve the label type from a node properties type.
template <Kokkos::Impl::NodeProperties NodeProps>
using label_of_t = Kokkos::utils::impl::type_list_at_t<
    0,
    Kokkos::Impl::filter_type_list_t<Kokkos::Impl::is_view_label, typename NodeProps::properties_value_type_list_t>
>;
} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP
