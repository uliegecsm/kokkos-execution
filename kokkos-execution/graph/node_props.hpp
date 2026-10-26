#ifndef KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP
#define KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::GraphImpl {

template <Kokkos::ExecutionSpace Exec, typename Label>
using node_props_t = typename Kokkos::Impl::NodeCtorProps<Kokkos::Impl::DeviceHandle<Exec>, Label>::uniform_type;

//! Retrieve the @c Kokkos::Impl::DeviceHandle type from a node properties type.
template <Kokkos::ExecutionSpace Exec>
auto device_handle_of(const Kokkos::Impl::NodeCtorProp<Kokkos::Impl::DeviceHandle<Exec>>&)
    -> std::type_identity<Kokkos::Impl::DeviceHandle<Exec>>;

template <typename T>
using device_handle_of_t = typename decltype(device_handle_of(std::declval<const T&>()))::type;

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

template <Kokkos::Impl::ValidNodeProperty... Props, typename PropsContainer>
requires Kokkos::Impl::NodeProperties<std::remove_cvref_t<PropsContainer>>
[[nodiscard]]
constexpr auto get_properties(PropsContainer&& props) { // NOLINT(cppcoreguidelines-missing-std-forward)
    if constexpr (std::is_lvalue_reference_v<PropsContainer>) {
        return std::tie(Kokkos::Impl::get_property<Props>(props)...);
    } else {
        return std::tuple<Props...>{
            Kokkos::Impl::forward_like<PropsContainer>(Kokkos::Impl::get_property<Props>(props))...};
    }
}

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP
