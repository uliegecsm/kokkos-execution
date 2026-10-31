#ifndef KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP
#define KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/label.hpp"
#include "kokkos-execution/impl/properties.hpp"
#include "kokkos-execution/impl/type_info.hpp"

namespace Kokkos::Execution::GraphImpl {

template <Kokkos::ExecutionSpace Exec, typename Label>
using node_props_t = typename Kokkos::Impl::NodeCtorProps<Kokkos::Impl::DeviceHandle<Exec>, Label>::uniform_type;

template <Kokkos::ExecutionSpace Exec, typename Data>
requires(Impl::labeled<std::remove_cvref_t<Data>>)
constexpr auto make_node_props(const Exec& exec, Data&& data) STDEXEC_AUTO_RETURN(
    Kokkos::Experimental::node_props(
        Kokkos::Experimental::get_device_handle(exec),
        Impl::as_convertible_to_string(Impl::get_property<Impl::Label>(std::forward<Data>(data)))))

template <Kokkos::ExecutionSpace Exec, typename Data>
constexpr auto make_node_props(const Exec& exec, Data&&)
    STDEXEC_AUTO_RETURN(Kokkos::Experimental::node_props(Kokkos::Experimental::get_device_handle(exec)))

template <Kokkos::ExecutionSpace Exec, typename Data>
using make_node_props_t = decltype(make_node_props(std::declval<const Exec&>(), std::declval<Data&&>()));

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

/**
 * @brief If the node properties contain a label, return it. Otherwise, use the display string of @p T.
 *
 * @note It avoids instantiating @ref Kokkos::Execution::Impl::display_string_of if it is not necessary.
 */
template <typename T, typename PropsContainer>
requires Kokkos::Impl::NodeProperties<std::remove_cvref_t<PropsContainer>>
decltype(auto) get_label_or_display_string_of(PropsContainer&& props) {
    if constexpr (std::remove_cvref_t<PropsContainer>::template has<std::string>) {
        return Kokkos::Impl::get_property<std::string>(std::forward<PropsContainer>(props));
    } else {
        return Kokkos::Execution::Impl::display_string_of<T>();
    }
}

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_NODE_PROPS_HPP
