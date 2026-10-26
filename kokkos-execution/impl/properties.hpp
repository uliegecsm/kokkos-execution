#ifndef KOKKOS_EXECUTION_IMPL_PROPERTIES_HPP
#define KOKKOS_EXECUTION_IMPL_PROPERTIES_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

/**
 * @file
 *
 * This file contains utilities to operate on operation properties.
 *
 * Do I need a property or @c stdexec::env?
 * ----------------------------------------
 *
 * If the property presence is "contextual" (*e.g.* it depends from where it runs, how the sender is connected, and so on),
 * use @c stdexec::write_env and queries @cite eel-write-env.
 * Such property could by a "dry-run" or a launch priority @cite cccl-launch-priority.
 *
 * If the property presence is "intrinsic" to the operation (*i.e.* it describes the operation, whatever the context in which it will run),
 * use a property *attached to the sender object itself* (such as @ref Kokkos::Execution::Impl::ParallelForData).
 * Such property is the label.
 *
 * Any property that the @c Kokkos execution policies support should not be re-implemented.
 * The rule is that @c Kokkos execution-policy-level knobs stay in the execution policy type.
 */

namespace Kokkos::Execution::Impl {

//! Type list of properties.
template <typename... Properties>
struct properties { };

namespace detail {

template <template <typename...> typename T, typename... Args>
consteval void deduce_from_base(const T<Args...>&);

template <template <typename...> typename T, typename... Args>
constexpr T<Args...>& as_base(T<Args...>& base) noexcept {
    return base;
}

template <template <typename...> typename T, typename... Args>
constexpr const T<Args...>& as_base(const T<Args...>& base) noexcept {
    return base;
}

template <typename Property, typename Data>
constexpr Property& as_base(Data& data) noexcept {
    return data;
}

template <typename Property, typename Data>
constexpr const Property& as_base(const Data& data) noexcept {
    return data;
}

} // namespace detail

//! Return @c true when @p Data derives from the property @p Property.
template <typename Data, typename Property>
consteval bool has_property() {
    return std::derived_from<Data, Property>;
}

//! Return @c true when @p Data derives from the templated property @p Property.
template <typename Data, template <typename...> typename Property>
consteval bool has_property() {
    return requires(const Data& data) { detail::deduce_from_base<Property>(data); };
}

template <typename Property, typename Data>
requires(has_property<std::remove_cvref_t<Data>, Property>())
constexpr decltype(auto) get_property(Data&& data) noexcept { // NOLINT(cppcoreguidelines-missing-std-forward)
    return stdexec::__forward_like<Data>(detail::as_base<Property>(data).m_value);
}

template <template <typename...> typename Property, typename Data>
requires(has_property<std::remove_cvref_t<Data>, Property>())
constexpr decltype(auto) get_property(Data&& data) noexcept { // NOLINT(cppcoreguidelines-missing-std-forward)
    return stdexec::__forward_like<Data>(detail::as_base<Property>(data).m_value);
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_PROPERTIES_HPP
