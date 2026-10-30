#ifndef KOKKOS_EXECUTION_IMPL_LABEL_HPP
#define KOKKOS_EXECUTION_IMPL_LABEL_HPP

#if defined(__cpp_impl_reflection)
#    include <meta>
#endif

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/fixed_string.hpp"
#include "kokkos-execution/impl/properties.hpp"

/**
 * @file
 *
 * This file contains the label property and related helpers.
 *
 * If the incoming type is already a @ref Kokkos::Execution::Impl::Label,
 * it should be kept as is.
 * Otherwise, if the type is **not** convertible to @c std::string, a new @c std::string will be materialized.
 *
 * In order to pass a @c std::string_view, the user has to use @ref Kokkos::Execution::label,
 * thus enforcing that the @c std::string_view points to static storage.
 */

namespace Kokkos::Execution {

namespace Impl {

template <typename T>
struct Label;

//! Constrain the accepted label types.
template <typename T>
concept label = std::convertible_to<T, std::string> || stdexec::__is_instance_of<T, Label>;

//! Label property.
template <typename T>
struct Label {
    using type = T;

    T m_value;
};

//! @name CTAD for @ref Label.
///@{
template <std::convertible_to<std::string> T>
Label(T) -> Label<std::string>;

template <typename T>
Label(T) -> Label<T>;
///@}

template <typename Data>
concept labeled = has_property<Data, Label>();

/**
 * @brief Convert to a value usable as a @c std::string.
 *
 * If @p value is (implicitly) convertible to a @c std::string, it is forwarded as-is.
 *
 * @todo While @c Kokkos::parallel_for allows for convertible-to @c std::string labels,
 *       @c Kokkos::parallel_reduce does not.
 *       See also:
 *          - https://github.com/kokkos/kokkos/blob/2c577ed795cf8eb36a665d1c15d722ef6461d1b1/core/src/Kokkos_Parallel_For.hpp#L50
 *          - https://github.com/kokkos/kokkos/blob/2c577ed795cf8eb36a665d1c15d722ef6461d1b1/core/src/Kokkos_Parallel_Reduce.hpp#L329
 */
template <typename T>
requires std::convertible_to<T, std::string>
constexpr auto as_convertible_to_string(T&& value) STDEXEC_AUTO_RETURN(std::forward<T>(value));

template <typename T>
requires(!std::convertible_to<T, std::string>)
constexpr auto
    as_convertible_to_string(T&& value) STDEXEC_AUTO_RETURN(static_cast<std::string>(std::forward<T>(value)));

} // namespace Impl

/**
 * @name Marker for a string literal.
 *
 * Pre-C++26 has no way to determine if an incoming pointer is a string literal.
 * See also @cite cppreference-is-string-literal.
 */
///@{
#if !defined(__cpp_lib_define_static)
consteval Impl::Label<std::string_view> label(std::string_view str) noexcept {
    return {str};
}
#else
consteval Impl::Label<std::string_view> label(std::string_view str) noexcept {
    return {std::define_static_string(std::move(str))};
}
#endif
///@}

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_IMPL_LABEL_HPP
