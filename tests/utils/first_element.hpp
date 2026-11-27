#ifndef KOKKOS_EXECUTION_TESTS_UTILS_FIRST_ELEMENT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_FIRST_ELEMENT_HPP

#include "Kokkos_Macros.hpp"

namespace Tests::Utils {

namespace Impl {

//! Always 0.
template <std::size_t>
constexpr std::integral_constant<std::size_t, 0> zero{};

//! When accessing through @c operator(), such as for @c Kokkos::View.
template <typename DataType, std::size_t... Idx>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto)
    first_element(const DataType& data, std::index_sequence<Idx...>) noexcept requires requires { data(zero<Idx>...); }
{
    return data(zero<Idx>...);
}

//! When accessing through @c operator[], such as for @c std::mdspan.
template <typename DataType, std::size_t... Idx>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto)
    first_element(const DataType& data, std::index_sequence<Idx...>) noexcept
    requires requires { data.operator[](zero<Idx>...); }
{
    return data.operator[](zero<Idx>...);
}

} // namespace Impl

//! Reference to the element at index {0, ..., 0} of a @c Kokkos::View or a @c std::mdspan.
template <typename DataType>
KOKKOS_INLINE_FUNCTION constexpr decltype(auto) first_element(const DataType& data) noexcept {
    return Impl::first_element(data, std::make_index_sequence<DataType::rank()>{});
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_FIRST_ELEMENT_HPP
