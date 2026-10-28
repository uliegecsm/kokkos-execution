#ifndef KOKKOS_EXECUTION_IMPL_TYPE_INFO_HPP
#define KOKKOS_EXECUTION_IMPL_TYPE_INFO_HPP

/**
 * @file
 *
 * If reflection is available, use @c std::meta::display_string_of.
 * Otherwise, use @c Kokkos::Impl::TypeInfo::name.
 */

#if !defined(__cpp_impl_reflection)
#    include "Kokkos_TypeInfo.hpp"
#else
#    include <meta>
#endif

namespace Kokkos::Execution::Impl {

template <typename T>
constexpr auto display_string_of() noexcept -> std::string_view {
#if !defined(__cpp_impl_reflection)
    return Kokkos::Impl::TypeInfo<T>::name();
#else
    return std::meta::display_string_of(^^T);
#endif
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_TYPE_INFO_HPP
