/**
 * @file @c Kokkos ships with an implementation of @c mdspan, but does not
 *       ship it as a module. Therefore, their implementation is unreachable for us,
 *       and we need to build a custom module to access it.
 */

module;

#include "Kokkos_Core.hpp"

export module kokkos.mdspan;

export namespace Kokkos::detail {
using ::Kokkos::detail::constant_wrapper;
using ::Kokkos::detail::is_constant_wrapper;
using ::Kokkos::detail::is_range_slice;
using ::Kokkos::detail::impl_is_extents_v;
} // namespace Kokkos::detail
