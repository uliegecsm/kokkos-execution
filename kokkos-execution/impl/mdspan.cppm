/**
 * @file @c Kokkos ships with an implementation of @c mdspan, but does not
 *       ship it as a module. Therefore, their implementation details are unreachable to us,
 *       and we need to build a custom module to access them.
 */

module;

#include "Kokkos_Core.hpp"

export module kokkos.execution.mdspan;

export namespace Kokkos::detail {
using ::Kokkos::detail::constant_wrapper;
using ::Kokkos::detail::is_constant_wrapper;
using ::Kokkos::detail::is_range_slice;
using ::Kokkos::detail::impl_is_extents_v;
} // namespace Kokkos::detail
