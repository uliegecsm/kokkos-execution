#ifndef KOKKOS_EXECUTION_IMPL_EXTENTS_HPP
#define KOKKOS_EXECUTION_IMPL_EXTENTS_HPP

#include <type_traits>

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::Impl {

template <typename T>
struct IsExtents : std::false_type { };

template <typename IndexType, size_t... Extents>
struct IsExtents<Kokkos::extents<IndexType, Extents...>> : std::true_type { };

template <typename T>
concept extents = IsExtents<T>::value;

template <typename T>
concept execution_policy_or_extents = Kokkos::ExecutionPolicy<T> || extents<T>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_EXTENTS_HPP
