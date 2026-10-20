#ifndef KOKKOS_EXECUTION_IMPL_ITERATION_SPACE_HPP
#define KOKKOS_EXECUTION_IMPL_ITERATION_SPACE_HPP

#include <type_traits>

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::Impl {

template <typename T>
concept range_slice = Kokkos::detail::is_range_slice<T>::value;

template <typename T>
concept range_slice_with_default_stride = range_slice<T>
                                       && Kokkos::detail::is_constant_wrapper<decltype(std::declval<T>().stride)>
                                       && requires(const T& rgsl) { rgsl.stride == 1; };

//! Concept that constrains types that may represent an iteration space.
template <typename T>
concept iteration_space = range_slice_with_default_stride<T>;

template <typename T>
concept execution_policy_or_iteration_space = Kokkos::ExecutionPolicy<T> || iteration_space<T>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_ITERATION_SPACE_HPP
