#ifndef KOKKOS_EXECUTION_IMPL_EMPTY_HPP
#define KOKKOS_EXECUTION_IMPL_EMPTY_HPP

#include <concepts>

namespace Kokkos::Execution::Impl {

struct Empty { };

template <typename T>
concept not_empty = !std::same_as<T, Empty>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_EMPTY_HPP
