#ifndef KOKKOS_EXECUTION_IMPL_HPX_DEPENDENCY_HPP
#define KOKKOS_EXECUTION_IMPL_HPX_DEPENDENCY_HPP

/**
 * @file
 *
 * Specialization of types from @ref kokkos-execution/impl/dependency.hpp for @c Kokkos::Experimental::HPX.
 */

namespace Kokkos::Execution::Impl {

template <>
struct HasExecWaitEvent<Kokkos::Experimental::HPX> : std::true_type { };

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HPX_DEPENDENCY_HPP
