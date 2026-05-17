#ifndef KOKKOS_EXECUTION_IMPL_SYCL_DEPENDENCY_HPP
#define KOKKOS_EXECUTION_IMPL_SYCL_DEPENDENCY_HPP

#include "kokkos-execution/impl/event.hpp"

/**
 * @file
 *
 * Specialization of types from @ref kokkos-execution/impl/dependency.hpp for @c Kokkos::SYCL.
 */

namespace Kokkos::Execution::Impl {

template <>
struct HasExecWaitEvent<Kokkos::SYCL> : std::true_type { };

template <>
struct DependencyWithEvent<Kokkos::SYCL> {
    Event<Kokkos::SYCL> event{};

    DependencyWithEvent(const Kokkos::SYCL& exec_to, const Kokkos::SYCL& exec_from) {
        if (exec_from != exec_to) {
            record(event, exec_from);
            wait(exec_to, event);
        }
    }
};

template <>
struct Dependency<Kokkos::SYCL, Kokkos::SYCL> : public DependencyWithEvent<Kokkos::SYCL> {
    using DependencyWithEvent<Kokkos::SYCL>::DependencyWithEvent;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SYCL_DEPENDENCY_HPP
