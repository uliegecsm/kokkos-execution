#ifndef KOKKOS_EXECUTION_IMPL_CUDA_DEPENDENCY_HPP
#define KOKKOS_EXECUTION_IMPL_CUDA_DEPENDENCY_HPP

#include "kokkos-execution/impl/event.hpp"

/**
 * @file
 *
 * Specialization of types from @ref kokkos-execution/impl/dependency.hpp for @c Kokkos::Cuda.
 */

namespace Kokkos::Execution::Impl {

template <>
struct HasExecWaitEvent<Kokkos::Cuda> : std::true_type { };

template <>
struct DependencyWithEvent<Kokkos::Cuda> {
    Event<Kokkos::Cuda> event{};

    DependencyWithEvent(const Kokkos::Cuda& exec_to, const Kokkos::Cuda& exec_from) {
        if (exec_from != exec_to) {
            record(event, exec_from);
            wait(exec_to, event);
        }
    }
};

template <>
struct Dependency<Kokkos::Cuda, Kokkos::Cuda> : public DependencyWithEvent<Kokkos::Cuda> {
    using DependencyWithEvent<Kokkos::Cuda>::DependencyWithEvent;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_CUDA_DEPENDENCY_HPP
