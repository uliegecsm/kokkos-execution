#ifndef KOKKOS_EXECUTION_IMPL_HIP_DEPENDENCY_HPP
#define KOKKOS_EXECUTION_IMPL_HIP_DEPENDENCY_HPP

#include "kokkos-execution/impl/event.hpp"

/**
 * @file
 *
 * Specialization of types from @ref kokkos-execution/impl/dependency.hpp for @c Kokkos::HIP.
 */

namespace Kokkos::Execution::Impl {

template <>
struct HasExecWaitEvent<Kokkos::HIP> : std::true_type { };

template <>
struct DependencyWithEvent<Kokkos::HIP> {
    Event<Kokkos::HIP> event{};

    DependencyWithEvent(const Kokkos::HIP& exec_to, const Kokkos::HIP& exec_from) {
        if (exec_from != exec_to) {
            record(event, exec_from);
            wait(exec_to, event);
        }
    }
};

template <>
struct Dependency<Kokkos::HIP, Kokkos::HIP> : public DependencyWithEvent<Kokkos::HIP> {
    using DependencyWithEvent<Kokkos::HIP>::DependencyWithEvent;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_HIP_DEPENDENCY_HPP
