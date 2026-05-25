#ifndef KOKKOS_EXECUTION_IMPL_DEPENDENCY_HPP
#define KOKKOS_EXECUTION_IMPL_DEPENDENCY_HPP

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/event.hpp"

namespace Kokkos::Execution::Impl {

template <Kokkos::ExecutionSpace Exec>
struct HasExecWaitEvent : std::false_type { };

//! Determine if the @c Kokkos backend can enqueue a wait for an event into an execution space instance.
template <typename Exec>
concept has_exec_wait_event = HasExecWaitEvent<Exec>::value;

/**
 * @brief This is the default implementation.
 *
 * Create a dependency between operations enqueud into one execution space instance @p exec_from (from any thread)
 * with those that will be enqueud into another execution space instance @p exec_to.
 *
 * The dependency results in the same semantic guarantees as a @c Kokkos fence, *i.e.* it
 * guarantees both the ordering and the side effects visibility.
 *
 * References:
 *  - https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/fence.html#semantics
 */
template <Kokkos::ExecutionSpace ExecTo, Kokkos::ExecutionSpace ExecFrom>
struct Dependency {
    Dependency(const ExecTo&, const ExecFrom& exec_from) {
        exec_from.fence(std::string(Impl::dispatch_label<ExecFrom, ": dependency">()));
    }
};

template <Kokkos::ExecutionSpace Exec>
requires(!has_exec_wait_event<Exec>)
struct Dependency<Exec, Exec> {
    Dependency(const Exec& exec_to, const Exec& exec_from) {
        if (exec_from != exec_to) {
            exec_from.fence(std::string(Impl::dispatch_label<Exec, ": dependency">()));
        }
    }
};

template <Kokkos::ExecutionSpace Exec>
requires has_exec_wait_event<Exec>
struct Dependency<Exec, Exec> {
    Event<Exec> event{};

    Dependency(const Exec& exec_to, const Exec& exec_from) {
        if (exec_from != exec_to) {
            record(event, exec_from);
            wait(exec_to, event);
        }
    }
};

} // namespace Kokkos::Execution::Impl

#if defined(KOKKOS_ENABLE_CUDA)
#    include "kokkos-execution/impl/Cuda/dependency.hpp"
#endif
#if defined(KOKKOS_ENABLE_HIP)
#    include "kokkos-execution/impl/HIP/dependency.hpp"
#endif
#if defined(KOKKOS_ENABLE_SYCL)
#    include "kokkos-execution/impl/SYCL/dependency.hpp"
#endif

#endif // KOKKOS_EXECUTION_IMPL_DEPENDENCY_HPP
