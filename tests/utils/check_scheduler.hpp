#ifndef KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SCHEDULER_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SCHEDULER_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Tests::Utils {

//! Check that the given type models the @c stdexec::scheduler concept.
template <stdexec::scheduler SchedulerType>
consteval bool check_scheduler() {
    /**
     * According to https://eel.is/c++draft/exec.sched#1, a valid scheduler must have a @c scheduler_concept
     * alias.
     * However, as of https://github.com/NVIDIA/stdexec/blob/0e9983599d0c95fca3fd11baa02564eb53fb14f6/include/stdexec/__detail/__schedulers.hpp#L74,
     * it is not checked by @c stdexec::scheduler.
     *
     * Related to https://github.com/NVIDIA/stdexec/issues/1406.
     */
    static_assert(std::derived_from<typename SchedulerType::scheduler_concept, stdexec::scheduler_t>);

    //! According to https://eel.is/c++draft/exec.sched#1, a @c schedule invocation must return a sender.
    static_assert(stdexec::sender<decltype(stdexec::schedule(std::declval<const SchedulerType&>()))>);

    return true;
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_CHECK_SCHEDULER_HPP
