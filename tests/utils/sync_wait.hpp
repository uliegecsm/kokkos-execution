#ifndef KOKKOS_EXECUTION_TESTS_UTILS_SYNC_WAIT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_SYNC_WAIT_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-utils/callbacks/RecorderListener.hpp"

namespace Tests::Utils {

template <
    stdexec::__is_instance_of<Kokkos::utils::callbacks::RecorderListener> RecorderListenerType,
    stdexec::sender Sndr
>
auto record_sync_wait(Sndr&& sndr) {
    return RecorderListenerType::record(
        [sndr = std::forward<Sndr>(sndr)]() mutable { // NOLINT(performance-move-const-arg)
            stdexec::sync_wait(std::move(sndr));      // NOLINT(performance-move-const-arg)
        });
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_SYNC_WAIT_HPP
