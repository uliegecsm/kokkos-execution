#ifndef KOKKOS_EXECUTION_TESTS_UTILS_JUST_STOPPED_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_JUST_STOPPED_HPP

#include "kokkos-execution/stdexec.hpp"

#include "tests/utils/stdexec.hpp"

namespace Tests::Utils {

template <typename SchedulerType, typename Tag, typename... Args>
consteval bool check_continues_on_after_just_stopped() {
    using sndr_t =
        decltype(stdexec::just_stopped() | stdexec::continues_on(std::declval<SchedulerType>()) | Tag{}(std::declval<Args>()...));

    static_assert(Tests::Utils::has_completion_signatures<sndr_t, stdexec::__mset<stdexec::set_stopped_t()>>);

    static_assert(
        Tests::Utils::has_completion_signatures<sndr_t, stdexec::__mset<stdexec::set_stopped_t()>, stdexec::env<>>);

    /// Trying to pass the sender to @c stdexec::sync_wait will fail
    /// on the @c static_assert in https://github.com/NVIDIA/stdexec/blob/7c8c3c3d5a0d0b16963a00192dbfae5db9c2c627/include/stdexec/__detail/__sync_wait.hpp#L199-L219.

    return true;
}

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_JUST_STOPPED_HPP
