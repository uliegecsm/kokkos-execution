#ifndef KOKKOS_EXECUTION_TESTS_UTILS_STDEXEC_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_STDEXEC_HPP

#include <concepts>

namespace Tests::Utils {

//! See https://github.com/NVIDIA/stdexec/pull/1873#discussion_r2834863237.
template <typename... Args>
using basic_sender_t = typename stdexec::__basic_sender<Args...>::type;

template <typename Sndr, typename Signatures, typename... Env>
concept has_completion_signatures = stdexec::__mset_eq<Signatures, stdexec::__completion_signatures_of_t<Sndr, Env...>>;

template <typename Sndr, typename Tag, typename... Env>
concept has_completion_scheduler_for =
    std::invocable<stdexec::get_completion_scheduler_t<Tag>, const stdexec::env_of_t<Sndr>&, const Env&...>;

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_STDEXEC_HPP
