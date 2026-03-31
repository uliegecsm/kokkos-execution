#ifndef KOKKOS_EXECUTION_IMPL_SENDER_INTROSPECTION_HPP
#define KOKKOS_EXECUTION_IMPL_SENDER_INTROSPECTION_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Retrieve the completion scheduler for a given completion tag.
 *
 * It is heavily inspired, yet different from
 * https://github.com/NVIDIA/stdexec/blob/45c0f5803c190366a8529833901d1f6340b40d2e/include/stdexec/__detail/__schedulers.hpp#L395-L398.
 *
 * While @c stdexec::__completion_scheduler_of_t is constrained to a sender that "sends", this version does not and returns
 * what a genuine call to @c stdexec::get_completion_scheduler would.
 */
template <typename Tag, stdexec::sender Sndr, typename... Env>
using completion_scheduler_of_t =
    std::invoke_result_t<stdexec::get_completion_scheduler_t<Tag>, stdexec::env_of_t<Sndr>, Env...>;

//! Type of the execution space extracted from a sender's completion scheduler.
template <typename Sndr, typename... Env>
using exec_of_t = typename completion_scheduler_of_t<stdexec::set_value_t, Sndr, Env...>::execution_space;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SENDER_INTROSPECTION_HPP
