#ifndef KOKKOS_EXECUTION_TESTS_UTILS_STDEXEC_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_STDEXEC_HPP

#include <concepts>

#include "kokkos-execution/stdexec.hpp"

namespace Tests::Utils {

//! See https://github.com/NVIDIA/stdexec/pull/1873#discussion_r2834863237.
template <typename... Args>
using basic_sender_t = typename stdexec::__basic_sender<Args...>::type;

template <typename Sndr, typename Signatures, typename... Env>
concept has_completion_signatures = stdexec::__mset_eq<Signatures, stdexec::__completion_signatures_of_t<Sndr, Env...>>;

template <typename Sndr, typename Tag, typename... Env>
concept has_completion_scheduler_for =
    std::invocable<stdexec::get_completion_scheduler_t<Tag>, stdexec::env_of_t<Sndr>, Env...>;

/**
 * @brief A stricter variant of @c stdexec::operation_state.
 *
 * @c stdexec::operation_state intentionally omits these requirements
 * from its constraints: it checks that the type is a destructible object and has a @c start()
 * member invocable via @c stdexec::start, but leaves the @c noexcept and @c void return as *mandates*.
 * This produces targeted diagnostics when a user forgets @c noexcept on their @c start() member.
 *
 * This concept strengthens @c stdexec::operation_state by additionally *constraining* that
 * @c start() be callable on an lvalue, return @c void, and be @c noexcept, making
 * the concept a faithful predicate for "startable" given both *constraints* and *mandates* from the @c stdexec framework.
 *
 * @note See https://github.com/NVIDIA/stdexec/pull/2003 and @cite P2874R2 for the
 *       design rationale behind the mandates-vs-constraints choice in the standard.
 */
template <typename OpState>
concept operation_state = stdexec::operation_state<OpState> && requires(OpState& op_state) {
    { op_state.start() } noexcept -> std::same_as<void>;
};

/**
 * @brief Check that @c apply_sender is @c noexcept.
 *
 * Inspired by:
 *  * https://github.com/NVIDIA/stdexec/blob/45c0f5803c190366a8529833901d1f6340b40d2e/include/stdexec/__detail/__domain.hpp#L43
 *  * https://github.com/NVIDIA/stdexec/blob/45c0f5803c190366a8529833901d1f6340b40d2e/include/stdexec/__detail/__domain.hpp#L56
 */
template <typename DomainOrTag, typename... Args>
concept has_nothrow_apply_sender = requires(DomainOrTag domain_or_tag, Args&&... args) {
    { domain_or_tag.apply_sender(std::forward<Args>(args)...) } noexcept;
};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_STDEXEC_HPP
