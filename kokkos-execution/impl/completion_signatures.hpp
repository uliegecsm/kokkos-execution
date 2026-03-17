#ifndef KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNATURES_HPP
#define KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNATURES_HPP

#include "stdexec/execution.hpp"
#include <exec/completion_signatures.hpp>

namespace Kokkos::Execution::Impl {

/**
 * @brief Completion signatures of @c _sndr_type_.
 *
 * The @c stdexec::set_value_t() completion signature is added only if the child can complete successfully.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/commit/a0d95e90fc188f4f73328c4274551434edba3165
 *  - @cite P3557R3.
 */ // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_COMPL_SIGS_ADD(_sndr_type_, ...)                                                              \
    template <::stdexec::__decays_to<_sndr_type_> Self, typename... Env>                                               \
    static consteval auto get_completion_signatures() {                                                                \
        using child_completions_t =                                                                                    \
            ::stdexec::__completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>;                    \
        constexpr auto success_completion_count =                                                                      \
            ::stdexec::__msize_t<::stdexec::__detail::__count_of<::stdexec::set_value_t, child_completions_t>>::value; \
        if constexpr (success_completion_count > 0) {                                                                  \
            return experimental::execution::transform_completion_signatures(                                           \
                child_completions_t{},                                                                                 \
                experimental::execution::keep_completion<stdexec::set_value_t>(),                                      \
                experimental::execution::ignore_completion(),                                                          \
                experimental::execution::ignore_completion(),                                                          \
                stdexec::completion_signatures<__VA_ARGS__>());                                                        \
        } else {                                                                                                       \
            return child_completions_t{};                                                                              \
        }                                                                                                              \
    }

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_COMPL_SIGS_KEEP(_sndr_type_)                                                                  \
    template <::stdexec::__decays_to<_sndr_type_> Self, typename... Env>                                               \
    static consteval auto get_completion_signatures() {                                                                \
        return ::stdexec::__completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>{};               \
    }

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNATURES_HPP
