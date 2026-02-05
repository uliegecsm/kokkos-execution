#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_COMPLETION_SIGNATURES_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_COMPLETION_SIGNATURES_HPP

#include "stdexec/execution.hpp"

namespace Kokkos::Experimental::details::impl {

/**
 * @brief Completion signatures of @c _sndr_type_.
 *
 * The @c stdexec::set_value_t() completion signature is always added.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/commit/a0d95e90fc188f4f73328c4274551434edba3165
 *  - @cite P3557R3.
 */ // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_ADD(_sndr_type_, ...)                                                  \
    template <::stdexec::__decays_to<_sndr_type_> Self, typename... Env>                                               \
    static consteval auto get_completion_signatures() {                                                                \
        using child_completions_t =                                                                                    \
            ::stdexec::__completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>;                    \
        return ::stdexec::transform_completion_signatures<                                                             \
            child_completions_t,                                                                                       \
            ::stdexec::completion_signatures<::stdexec::set_value_t() __VA_OPT__(, ) __VA_ARGS__>                      \
        >{};                                                                                                           \
    }

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_DISPATCHING_KOKKOS_EXT_COMPL_SIGS_KEEP(_sndr_type_)                                                      \
    template <::stdexec::__decays_to<_sndr_type_> Self, typename... Env>                                               \
    static consteval auto get_completion_signatures() {                                                                \
        return ::stdexec::__completion_signatures_of_t<::stdexec::__copy_cvref_t<Self, Sndr>, Env...>{};               \
    }

} // namespace Kokkos::Experimental::details::impl

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_COMPLETION_SIGNATURES_HPP
