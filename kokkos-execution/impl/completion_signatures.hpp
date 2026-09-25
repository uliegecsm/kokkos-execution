#ifndef KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNATURES_HPP
#define KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNATURES_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Concatenate @p Sndr completion signatures with @p ExtraSigs
 *        if @p Sndr sends on the value channel; otherwise return @p Sndr
 *        completion signatures unchanged.
 *
 * References:
 *  - https://github.com/NVIDIA/stdexec/commit/a0d95e90fc188f4f73328c4274551434edba3165
 *  - @cite P3557R3.
 */
template <stdexec::sender Sndr, stdexec::__valid_completion_signatures ExtraSigs, typename... Env>
consteval auto completion_signatures_add() {
    constexpr auto completions = stdexec::get_completion_signatures<Sndr, Env...>();

    if constexpr (stdexec::__sends<stdexec::set_value_t, Sndr, Env...>) {
        return stdexec::__concat_completion_signatures(completions, ExtraSigs{});
    } else {
        return completions;
    }
}

template <stdexec::sender Sndr, stdexec::__valid_completion_signatures ExtraSigs, typename... Env>
using completion_signatures_add_t = decltype(completion_signatures_add<Sndr, ExtraSigs, Env...>());

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_COMPL_SIGS_ADD(_decayed_self_type_, _sndr_type_, ...)                                         \
    template <stdexec::__decays_to<_decayed_self_type_> Self, typename... Env>                                         \
    static consteval auto get_completion_signatures() {                                                                \
        return Kokkos::Execution::Impl::completion_signatures_add<                                                     \
            stdexec::__copy_cvref_t<Self, _sndr_type_>,                                                                \
            stdexec::completion_signatures<__VA_ARGS__>,                                                               \
            Env...                                                                                                     \
        >();                                                                                                           \
    }

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_COMPL_SIGS_KEEP(_decayed_self_type_, _sndr_type_)                                             \
    template <stdexec::__decays_to<_decayed_self_type_> Self, typename... Env>                                         \
    static consteval auto get_completion_signatures() {                                                                \
        return stdexec::get_completion_signatures<stdexec::__copy_cvref_t<Self, _sndr_type_>, Env...>();               \
    }

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNATURES_HPP
