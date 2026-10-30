#ifndef KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP
#define KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP

#include <algorithm>
#include <array>
#if defined(__cpp_impl_reflection)
#    include <meta>
#endif
#include <ranges>
#include <string_view>

#include "stdexec/__detail/__config.hpp"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/fixed_string.hpp"

namespace Kokkos::Execution::Impl {

#if !defined(__cpp_impl_reflection)
namespace detail {

template <Kokkos::ExecutionSpace Exec, FixedString Suffix>
consteval auto dispatch_label() {
    constexpr std::string_view prefix = Kokkos::Impl::TypeInfo<Exec>::name();
    std::array<char, prefix.size() + Suffix.size()> buf{};
    std::ranges::copy(Suffix, std::ranges::copy(prefix, buf.begin()).out);
    return buf;
}

} // namespace detail

//! Get the dispatch label from @p Exec and @p Suffix.
template <Kokkos::ExecutionSpace Exec, FixedString Suffix>
constexpr auto dispatch_label_v = detail::dispatch_label<Exec, Suffix>();

//! View the dispatch label as a @c std::string_view.
template <Kokkos::ExecutionSpace Exec, FixedString Suffix>
consteval auto dispatch_label()
    STDEXEC_AUTO_RETURN(std::string_view{dispatch_label_v<Exec, Suffix>.data(), dispatch_label_v<Exec, Suffix>.size()})

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(_exec_, _suffix_)                                                     \
        Kokkos::Execution::Impl::dispatch_label<KOKKOS_IMPL_STRIP_PARENS(_exec_), _suffix_>()
#else
//! See @cite P3491R3.
template <Kokkos::ExecutionSpace Exec>
consteval std::string_view dispatch_label(std::string_view suffix) noexcept {
    return std::define_static_string(std::views::concat(std::meta::display_string_of(^^Exec), suffix));
}

#    define KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(_exec_, _suffix_)                                                     \
        Kokkos::Execution::Impl::dispatch_label<KOKKOS_IMPL_STRIP_PARENS(_exec_)>(_suffix_)
#endif
} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP
