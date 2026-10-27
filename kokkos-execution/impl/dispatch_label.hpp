#ifndef KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP
#define KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP

#include <algorithm>
#include <array>
#include <string_view>

#include "stdexec/__detail/__config.hpp"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/fixed_string.hpp"

namespace Kokkos::Execution::Impl {

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

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP
