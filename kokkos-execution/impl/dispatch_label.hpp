#ifndef KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP
#define KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP

#include <algorithm>
#include <array>
#include <string_view>

#include "Kokkos_Core.hpp"

#include "kokkos-execution/impl/fixed_string.hpp"

namespace Kokkos::Execution::Impl {

//! Get the dispatch label from @p Exec and @p Suffix.
template <Kokkos::ExecutionSpace Exec, FixedString Suffix>
static constexpr auto dispatch_label_v = [] {
    constexpr auto prefix = Kokkos::Impl::TypeInfo<Exec>::name();
    std::array<char, prefix.size() + Suffix.size() + 1> buf{};
    auto iter = buf.begin();
    for (const auto charac: prefix)
        *iter++ = charac;
    for (const auto charac: Suffix)
        *iter++ = charac;
    return buf;
}();

//! View the dispatch label as a @c std::string_view.
template <Kokkos::ExecutionSpace Exec, FixedString Suffix>
consteval std::string_view dispatch_label() noexcept {
    return {dispatch_label_v<Exec, Suffix>.data(), dispatch_label_v<Exec, Suffix>.size() - 1};
}

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP
