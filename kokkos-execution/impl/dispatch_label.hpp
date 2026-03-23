#ifndef KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP
#define KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL_HPP

#include <algorithm>
#include <array>
#include <string_view>

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::Impl {

template <size_t N>
struct FixedString {
    char data[N]{};

    consteval FixedString(const char (&str)[N]) noexcept { // NOLINT(google-explicit-constructor)
        std::copy_n(str, N, data);
    }

    consteval auto size() const {
        return N - 1;
    }

    consteval auto begin() const {
        return data;
    }
    consteval auto end() const {
        return data + N - 1;
    }
};

//! Get the dispatch label from @p Exec and @p Suffix.
template <Kokkos::ExecutionSpace Exec, FixedString Suffix>
static constexpr auto dispatch_label_v = [] {
    constexpr auto prefix = Kokkos::Impl::TypeInfo<Exec>::name();
    std::array<char, prefix.size() + Suffix.size() + 1> buf{};
    auto iter = buf.begin();
    for (auto charac: prefix)
        *iter++ = charac;
    for (auto charac: Suffix)
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
