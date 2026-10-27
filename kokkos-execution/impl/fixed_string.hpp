#ifndef KOKKOS_EXECUTION_IMPL_FIXED_STRING_HPP
#define KOKKOS_EXECUTION_IMPL_FIXED_STRING_HPP

#include <algorithm>
#include <cstddef>
#include <string_view>

#include "stdexec/__detail/__config.hpp"

namespace Kokkos::Execution::Impl {

/**
 * @brief Fixed-capacity string built from a @c char array.
 *
 * @note Nothing is done to verify if the @c null terminator is present.
 *
 * @note It is a structural type, so it can be used as a non-type template parameter.
 */
template <std::size_t N>
struct FixedString {
    char data[N]{};

    constexpr FixedString(const char (&str)[N]) noexcept { // NOLINT(google-explicit-constructor)
        std::ranges::copy(str, data);
    }

    constexpr auto size() const noexcept {
        return N - 1;
    }

    constexpr auto begin() const noexcept {
        return data;
    }

    constexpr auto end() const noexcept {
        return data + N - 1;
    }

    constexpr operator std::string_view() const noexcept(noexcept(view())) { // NOLINT(google-explicit-constructor)
        return view();
    }

    [[nodiscard]]
    constexpr auto view() const STDEXEC_AUTO_RETURN(std::string_view{data, size()})
};

template <typename T>
constexpr bool is_fixed_string_v = false;

template <std::size_t N>
constexpr bool is_fixed_string_v<FixedString<N>> = true;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_FIXED_STRING_HPP
