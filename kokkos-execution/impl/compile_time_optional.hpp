#ifndef KOKKOS_EXECUTION_IMPL_COMPILE_TIME_OPTIONAL_HPP
#define KOKKOS_EXECUTION_IMPL_COMPILE_TIME_OPTIONAL_HPP

#include <concepts>
#include <type_traits>
#include <utility>

namespace Kokkos::Execution::Impl {

/**
 * @brief Compile-time optional value that holds a @c T when @p Enabled, is an empty type otherwise.
 *
 * Unlike @c std::optional, there is no run-time engaged state and no storage for one.
 * When disabled, constructor arguments are accepted and discarded.
 */
template <bool Enabled, typename T>
struct CompileTimeOptional;

template <typename T>
struct CompileTimeOptional<false, T> {
    using value_type = T;

    constexpr CompileTimeOptional() noexcept = default;

    //! Silently ignore everything, except copies and moves of this very type.
    template <typename... Args>
    requires(!(sizeof...(Args) == 1 && (std::same_as<std::remove_cvref_t<Args>, CompileTimeOptional> && ...)))
    constexpr explicit CompileTimeOptional(Args&&...) noexcept {
    }
};

template <typename T>
struct CompileTimeOptional<true, T> {
    using value_type = T;

    template <typename... Args>
    requires(
        std::constructible_from<T, Args...>
        && !(sizeof...(Args) == 1 && (std::same_as<std::remove_cvref_t<Args>, CompileTimeOptional> && ...)))
    constexpr explicit CompileTimeOptional(Args&&... args) noexcept(std::is_nothrow_constructible_v<T, Args&&...>)
        : value(std::forward<Args>(args)...) {
    }

    T value;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_COMPILE_TIME_OPTIONAL_HPP
