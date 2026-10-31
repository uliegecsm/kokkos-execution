#ifndef KOKKOS_EXECUTION_IMPL_MACROS_HPP
#define KOKKOS_EXECUTION_IMPL_MACROS_HPP

#include "Kokkos_Macros.hpp"

/**
 * @brief Define a function by returning a list-initialized object of an explicit type.
 *
 * Expands to a @c noexcept-specifier, a trailing return type and the function body, so that
 * @code
 * auto make(A a, B b) KOKKOS_EXECUTION_RETURNS_BRACED((Pair<A, B>), a, b)
 * @endcode
 * is equivalent to
 * @code
 * auto make(A a, B b) noexcept(noexcept(Pair<A, B>{a, b})) -> Pair<A, B> {
 *     return Pair<A, B>{a, b};
 * }
 * @endcode
 *
 * @note Prefer @c STDEXEC_AUTO_RETURN whenever the return type can be deduced from the expression.
 *       Use this macro only when it cannot, that is, when the return type must be stated explicitly.
 */ // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_RETURNS_BRACED(_type_, ...)                                                                   \
    noexcept(noexcept(KOKKOS_IMPL_STRIP_PARENS(_type_){__VA_ARGS__}))->KOKKOS_IMPL_STRIP_PARENS(_type_) {              \
        return KOKKOS_IMPL_STRIP_PARENS(_type_){__VA_ARGS__};                                                          \
    }

#endif // KOKKOS_EXECUTION_IMPL_MACROS_HPP
