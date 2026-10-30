#ifndef KOKKOS_EXECUTION_IMPL_MACROS_HPP
#define KOKKOS_EXECUTION_IMPL_MACROS_HPP

#include "Kokkos_Macros.hpp"

/**
 * @brief Similar to @c STDEXEC_AUTO_RETURN except that the return type is specified explicitly,
 *        rather than deduced from the expression.
 */ // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_AUTO_RETURN_AS(_type_, ...)                                                                   \
    noexcept(noexcept(__VA_ARGS__))->KOKKOS_IMPL_STRIP_PARENS(_type_) {                                                \
        return __VA_ARGS__;                                                                                            \
    }

#endif // KOKKOS_EXECUTION_IMPL_MACROS_HPP
