#ifndef KOKKOS_EXECUTION_IMPL_ENV_HPP
#define KOKKOS_EXECUTION_IMPL_ENV_HPP

#include "kokkos-execution/stdexec.hpp"

//! Retrieve the environment of @p _obj_ (with forwarding). // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_FORWARDING_GET_ENV(_type_, _obj_)                                                             \
    [[nodiscard]]                                                                                                      \
    constexpr auto get_env() const noexcept -> ::stdexec::__fwd_env_t<::stdexec::env_of_t<_type_>> {                   \
        return ::stdexec::__fwd_env(::stdexec::get_env(_obj_));                                                        \
    }

#endif // KOKKOS_EXECUTION_IMPL_ENV_HPP
