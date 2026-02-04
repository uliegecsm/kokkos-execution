#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP

#include "stdexec/execution.hpp"

//! Retrieve the environment of @p _obj_ (with forwarding). // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GRAPH_DISPATCHING_KOKKOS_EXT_FORWARDING_GET_ENV(_type_, _obj_)                                                 \
    [[nodiscard]]                                                                                                      \
    constexpr auto get_env() const noexcept -> ::stdexec::__fwd_env_t<::stdexec::env_of_t<_type_>> {                   \
        return ::stdexec::__fwd_env(::stdexec::get_env(_obj_));                                                        \
    }

namespace Kokkos::Experimental::details::impl {
//! An environment whose sole query is @c stdexec::get_domain_t.
template <typename Domain>
struct domain_queryable_env_t {
    static constexpr auto query(stdexec::get_domain_t) noexcept -> Domain {
        return {};
    }
};
} // namespace Kokkos::Experimental::details::impl

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP
