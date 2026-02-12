#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

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

namespace exec {
namespace impl {
struct upsert_in_env_fn {
    //! The environment is empty. Fill it.
    template <typename Tag, typename Value>
    constexpr auto operator()(Tag tag, stdexec::env<>, Value&& value) const noexcept {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)}
        };
    }

    //! The environment contains the required property, possibly with a value mismatch.
    template <typename Tag, typename PropValue, typename Value>
    constexpr auto operator()(Tag tag, stdexec::env<stdexec::prop<Tag, PropValue>>, Value&& value) const noexcept {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)}
        };
    }

    //! The environment does not contain the required property.
    template <typename Tag, typename OtherTag, typename OtherPropValue, typename Value>
    requires(!std::same_as<Tag, OtherTag>)
    constexpr auto operator()(Tag tag, const stdexec::env<stdexec::prop<OtherTag, OtherPropValue>>& env, Value&& value)
        const noexcept {
        return stdexec::env{
            stdexec::prop{OtherTag{},      env.query(OtherTag{})},
            stdexec::prop{       tag, std::forward<Value>(value)}
        };
    }

    //! Multiple properties are carried, the first one matches.
    template <typename Tag, typename PropValue, typename Env2, typename Value>
    constexpr auto
        operator()(Tag tag, stdexec::env<stdexec::prop<Tag, PropValue>, Env2> env, Value&& value) const noexcept {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)},
            env.env2_
        };
    }

    //! Multiple properties are carried, but the first one doesn't match.
    template <typename Tag, typename PropTag, typename PropValue, typename Env2, typename Value>
    requires(!std::same_as<Tag, PropTag>)
    constexpr auto
        operator()(Tag tag, stdexec::env<stdexec::prop<PropTag, PropValue>, Env2> env, Value&& value) const noexcept {
        return stdexec::env{
            stdexec::prop{PropTag{}, env.query(PropTag{})},
            this->operator()(tag, stdexec::env<Env2>{env.env2_},
            std::forward<Value>(value))
        };
    }

    //! Unwrap forwarding environment and delegate to the appropriate overload.
    template <typename Tag, typename Env, typename Value>
    constexpr auto operator()(Tag tag, stdexec::__env::__fwd<Env> env, Value&& value) const noexcept {
        return stdexec::__env::__fwd{this->operator()(tag, env.__env_, std::forward<Value>(value))};
    }
};
} // namespace impl

template <typename Tag, typename Env, typename Value>
using upsert_in_env_t = std::invoke_result_t<impl::upsert_in_env_fn, Tag, Env, Value>;

inline constexpr impl::upsert_in_env_fn upsert_in_env{};

} // namespace exec

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP
