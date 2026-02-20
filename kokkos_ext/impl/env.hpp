#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
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

namespace experimental::execution {
namespace impl {
struct upsert_in_env_fn {

    template <typename>
    struct EnvOneOf;

    template <typename Env1, typename Env2>
    struct EnvOneOf<stdexec::env<Env1, Env2>> {
        using type = Env1;
    };

    template <typename Env>
    using env1_of_t = typename EnvOneOf<std::remove_cvref_t<Env>>::type;

    //! Handle the case of a @c stdexec::prop without wrapping it into an @c stdexec::env.
    template <typename Tag, typename PropValue, typename Value>
    constexpr auto operator()(Tag tag, stdexec::prop<Tag, PropValue>, Value&& value) const {
        return stdexec::prop{tag, std::forward<Value>(value)};
    }

    //! The environment does not contain the required property.
    template <typename Tag, typename Env, typename Value>
    requires(!stdexec::__queryable_with<std::remove_cvref_t<Env>, Tag>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)},
            std::forward<Env>(env)
        };
    }

    //! The environment contains the required property, possibly with a value mismatch.
    template <typename Tag, typename PropValue, typename Value>
    constexpr auto operator()(Tag tag, stdexec::env<stdexec::prop<Tag, PropValue>>, Value&& value) const {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)}
        };
    }

    //! Multiple properties are carried, the first one matches.
    template <typename Tag, typename Env, typename Value>
    requires(stdexec::__queryable_with<env1_of_t<std::remove_cvref_t<Env>>, Tag>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)},
            std::forward<Env>(env).__env2_
        };
    }

    //! Multiple properties are carried, but the first one doesn't match.
    template <typename Tag, typename Env, typename Value>
    requires(
        stdexec::__queryable_with<std::remove_cvref_t<Env>, Tag>
        && !stdexec::__queryable_with<env1_of_t<std::remove_cvref_t<Env>>, Tag>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const {
        return stdexec::env{
            std::forward<Env>(env).__env1_,
            this->operator()(
                tag, std::forward<Env>(env).__env2_, std::forward<Value>(value))}; // NOLINT(bugprone-use-after-move)
    }

    //! Unwrap forwarding environment and delegate to the appropriate overload.
    template <typename Tag, typename Env, typename Value>
    requires(stdexec::__queryable_with<Env &&, Tag> && stdexec::__env::__is_fwd_env<Env>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const {
        return stdexec::__env::__fwd{this->operator()(tag, std::forward<Env>(env).__env_, std::forward<Value>(value))};
    }
};
} // namespace impl

template <typename Tag, typename Env, typename Value>
using upsert_in_env_t = std::invoke_result_t<impl::upsert_in_env_fn, Tag, Env, Value>;

inline constexpr impl::upsert_in_env_fn upsert_in_env{};

} // namespace experimental::execution

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_ENV_HPP
