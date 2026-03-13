#ifndef KOKKOS_EXECUTION_IMPL_ENV_HPP
#define KOKKOS_EXECUTION_IMPL_ENV_HPP

#include "kokkos-execution/stdexec.hpp"

//! Retrieve the environment of @p _obj_ (with forwarding). // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define KOKKOS_EXECUTION_FORWARDING_GET_ENV(_type_, _obj_)                                                             \
    [[nodiscard]]                                                                                                      \
    constexpr auto get_env() const noexcept -> ::stdexec::__fwd_env_t<::stdexec::env_of_t<_type_>> {                   \
        return ::stdexec::__fwd_env(::stdexec::get_env(_obj_));                                                        \
    }

namespace Kokkos::Execution::Impl {

struct UpsertInEnvFn {

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
    constexpr auto operator()(Tag tag, stdexec::prop<Tag, PropValue>, Value&& value) const noexcept {
        return stdexec::prop{tag, std::forward<Value>(value)};
    }

    //! The environment does not contain the required property.
    template <typename Tag, typename Env, typename Value>
    requires(!stdexec::__queryable_with<std::remove_cvref_t<Env>, Tag>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const noexcept {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)},
            std::forward<Env>(env)
        };
    }

    //! The environment contains the required property, possibly with a value mismatch.
    template <typename Tag, typename PropValue, typename Value>
    constexpr auto operator()(Tag tag, stdexec::env<stdexec::prop<Tag, PropValue>>, Value&& value) const noexcept {
        return stdexec::env{
            stdexec::prop{tag, std::forward<Value>(value)}
        };
    }

    //! Multiple properties are carried, the first one matches.
    template <typename Tag, typename Env, typename Value>
    requires(stdexec::__queryable_with<env1_of_t<std::remove_cvref_t<Env>>, Tag>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const noexcept {
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
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const noexcept {
        return stdexec::env{
            std::forward<Env>(env).__env1_,
            this->operator()(
                tag, std::forward<Env>(env).__env2_, std::forward<Value>(value))}; // NOLINT(bugprone-use-after-move)
    }

    //! Unwrap forwarding environment and delegate to the appropriate overload.
    template <typename Tag, typename Env, typename Value>
    requires(
        stdexec::__queryable_with<Env &&, Tag> && stdexec::__env::__is_fwd_env<Env>
        && requires(
            UpsertInEnvFn const & self,
            Tag tag,
            Env&& env,
            Value&& value) { self(tag, std::forward<Env>(env).__env_, std::forward<Value>(value)); })
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const noexcept {
        return stdexec::__env::__fwd{this->operator()(tag, std::forward<Env>(env).__env_, std::forward<Value>(value))};
    }
};

struct UpsertInEnvOrJoinFn {
    template <typename Tag, typename Env, typename Value>
    requires(std::is_invocable_v<UpsertInEnvFn, Tag, Env &&, Value &&>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const noexcept {
        return UpsertInEnvFn{}.operator()(tag, std::forward<Env>(env), std::forward<Value>(value));
    }

    template <typename Tag, typename Env, typename Value>
    requires(!std::is_invocable_v<UpsertInEnvFn, Tag, Env &&, Value &&>)
    constexpr auto operator()(Tag tag, Env&& env, Value&& value) const noexcept {
        return stdexec::__env::__join(stdexec::prop{tag, std::forward<Value>(value)}, std::forward<Env>(env));
    }
};

template <typename Tag, typename Env, typename Value>
using upsert_in_env_t = std::invoke_result_t<UpsertInEnvFn, Tag, Env, Value>;

inline constexpr UpsertInEnvFn upsert_in_env{};

template <typename Tag, typename Env, typename Value>
using upsert_in_env_or_join_t = std::invoke_result_t<UpsertInEnvOrJoinFn, Tag, Env, Value>;

inline constexpr UpsertInEnvOrJoinFn upsert_in_env_or_join{};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_ENV_HPP
