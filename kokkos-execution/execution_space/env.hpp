#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_ENV_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_ENV_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/impl/get_exec.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct WithExecEnvPolicy { };
struct WithoutExecEnvPolicy { };

//! Unconditionally join @p exec to @p env.
template <typename Env, Kokkos::ExecutionSpace Exec>
constexpr auto join_env_with_exec(Env&& env, const Exec& exec) noexcept {
    return stdexec::__env::__join(
        stdexec::prop{Impl::get_exec, Impl::ExecutionSpaceRef{exec}}, stdexec::__fwd_env(std::forward<Env>(env)));
}

//! Join @p exec to @p env if the policy is @ref WithExecEnvPolicy.
template <typename ExecEnvPolicy, typename Env, Kokkos::ExecutionSpace Exec>
constexpr auto join_env_with_exec(Env&& env, const Exec& exec) noexcept {
    if constexpr (std::same_as<ExecEnvPolicy, WithExecEnvPolicy>) {
        return join_env_with_exec(std::forward<Env>(env), exec);
    } else {
        return stdexec::__fwd_env(std::forward<Env>(env));
    }
}

template <typename ExecEnvPolicy, typename Env, Kokkos::ExecutionSpace Exec>
using join_env_with_exec_t = decltype(join_env_with_exec<ExecEnvPolicy>(std::declval<Env>(), std::declval<Exec>()));

/**
 * If the policy is @ref WithExecEnvPolicy, extract the @ref Impl::ExecutionSpaceRef and re-inject it into the
 * environment to extend its availability.
 *
 * @note This is not the same intent as using a forwarding query.
 */
template <typename ExecEnvPolicy, typename Env>
constexpr auto extend_env(Env&& env) noexcept {
    if constexpr (std::same_as<ExecEnvPolicy, WithExecEnvPolicy>) {
        auto ref = Impl::get_exec(env);
        return stdexec::__env::__join(
            stdexec::prop{Impl::get_exec, std::move(ref)}, stdexec::__fwd_env(std::forward<Env>(env)));
    } else {
        return stdexec::__fwd_env(std::forward<Env>(env));
    }
}

template <typename ExecEnvPolicy, typename Env>
using extend_env_t = decltype(extend_env<ExecEnvPolicy>(std::declval<Env>()));

//! If @p Env is queryable with @ref Impl::get_exec_t, use @ref WithExecEnvPolicy.
template <typename Env>
using extend_exec_env_policy_t =
    std::conditional_t<stdexec::__queryable_with<Env, Impl::get_exec_t>, WithExecEnvPolicy, WithoutExecEnvPolicy>;

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_ENV_HPP
