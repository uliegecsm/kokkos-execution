#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct when_all_t { };

/**
 * Implementation of our @ref Kokkos::Execution::ExecutionSpaceImpl::when_all_t, using the
 * @c stdexec::__facade algorithm of @c stdexec::when_all_t, see
 * https://github.com/NVIDIA/stdexec/blob/4781797199f20ef742b39ea00785a1503badd218/include/stdexec/__detail/__when_all.hpp#L670.
 *
 * It is the same algorithm as @c stdexec::when_all, except that, if the environment of the receiver can be queried with
 * @ref Impl::get_exec_t, the environment of the children is extended with it.
 */
struct when_all_algo : stdexec::__when_all::__algo {
    using base_t = stdexec::__when_all::__algo;

    template <typename Key, typename Sender, typename Env>
    [[nodiscard]]
    static constexpr auto
        __env_for(const Key& key, const Sender& sndr, Env&& env) noexcept // NOLINT(bugprone-reserved-identifier)
    {
        return extend(env, [&] { return base_t::__env_for(key, sndr, std::forward<Env>(env)); });
    }

    template <typename Key, typename State>
    [[nodiscard]]
    static constexpr auto __get_env(const Key& key, const State& state) noexcept // NOLINT(bugprone-reserved-identifier)
    {
        return extend(stdexec::get_env(state.__rcvr_), [&] { return base_t::__get_env(key, state); });
    }

   private:
    //! Extend the environment returned by @p make_env with the execution space of @p env, if it has one.
    template <typename Env, typename MakeEnv>
    static constexpr auto extend(const Env& env, MakeEnv&& make_env) noexcept {
        if constexpr (std::same_as<extend_env_with_exec_policy_t<Env>, WithExecEnvPolicy>) {
            const auto& exec = Impl::get_exec(env).get();
            return join_env_with_exec(std::forward<MakeEnv>(make_env)(), exec);
        } else {
            return std::forward<MakeEnv>(make_env)();
        }
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

namespace stdexec::__facade {

template <>
struct __algorithm<Kokkos::Execution::ExecutionSpaceImpl::when_all_t>
    : Kokkos::Execution::ExecutionSpaceImpl::when_all_algo { };

} // namespace stdexec::__facade

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <>
struct TransformSenderFor<stdexec::when_all_t> {
    template <typename Env, typename Data, typename... Sndrs>
    auto operator()(const Env&, stdexec::when_all_t, Data&& data, Sndrs&&... sndrs) const {
        return stdexec::__facade::__sender{
            Kokkos::Execution::ExecutionSpaceImpl::when_all_t{},
            std::forward<Data>(data),
            std::forward<Sndrs>(sndrs)...};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP
