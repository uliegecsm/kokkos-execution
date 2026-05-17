#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct when_all_t { };

} // namespace Kokkos::Execution::ExecutionSpaceImpl

namespace stdexec {
template <>
struct __sexpr_impl<Kokkos::Execution::ExecutionSpaceImpl::when_all_t> : stdexec::__when_all::__when_all_impl {
    using base_t = stdexec::__when_all::__when_all_impl;

    template <typename State>
    using state_env_t = stdexec::env_of_t<decltype(std::declval<const State&>().__rcvr_)>;

    template <typename State>
    using state_exec_env_policy_t = Kokkos::Execution::ExecutionSpaceImpl::exec_env_policy_t<state_env_t<State>>;

    template <typename State>
    using base_env_t = decltype(base_t::__get_env(stdexec::__ignore{}, std::declval<const State&>()));

    template <typename ExecEnvPolicy, typename State>
    struct get_env_impl;

    template <typename State>
    struct get_env_impl<Kokkos::Execution::ExecutionSpaceImpl::WithExecEnvPolicy, State> {
        using execution_space = typename std::remove_cvref_t<
            stdexec::__query_result_t<state_env_t<State>, Kokkos::Execution::Impl::get_exec_t>
        >::execution_space;

        using type = decltype(Kokkos::Execution::ExecutionSpaceImpl::join_env_with_exec(
            std::declval<base_env_t<State>>(),
            std::declval<execution_space>()));

        [[nodiscard]]
        static constexpr auto get_env(const State& state) noexcept -> type {
            return Kokkos::Execution::ExecutionSpaceImpl::join_env_with_exec(
                base_t::__get_env(stdexec::__ignore{}, state),
                Kokkos::Execution::Impl::get_exec(stdexec::get_env(state.__rcvr_)).get());
        }
    };

    template <typename State>
    struct get_env_impl<Kokkos::Execution::ExecutionSpaceImpl::WithoutExecEnvPolicy, State> {
        using type = base_env_t<State>;

        [[nodiscard]]
        static constexpr auto get_env(const State& state) noexcept -> type {
            return base_t::__get_env(stdexec::__ignore{}, state);
        }
    };

    /**
     * Customize the environment of the receivers that @c stdexec::when_all connects the child senders with.
     *
     * See:
     * - https://github.com/NVIDIA/stdexec/blob/56613d3498bc39724dfbae0914cff2aaf3f9dcc6/include/stdexec/__detail/__basic_sender.hpp#L269-L274
     * - https://github.com/NVIDIA/stdexec/blob/56613d3498bc39724dfbae0914cff2aaf3f9dcc6/include/stdexec/__detail/__when_all.hpp#L454-L458
     */
    static constexpr auto __get_env = // NOLINT(bugprone-reserved-identifier)
        []<typename State>(stdexec::__ignore, const State& state) noexcept ->
        typename get_env_impl<state_exec_env_policy_t<State>, State>::type {
            return get_env_impl<state_exec_env_policy_t<State>, State>::get_env(state);
        };

    /**
     * This code reproduces the implementation of @c __get_completion_signatures of @c stdexec::when_all, except that it
     * does not statically assert that the tag is @c stdexec::when_all_t. This workaround is necessary because this sender
     * uses the tag @ref Kokkos::Execution::ExecutionSpaceImpl::when_all_t to distinguish from @c stdexec::when_all_t.
     *
     * See:
     * - https://github.com/NVIDIA/stdexec/blob/56613d3498bc39724dfbae0914cff2aaf3f9dcc6/include/stdexec/__detail/__when_all.hpp#L435
     */
    template <typename Self, typename... Envs>
    static consteval auto __get_completion_signatures() { // NOLINT(bugprone-reserved-identifier)
        if constexpr (stdexec::__minvocable_q<stdexec::__when_all::__when_all_impl::__completions_t, Self, Envs...>) {
            return stdexec::__when_all::__when_all_impl::__completions_t<Self, Envs...>{};
        } else if constexpr (sizeof...(Envs) == 0) {
            return stdexec::__throw_dependent_sender_error<Self>();
        }
    }
};

} // namespace stdexec

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <>
struct TransformSenderFor<stdexec::when_all_t> {
    template <typename Env, typename... Sndrs>
    auto operator()(const Env&, stdexec::when_all_t, stdexec::__, Sndrs&&... sndrs) const {
        return stdexec::__make_sexpr<Kokkos::Execution::ExecutionSpaceImpl::when_all_t>(
            stdexec::__{}, std::forward<Sndrs>(sndrs)...);
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP
