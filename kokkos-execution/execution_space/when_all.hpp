#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_WHEN_ALL_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/execution_space/get_exec.hpp"
#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct when_all_t { };

} // namespace Kokkos::Execution::ExecutionSpaceImpl

namespace stdexec {
template <>
struct __sexpr_impl<Kokkos::Execution::ExecutionSpaceImpl::when_all_t> : stdexec::__when_all::__when_all_impl {
    /**
     * Customize the environment of the receivers that @c stdexec::when_all connects the child senders with.
     *
     * See:
     * - https://github.com/NVIDIA/stdexec/blob/56613d3498bc39724dfbae0914cff2aaf3f9dcc6/include/stdexec/__detail/__basic_sender.hpp#L269-L274
     * - https://github.com/NVIDIA/stdexec/blob/56613d3498bc39724dfbae0914cff2aaf3f9dcc6/include/stdexec/__detail/__when_all.hpp#L454-L458
     */
    static constexpr auto __get_env = // NOLINT(bugprone-reserved-identifier)
        []<typename State>(stdexec::__ignore, const State& state) noexcept {
            auto env = stdexec::__when_all::__when_all_impl::__get_env(stdexec::__ignore{}, state);

            using exec_env_policy_t =
                Kokkos::Execution::ExecutionSpaceImpl::exec_env_policy_t<stdexec::env_of_t<decltype(state.__rcvr_)>>;

            if constexpr (std::same_as<exec_env_policy_t, Kokkos::Execution::ExecutionSpaceImpl::WithExecEnvPolicy>) {
                return Kokkos::Execution::ExecutionSpaceImpl::join_env_with_exec(
                    std::move(env),
                    Kokkos::Execution::ExecutionSpaceImpl::get_exec(stdexec::get_env(state.__rcvr_)).get());
            } else {
                return std::move(env);
            }
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
            return stdexec::__dependent_sender<Self>();
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
