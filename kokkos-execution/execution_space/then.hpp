#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/parallel_for.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Inspired by https://github.com/kokkos/kokkos/blob/69273c3a4e7b6adeb95066341ca201d62fe1e698/core/src/impl/Kokkos_GraphNodeThenImpl.hpp#L28.
template <typename Functor>
requires(std::same_as<void, std::invoke_result_t<Functor>>)
struct ThenWrapper {
    Functor functor; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    template <std::integral T>
    KOKKOS_FUNCTION void operator()(const T) const {
        functor();
    }
};

template <>
struct TransformSenderFor<stdexec::then_t> {
    template <typename Sndr, typename Env>
    using schd_t = stdexec::__completion_scheduler_of_t<stdexec::set_value_t, Sndr, const Env&>;

    template <typename Sndr, typename Env>
    using execution_space = typename schd_t<Sndr, Env>::execution_space;

    template <typename Sndr, typename Env>
    using policy_t = Kokkos::RangePolicy<execution_space<Sndr, Env>, Kokkos::LaunchBounds<1>>;

    template <typename Sndr, typename Functor, typename Env>
    using sndr_t = ParallelForSender<Sndr, ThenWrapper<Functor>, policy_t<Sndr, Env>>;

    template <typename Env, typename Functor, execution_space_completing_sender<Env> Sndr>
    requires requires { typename schd_t<Sndr, Env>; }
    auto operator()(const Env& env, stdexec::then_t, Functor&& functor, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 sndr_t<Sndr, Functor, Env>,
                 parallel_for_t,
                 typename sndr_t<Sndr, Functor, Env>::closure_t&&,
                 Sndr&&
        >) {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return sndr_t<Sndr, Functor, Env>{
            {},
            {{std::string(std::format("{}: then", Kokkos::Impl::TypeInfo<execution_space<Sndr, Env>>::name())),
              ThenWrapper<Functor>{std::forward<Functor>(functor)},
              policy_t<Sndr, Env>(schd.state->exec, 0, 1)}},
            std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
