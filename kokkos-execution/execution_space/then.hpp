#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/diagnostics.hpp"
#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

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
    using policy_t = Kokkos::RangePolicy<Impl::exec_of_t<Sndr, Env>, Kokkos::LaunchBounds<1>>;

    template <typename Env, typename Functor, typename Sndr>
    using trnsfrmd_sndr_t =
        ParallelForSender<stdexec::then_t, Sndr, std::string_view, ThenWrapper<Functor>, policy_t<Sndr, Env>>;

    template <typename Env, typename Functor, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(const Env& env, stdexec::then_t, Functor&& functor, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Functor, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Functor, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        const auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return trnsfrmd_sndr_t<Env, Functor, Sndr>{
            {.label = Impl::dispatch_label<Impl::exec_of_t<Sndr, Env>, ": then">(),
             .functor = ThenWrapper<Functor>{std::forward<Functor>(functor)},
             .policy = policy_t<Sndr, Env>(schd.state->exec, 0, 1)},
            std::forward<Sndr>(sndr)
        };
    }

    template <typename Env, typename Functor, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::then_t, Functor&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<stdexec::then_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_THEN_HPP
