#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/diagnostics.hpp"
#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/bulk.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

template <>
struct TransformSenderFor<stdexec::bulk_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelForSender<
        stdexec::bulk_t,
        Sndr,
        std::string_view,
        typename Kokkos::Execution::Impl::bulk_traits<Data>::functor_t,
        Impl::policy_for_bulk_t<Env, Data, Sndr>
    >;

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(const Env& env, stdexec::bulk_t, Data&& data, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t&&,
                 Sndr&&
        >) {
        auto&& [parallel_policy, shape, functor] = std::forward<Data>(data);

        const auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return trnsfrmd_sndr_t<Env, Data, Sndr>{
            {.label = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL((Impl::exec_of_t<Sndr, Env>), ": bulk"),
             .functor = stdexec::__forward_like<Data>(functor),
             .policy =
                 Impl::policy_for_bulk_t<Env, Data, Sndr>(schd.state->exec, 0, stdexec::__forward_like<Data>(shape))},
            std::forward<Sndr>(sndr)
        };
    }

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::bulk_t, Data&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<stdexec::bulk_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
