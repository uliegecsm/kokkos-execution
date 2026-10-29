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
        Impl::make_policy_t<
            Impl::exec_of_t<Sndr, Env>,
            Kokkos::dims<1, typename Impl::bulk_traits<Data>::shape_t>,
            Kokkos::IndexType<typename Impl::bulk_traits<Data>::shape_t>
        >
    >;

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::bulk_t, Data&&, Sndr&&) const noexcept {
        return no_execution_space_scheduler_in_env<stdexec::bulk_t, Sndr, Env>();
    }

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && execution_space_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        stdexec::bulk_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        STDEXEC_AUTO_RETURN(
            trnsfrmd_sndr_t<Env, Data, Sndr>{
                {.label = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL((Impl::exec_of_t<Sndr, Env>), ": bulk"),
                 .functor = stdexec::__forward_like<Data>(data.__fun_),
                 .policy = Impl::make_policy(
                     stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env).state->exec,
                 Kokkos::dims<1, typename Impl::bulk_traits<Data>::shape_t>{
                         stdexec::__forward_like<Data>(data.__shape_)},
                 Kokkos::IndexType<typename Impl::bulk_traits<Data>::shape_t>{})},
                std::forward<Sndr>(sndr)
    })
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_BULK_HPP
