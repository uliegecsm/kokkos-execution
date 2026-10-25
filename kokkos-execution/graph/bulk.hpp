#ifndef KOKKOS_EXECUTION_GRAPH_BULK_HPP
#define KOKKOS_EXECUTION_GRAPH_BULK_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Graph.hpp"

#include "kokkos-execution/graph/node_props.hpp"
#include "kokkos-execution/graph/parallel_for.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/bulk.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::GraphImpl {

template <>
struct TransformSenderFor<stdexec::bulk_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = ParallelForSender<
        Sndr,
        std::string_view,
        Kokkos::Impl::DeviceHandle<Impl::exec_of_t<Sndr, Env>>,
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
        return no_graph_scheduler_in_env<stdexec::bulk_t, Sndr, Env>();
    }

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && graph_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        stdexec::bulk_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        STDEXEC_AUTO_RETURN(
            trnsfrmd_sndr_t<Env, Data, Sndr>{
                .clsr =
                    {.label = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL((Impl::exec_of_t<Sndr, Env>), ": bulk"),
                           .device_handle =
                         stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                             .state->get_device_handle(),
                           .functor = stdexec::__forward_like<Data>(data.__fun_),
                           .policy_or_iter_space = Impl::make_policy<Impl::exec_of_t<Sndr, Env>>(
                         Kokkos::dims<1, typename Impl::bulk_traits<Data>::shape_t>{
                             stdexec::__forward_like<Data>(data.__shape_)},
                           Kokkos::IndexType<typename Impl::bulk_traits<Data>::shape_t>{})},
                .sndr = std::forward<Sndr>(sndr)
    })
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_BULK_HPP
