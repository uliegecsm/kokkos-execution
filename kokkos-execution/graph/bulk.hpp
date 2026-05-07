#ifndef KOKKOS_EXECUTION_GRAPH_BULK_HPP
#define KOKKOS_EXECUTION_GRAPH_BULK_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "Kokkos_Graph.hpp"

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
        std::string,
        typename Kokkos::Execution::Impl::bulk_traits<Data>::functor_t,
        Kokkos::RangePolicy<Impl::exec_of_t<Sndr, Env>>
    >;

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(
        const Env& env,
        stdexec::bulk_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t,
                 Sndr&&
        >) {
        if constexpr (graph_completing_sender<Sndr, Env>) {
            auto& [parallel_policy, shape, functor] = data;

            auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

            return trnsfrmd_sndr_t<Env, Data, Sndr>{
                .clsr =
                    {.node_props = Kokkos::Experimental::node_props(
                         std::string(Impl::dispatch_label<Impl::exec_of_t<Sndr, Env>, ": bulk">()),
                           Kokkos::Experimental::get_device_handle(schd.state->exec)),
                           .functor = stdexec::__forward_like<Data>(functor),
                           .policy = Kokkos::RangePolicy<Impl::exec_of_t<Sndr, Env>>(0, shape)},
                .sndr = std::forward<Sndr>(sndr)
            };
        } else {
            return no_graph_scheduler_in_env<stdexec::bulk_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_BULK_HPP
