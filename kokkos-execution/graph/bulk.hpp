#ifndef KOKKOS_EXECUTION_GRAPH_BULK_HPP
#define KOKKOS_EXECUTION_GRAPH_BULK_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.graph;
#else
#    include "Kokkos_Core.hpp"
#    include "Kokkos_Graph.hpp"
#endif

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
    struct Transformer {
        using execution_space = Impl::exec_of_t<Sndr, Env>;

        using functor_t = Impl::functor_for_bulk_t<Data>;
        using iter_space_t = Impl::iter_space_for_bulk_t<Data>;
        using make_policy_t = Impl::make_policy_for_bulk_t<execution_space, Data>;

        using policy_t = Impl::make_policy_result_t<make_policy_t, iter_space_t>;

        using data_t = Impl::ParallelForData<functor_t, policy_t, Impl::Label<std::string_view>>;

        using sndr_t = ParallelForSender<Sndr, execution_space, data_t>;

        auto operator()(const Env& env, std::remove_reference_t<Data>& data, std::remove_reference_t<Sndr>& sndr) const
            STDEXEC_AUTO_RETURN(
                sndr_t{
                    {.state = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                                  .state,
                     .data =
                         {{Impl::Label{KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(execution_space, ": bulk")}},
                          stdexec::__forward_like<Data>(data.__fun_),
                          make_policy_t{}(iter_space_t(stdexec::__forward_like<Data>(data.__shape_)))}},
                    std::forward<Sndr>(sndr)
        })
    };

    template <typename Env, Impl::has_parallel_policy Data, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && graph_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        stdexec::bulk_t,
        Data&& data, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr  // NOLINT(cppcoreguidelines-missing-std-forward)
    ) const STDEXEC_AUTO_RETURN(Transformer<Env, Data, Sndr>{}(env, data, sndr))

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::bulk_t, Data&&, Sndr&&) const noexcept {
        return no_graph_scheduler_in_env<stdexec::bulk_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_BULK_HPP
