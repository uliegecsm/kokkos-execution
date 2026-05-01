#ifndef KOKKOS_EXECUTION_GRAPH_BULK_HPP
#define KOKKOS_EXECUTION_GRAPH_BULK_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "Kokkos_Graph.hpp"

#include "kokkos-execution/graph/events.hpp"
#include "kokkos-execution/graph/operation_state.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/bulk.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::GraphImpl {

template <Kokkos::ExecutionSpace Exec, typename Data>
struct BulkClosure {
    using execution_space = Exec;
    using device_handle_t = Kokkos::Impl::DeviceHandle<Exec>;
    using policy_t = Kokkos::RangePolicy<execution_space>;
    using functor_t = typename Impl::bulk_traits<Data>::functor_t;
    using node_props_t = typename Kokkos::Impl::NodeCtorProps<std::string, device_handle_t>::uniform_type;

    template <NodeRef Predecessor>
    using node_t = decltype(std::declval<const Predecessor&>().then_parallel_for(
        std::declval<node_props_t&&>(),
        std::declval<policy_t>(),
        std::declval<functor_t&&>()));

    //! @warning Unconditionally **not** @c noexcept because adding a node may throw.
    template <bool PredecessorIsRoot, NodeRef Predecessor>
    auto add_node(const Predecessor& predecessor) && noexcept(false) -> node_t<Predecessor> {
        auto& [parallel_policy, shape, functor] = data;
        auto node = predecessor.then_parallel_for(
            std::move(node_props),
            policy_t(0, stdexec::__forward_like<Data>(shape)),
            stdexec::__forward_like<Data>(functor));
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Adding 'bulk' node " << get_node_ptr(node) << " to graph " << get_graph_impl_ptr(node)
                  << " after " << get_node_ptr(predecessor) << " on device "
                  << Kokkos::Tools::Experimental::device_id(get_node_ptr(node)->get_device_handle().m_exec) << '.';
#endif
        graph_add_node_event<PredecessorIsRoot>(predecessor, node);
        return node;
    }

    node_props_t node_props;
    Data data; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

//! Sender for @c stdexec::bulk.
template <Kokkos::ExecutionSpace Exec, stdexec::sender Sndr, typename Data>
struct BulkSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = BulkClosure<Exec, Data>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(BulkSender, stdexec::set_error_t(std::exception_ptr))

    KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::bulk_t> {
    template <typename Env, typename Data, typename Sndr>
    using trnsfrmd_sndr_t = BulkSender<Impl::exec_of_t<Sndr, Env>, Sndr, Data>;

    template <typename Env, Kokkos::Execution::Impl::has_parallel_policy Data, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env& env, stdexec::bulk_t, Data&& data, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Data, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Data, Sndr>::closure_t,
                 Sndr&&
        >) {
        if constexpr (graph_completing_sender<Sndr, Env>) {
            auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

            return trnsfrmd_sndr_t<Env, Data, Sndr>{
                .clsr =
                    {.node_props = Kokkos::Experimental::node_props(
                         std::string(Impl::dispatch_label<Impl::exec_of_t<Sndr, Env>, ": bulk">()),
                           Kokkos::Experimental::get_device_handle(schd.state->exec)),
                           .data = std::forward<Data>(data)},
                .sndr = std::forward<Sndr>(sndr)
            };
        } else {
            return no_graph_scheduler_in_env<stdexec::bulk_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace stdexec::__detail {
template <Kokkos::ExecutionSpace Exec, typename Sndr, typename Data>
extern __mtype<Kokkos::Execution::GraphImpl::BulkSender<Exec, __demangle_t<Sndr>, Data>>
    __demangle_v<Kokkos::Execution::GraphImpl::BulkSender<Exec, Sndr, Data>>;
} // namespace stdexec::__detail
// NOLINTEND(bugprone-reserved-identifier)

#endif // KOKKOS_EXECUTION_GRAPH_BULK_HPP
