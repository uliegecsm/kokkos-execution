#ifndef KOKKOS_EXECUTION_GRAPH_THEN_HPP
#define KOKKOS_EXECUTION_GRAPH_THEN_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "Kokkos_Graph.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"

#include "kokkos-execution/graph/events.hpp"
#include "kokkos-execution/graph/operation_state.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::GraphImpl {

template <Kokkos::ExecutionSpace Exec, typename Functor>
struct ThenClosure {
    using execution_space = Exec;
    using device_handle_t = Kokkos::Impl::DeviceHandle<Exec>;
    using node_props_t = typename Kokkos::Impl::NodeCtorProps<std::string, device_handle_t>::uniform_type;

    template <NodeRef Predecessor>
    using node_t = decltype(std::declval<const Predecessor&>()
                                .then(std::declval<node_props_t&&>(), std::declval<Functor&&>()));

    //! @warning Unconditionally **not** @c noexcept because adding a node may throw.
    template <bool PredecessorIsRoot, NodeRef Predecessor>
    auto add_node(const Predecessor& predecessor) && noexcept(false) -> node_t<Predecessor> {
        auto node = predecessor.then(std::move(node_props), std::forward<Functor>(functor));
        KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING("then", node, predecessor)
        graph_add_node_event<PredecessorIsRoot>(predecessor, node);
        return node;
    }
    node_props_t node_props;
    Functor functor; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

//! Sender for @c stdexec::then.
template <typename Exec, typename Sndr, typename Functor>
struct ThenSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ThenClosure<Exec, Functor>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ThenSender, stdexec::set_error_t(std::exception_ptr))

    KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::then_t> {
    template <typename Env, typename Functor, typename Sndr>
    using trnsfrmd_sndr_t = ThenSender<Impl::exec_of_t<Sndr, Env>, Sndr, Functor>;

    template <typename Env, typename Functor, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env& env, stdexec::then_t, Functor&& functor, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 trnsfrmd_sndr_t<Env, Functor, Sndr>,
                 typename trnsfrmd_sndr_t<Env, Functor, Sndr>::closure_t,
                 Sndr&&
        >) {
        if constexpr (graph_completing_sender<Sndr, Env>) {
            auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

            return trnsfrmd_sndr_t<Env, Functor, Sndr>{
                .clsr =
                    {.node_props = Kokkos::Experimental::node_props(
                         std::string(Impl::dispatch_label<Impl::exec_of_t<Sndr, Env>, ": then">()),
                           Kokkos::Experimental::get_device_handle(schd.state->exec)),
                           .functor = std::forward<Functor>(functor)},
                .sndr = std::forward<Sndr>(sndr)
            };
        } else {
            return no_graph_scheduler_in_env<stdexec::then_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace stdexec::__detail {
template <typename Exec, typename Sndr, typename Functor>
extern __mtype<Kokkos::Execution::GraphImpl::ThenSender<Exec, __demangle_t<Sndr>, Functor>>
    __demangle_v<Kokkos::Execution::GraphImpl::ThenSender<Exec, Sndr, Functor>>;
} // namespace stdexec::__detail
// NOLINTEND(bugprone-reserved-identifier)

#endif // KOKKOS_EXECUTION_GRAPH_THEN_HPP
