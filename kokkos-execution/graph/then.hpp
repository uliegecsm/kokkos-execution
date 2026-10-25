#ifndef KOKKOS_EXECUTION_GRAPH_THEN_HPP
#define KOKKOS_EXECUTION_GRAPH_THEN_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "Kokkos_Graph.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"

#include "kokkos-execution/graph/events.hpp"
#include "kokkos-execution/graph/node_props.hpp"
#include "kokkos-execution/graph/operation_state.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::GraphImpl {

template <typename Label, typename DeviceHandle, typename Functor>
struct ThenClosure {
    using device_handle_t = DeviceHandle;
    using execution_space = exec_of_t<device_handle_t>;
    using node_props_t = typename Kokkos::Impl::NodeCtorProps<device_handle_t, std::string>::uniform_type;

    template <NodeRef Predecessor>
    using node_t = decltype(std::declval<const Predecessor&>()
                                .then(std::declval<node_props_t&&>(), std::declval<Functor&&>()));

    //! @warning Unconditionally **not** @c noexcept because adding a node may throw.
    template <NodeRef Predecessor>
    auto add_node(const Predecessor& predecessor) && noexcept(false) -> node_t<Predecessor> {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        auto label_copy = std::string(label);
        const auto device_handle_copy = device_handle;
#endif
        auto node = predecessor.then(
            Kokkos::Experimental::node_props(std::move(device_handle), std::string(std::move(label))),
            std::forward<Functor>(functor));
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING(label_copy, node, predecessor)
        graph_add_node_event(std::move(label_copy), predecessor, node, device_handle_copy);
#endif
        return node;
    }

    Label label;
    device_handle_t device_handle;
    Functor functor; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

//! Sender for @c stdexec::then.
template <typename Sndr, typename Label, typename DeviceHandle, typename Functor>
struct ThenSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ThenClosure<Label, DeviceHandle, Functor>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ThenSender, Sndr, stdexec::set_error_t(std::exception_ptr))

    KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::then_t> {
    template <typename Env, typename Functor, typename Sndr>
    using trnsfrmd_sndr_t = ThenSender<
        Sndr,
        std::string_view,
        Kokkos::Impl::DeviceHandle<Impl::exec_of_t<Sndr, Env>>,
        std::remove_cvref_t<Functor>
    >;

    template <typename Env, typename Functor, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::then_t, Functor&&, Sndr&&) const noexcept {
        return no_graph_scheduler_in_env<stdexec::then_t, Sndr, Env>();
    }

    template <typename Env, typename Functor, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && graph_completing_sender<Sndr, Env>)
    auto operator()(const Env& env, stdexec::then_t, Functor&& functor, Sndr&& sndr) const STDEXEC_AUTO_RETURN(
        trnsfrmd_sndr_t<Env, Functor, Sndr>{
            .clsr =
                {.label = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL((Impl::exec_of_t<Sndr, Env>), ": then"),
                       .device_handle = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                                      .state->get_device_handle(),
                       .functor = std::forward<Functor>(functor)},
            .sndr = std::forward<Sndr>(sndr)
    })
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_THEN_HPP
