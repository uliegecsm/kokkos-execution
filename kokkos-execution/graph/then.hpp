#ifndef KOKKOS_EXECUTION_GRAPH_THEN_HPP
#define KOKKOS_EXECUTION_GRAPH_THEN_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.graph;
import kokkos.graph_impl;
#else
#    include "Kokkos_Core.hpp"
#    include "Kokkos_Graph.hpp"
#endif

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
#include "kokkos-execution/impl/state.hpp"

namespace Kokkos::Execution::GraphImpl {

template <Kokkos::ExecutionSpace Exec, typename Label, typename Functor>
struct ThenClosure {
    using execution_space = Exec;

    using device_handle_t = Kokkos::Impl::DeviceHandle<execution_space>;

    using node_props_t = typename Kokkos::Impl::NodeCtorProps<device_handle_t, std::string>::uniform_type;

    template <NodeRef Predecessor>
    using node_t = decltype(std::declval<const Predecessor&>()
                                .then(std::declval<node_props_t&&>(), std::declval<Functor&&>()));

    //! @warning Unconditionally **not** @c noexcept because adding a node may throw.
    template <NodeRef Predecessor>
    auto add_node(const Predecessor& predecessor) && -> node_t<Predecessor> {
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        std::string label_copy(label);
#endif
        node_props_t node_props = std::move(*this).get_node_props();
        auto node = predecessor.then(std::move(node_props), std::move(functor));
#if defined(KOKKOS_EXECUTION_ENABLE_EVENT_DISPATCH) || defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING(label_copy, node, predecessor)
        graph_add_node_event(std::move(label_copy), predecessor, node);
#endif
        return node;
    }

    template <typename Self>
    [[nodiscard]]
    STDEXEC_EXPLICIT_THIS_BEGIN(auto get_node_props)(this Self&& self) STDEXEC_AUTO_RETURN( // NOLINT(cppcoreguidelines-missing-std-forward)
        Kokkos::Experimental::node_props(
            self.get_device_handle(),
            std::string(stdexec::__forward_like<Self>(self.label))))
    STDEXEC_EXPLICIT_THIS_END(get_node_props)

    [[nodiscard]]
    auto get_device_handle() const & noexcept -> device_handle_t {
        return Kokkos::Experimental::get_device_handle(state->exec);
    }

    const Impl::State<execution_space>* state;
    Label label;
    Functor functor; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

//! Sender for @c stdexec::then.
template <typename Sndr, Kokkos::ExecutionSpace Exec, typename Label, typename Functor>
struct ThenSender {
    using sender_concept = stdexec::sender_tag;

    using closure_t = ThenClosure<Exec, Label, Functor>;

    KOKKOS_EXECUTION_COMPL_SIGS_ADD(ThenSender, Sndr, stdexec::set_error_t(std::exception_ptr))

    KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    closure_t clsr;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::then_t> {
    template <typename Env, typename Functor, typename Sndr>
    struct Transformer {
        using execution_space = Impl::exec_of_t<Sndr, Env>;

        using sndr_t = ThenSender<Sndr, execution_space, std::string_view, std::remove_cvref_t<Functor>>;

        auto operator()(const Env& env, std::remove_reference_t<Functor>& functor, std::remove_reference_t<Sndr>& sndr)
            const STDEXEC_AUTO_RETURN(
                sndr_t{
                    {.state = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env)
                                  .state,
                     .label = KOKKOS_EXECUTION_IMPL_DISPATCH_LABEL(execution_space, ": then"),
                     .functor = std::forward<Functor>(functor)},
                    std::forward<Sndr>(sndr)
        })
    };

    template <typename Env, typename Functor, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && graph_completing_sender<Sndr, Env>)
    auto operator()(
        const Env& env,
        stdexec::then_t,
        Functor&& functor, // NOLINT(cppcoreguidelines-missing-std-forward)
        Sndr&& sndr        // NOLINT(cppcoreguidelines-missing-std-forward)
    ) const STDEXEC_AUTO_RETURN(Transformer<Env, Functor, Sndr>{}(env, functor, sndr))

    template <typename Env, typename Functor, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, stdexec::then_t, Functor&&, Sndr&&) const noexcept {
        return no_graph_scheduler_in_env<stdexec::then_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_THEN_HPP
