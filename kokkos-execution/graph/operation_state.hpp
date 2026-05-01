#ifndef KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_HPP
#define KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

#include "kokkos-execution/graph/events.hpp"
#include "kokkos-execution/graph/get_graph.hpp"
#include "kokkos-execution/graph/get_node.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/impl/immovable.hpp"
#include "kokkos-execution/impl/make_opstate.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"

namespace Kokkos::Execution::GraphImpl {

template <typename Clsr>
concept Closure = requires {
    typename Clsr::execution_space;
    requires std::same_as<typename Clsr::device_handle_t, Kokkos::Impl::DeviceHandle<typename Clsr::execution_space>>;
    typename Clsr::node_props_t;
};

template <typename GraphCompositionPolicy, Kokkos::ExecutionSpace Exec>
struct State;

//! If the graph composition policy is @ref GraphComposition::Attach, nothing needs to be stored.
template <Kokkos::ExecutionSpace Exec>
struct State<GraphComposition::Attach, Exec> {
    using graph_composition_policy_t = GraphComposition::Attach;

    explicit State(const Kokkos::Impl::DeviceHandle<Exec>&) {
    }
};

//! If the graph composition policy is @ref GraphComposition::Create, the device handle is already stored by the @ref graph itself.
template <Kokkos::ExecutionSpace Exec>
struct State<GraphComposition::Create, Exec> {
    using graph_composition_policy_t = GraphComposition::Create;

    using graph_t = Kokkos::Experimental::Graph<Exec>;

    graph_t graph;

    explicit State(const Kokkos::Impl::DeviceHandle<Exec>& device_handle)
        : graph(Kokkos::Execution::GraphImpl::create_graph(device_handle)) {
    }

    const auto& get_device_handle() const {
        return graph.get_device_handle();
    }
};

/**
 * @brief Operation state whose sole purpose is to propagate the completion signal to @ref rcvr.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/56613d3498bc39724dfbae0914cff2aaf3f9dcc6/include/stdexec/__detail/__receiver_adaptor.hpp#L111.
 */
template <stdexec::receiver Rcvr>
struct OpStateBase {
    using receiver_t = Rcvr;

    receiver_t rcvr;

    void propagate_completion_signal(stdexec::set_value_t) noexcept
        requires(stdexec::__callable<stdexec::set_value_t, receiver_t &&>)
    {
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void propagate_completion_signal(stdexec::set_error_t, Error&& error) noexcept
        requires(stdexec::__callable<stdexec::set_error_t, receiver_t &&, Error>)
    {
        stdexec::set_error(std::move(rcvr), std::forward<Error>(error));
    }

    void propagate_completion_signal(stdexec::set_stopped_t) noexcept
        requires(stdexec::__callable<stdexec::set_stopped_t, receiver_t &&>)
    {
        stdexec::set_stopped(std::move(rcvr));
    }
};

//! Add all nodes as a sequence. Hence, only the first node may be added after the root node.
template <bool PredecessorIsRoot, NodeRef Predecessor, Closure FirstClosure, Closure... RestOfClosures>
static auto add_nodes(Predecessor&& predecessor, FirstClosure&& clsr, RestOfClosures&&... clsrs) {
    auto node = std::forward<FirstClosure>(clsr)
                    .template add_node<PredecessorIsRoot>(std::forward<Predecessor>(predecessor));
    if constexpr (sizeof...(RestOfClosures) == 0) {
        return node;
    } else {
        return add_nodes<false>(std::move(node), std::forward<RestOfClosures>(clsrs)...);
    }
}

//! Operation state that adds all closures as a sequence of nodes.
template <stdexec::sender Sndr, stdexec::receiver Rcvr, Closure FirstClosure, Closure... RestOfClosures>
struct OpState
    : public Impl::Immovable
    , public OpStateBase<Rcvr> {
    using operation_state_concept = stdexec::operation_state_tag;

    using execution_space = typename FirstClosure::execution_space;
    using device_handle_t = typename FirstClosure::device_handle_t;

    //! Ensure that all closures are on the same execution space type.
    static_assert((std::same_as<typename RestOfClosures::execution_space, execution_space> && ...));

    using rcvr_t = Impl::Receiver<OpState>;
    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;
    using graph_composition_policy_t = GraphComposition::policy_t<inner_opstate_t>;
    using state_t = State<graph_composition_policy_t, execution_space>;
    using predecessor_t = GraphComposition::node_t<execution_space, inner_opstate_t>;

    static constexpr bool after_root = std::same_as<graph_composition_policy_t, GraphComposition::Create>;

    using node_t = decltype(add_nodes<after_root>(
        std::declval<predecessor_t>(),
        std::declval<FirstClosure>(),
        std::declval<RestOfClosures>()...));

    inner_opstate_t inner_opstate;
    state_t state;
    node_t node;

    //! @warning Unconditionally **not** @c noexcept because both graph and node construction may throw.
    OpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr_,
        FirstClosure clsr,
        RestOfClosures... clsrs) noexcept(false)
        : OpStateBase<Rcvr>{.rcvr = std::move(rcvr_)}
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this}))
        , state{Kokkos::Impl::get_property<device_handle_t>(clsr.node_props)}
        , node{add_nodes<after_root>(this->get_predecessor(), std::move(clsr), std::move(clsrs)...)} {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Operation state graph composition policy is "
                  << Kokkos::Impl::TypeInfo<graph_composition_policy_t>::name()
                  << " and the inner operation state is of type " << Kokkos::Impl::TypeInfo<inner_opstate_t>::name()
                  << '.';
#endif
    }

    /**
     * If the graph composition policy is @ref GraphComposition::Create, return the root node of @ref state graph.
     * Otherwise, return the result of querying @ref inner_opstate for @ref get_node_t.
     */
    predecessor_t get_predecessor() const {
        if constexpr (after_root) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_INFO << "The predecessor is the root node of graph " << get_graph_impl_ptr(state.graph.root_node())
                      << '.';
#endif
            return state.graph.root_node();
        } else {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_INFO << "The predecessor is the node " << get_node_ptr(inner_opstate.query(get_node)) << " of graph "
                      << get_graph_impl_ptr(inner_opstate.query(get_node)) << '.';
#endif
            return inner_opstate.query(get_node);
        }
    }

    void propagate_completion_signal(stdexec::set_value_t) noexcept {
        if constexpr (after_root) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_INFO << "Submitting graph " << get_graph_impl_ptr(state.graph.root_node()) << " on "
                      << Kokkos::Tools::Experimental::device_id(state.get_device_handle().m_exec) << '.';
#endif
            submit_graph(state.graph, state.get_device_handle().m_exec);
        }
        OpStateBase<Rcvr>::propagate_completion_signal(stdexec::set_value);
    }

    const auto& query(get_node_t) const & noexcept {
        return node;
    }

    const auto& query(get_graph_t) const & noexcept {
        if constexpr (after_root) {
            return state.graph;
        } else {
            return inner_opstate.query(get_graph);
        }
    }

    void start() & noexcept {
        stdexec::start(inner_opstate);
    }
};

template <typename Sndr, typename Rcvr, typename... Clsrs>
using make_opstate_t = Impl::MakeOpState<Domain, OpState>::Huddle<Sndr, Rcvr, Clsrs...>;

template <typename Sndr, typename Rcvr, typename... Clsrs>
using opstate_t = typename make_opstate_t<Sndr, Rcvr, Clsrs...>::type;

#define KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_CONNECT                                                                 \
    template <stdexec::receiver Rcvr>                                                                                  \
    constexpr auto connect(Rcvr rcvr) && noexcept(noexcept(make_opstate_t<Sndr, Rcvr, closure_t>{}(                    \
        std::declval<Sndr>(), std::declval<Rcvr>(), std::declval<closure_t>()))) -> opstate_t<Sndr, Rcvr, closure_t> { \
        return make_opstate_t<Sndr, Rcvr, closure_t>{}(std::forward<Sndr>(sndr), std::move(rcvr), std::move(clsr));    \
    }

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_HPP
