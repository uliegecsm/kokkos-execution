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
#include "kokkos-execution/impl/completion_signal.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/immovable.hpp"
#include "kokkos-execution/impl/make_opstate.hpp"
#include "kokkos-execution/impl/receiver.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/submitted.hpp"

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

    using device_handle_t = Kokkos::Impl::DeviceHandle<Exec>;

    using graph_t = Kokkos::Experimental::Graph<Exec>;
    using root_t = typename graph_t::root_t;

    graph_t graph;
    //! @todo The root node is stored to avoid reference counting incurred by https://github.com/kokkos/kokkos/blob/1945b637c3fab027fe90208753e8b2ec236302d4/core/src/Kokkos_Graph.hpp#L100.
    mutable root_t m_root{};

    explicit State(const device_handle_t& device_handle)
        : graph(Kokkos::Execution::GraphImpl::create_graph(device_handle)) {
    }

    const device_handle_t& get_device_handle() const {
        return graph.get_device_handle();
    }

    const root_t& get_root_node() const {
        if (Kokkos::Impl::GraphAccess::get_node_ptr(m_root) == nullptr) {
            m_root = graph.root_node();
        }
        return m_root;
    }
};

/**
 * @brief Operation state whose sole purpose is to propagate the completion signal.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/56613d3498bc39724dfbae0914cff2aaf3f9dcc6/include/stdexec/__detail/__receiver_adaptor.hpp#L111.
 */
template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct OpStateBase {
    static consteval auto select_completion_signal_policy() noexcept {
        if constexpr (Impl::supports_submitted_order_on<Rcvr>) {
            return Impl::SubmittedPolicy::OrderOnExec{};
        } else {
            return Impl::SyncPolicy::InlineFenceExec{};
        }
    }

    using completion_signal_policy_t = decltype(select_completion_signal_policy());
    using completion_signal_t = Impl::CompletionSignal<completion_signal_policy_t, Exec, Rcvr>;

    completion_signal_t completion_signal;

    constexpr explicit OpStateBase(Rcvr rcvr) noexcept(std::is_nothrow_constructible_v<completion_signal_t, Rcvr&&>)
        : completion_signal(std::move(rcvr)) {
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return stdexec::get_env(this->completion_signal.rcvr);
    }
};

//! Add all nodes as a sequence. Hence, only the first node may be added after the root node.
template <typename Predecessor, Closure FirstClosure, Closure... RestOfClosures>
requires NodeRef<std::remove_cvref_t<Predecessor>>
static auto add_nodes(Predecessor&& predecessor, FirstClosure&& clsr, RestOfClosures&&... clsrs) {
    auto node = std::forward<FirstClosure>(clsr).add_node(std::forward<Predecessor>(predecessor));
    if constexpr (sizeof...(RestOfClosures) == 0) {
        return node;
    } else {
        return add_nodes(std::move(node), std::forward<RestOfClosures>(clsrs)...);
    }
}

//! Operation state that adds all closures as a sequence of nodes.
template <stdexec::sender Sndr, stdexec::receiver Rcvr, Closure FirstClosure, Closure... RestOfClosures>
struct OpState
    : public Impl::Immovable
    , public OpStateBase<typename FirstClosure::execution_space, Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = OpStateBase<typename FirstClosure::execution_space, Rcvr>;

    using execution_space = typename FirstClosure::execution_space;
    using device_handle_t = typename FirstClosure::device_handle_t;

    //! Ensure that all closures are on the same execution space type.
    static_assert((std::same_as<typename RestOfClosures::execution_space, execution_space> && ...));

    using rcvr_t = Impl::Receiver<OpState>;

    using inner_opstate_t = stdexec::connect_result_t<Sndr, rcvr_t>;
    using graph_composition_policy_t = GraphComposition::policy_t<inner_opstate_t, Rcvr>;
    using state_t = State<graph_composition_policy_t, execution_space>;
    using predecessor_t = GraphComposition::node_t<execution_space, inner_opstate_t, Rcvr>;

    static constexpr bool after_root = std::same_as<graph_composition_policy_t, GraphComposition::Create>;

    using node_t = decltype(add_nodes(
        std::declval<predecessor_t>(),
        std::declval<FirstClosure>(),
        std::declval<RestOfClosures>()...));

    inner_opstate_t inner_opstate;
    state_t state;
    node_t node;

    //! @warning Unconditionally **not** @c noexcept because both graph and node construction may throw.
    constexpr OpState(
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        Rcvr rcvr,
        FirstClosure clsr,
        RestOfClosures... clsrs) noexcept(false)
        : OpStateBase<execution_space, Rcvr>(std::move(rcvr))
        , inner_opstate(stdexec::connect(std::forward<Sndr>(sndr), rcvr_t{this}))
        , state{Kokkos::Impl::get_property<device_handle_t>(clsr.node_props)}
        , node{add_nodes(this->get_predecessor(), std::move(clsr), std::move(clsrs)...)} {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Operation state graph composition policy is "
                  << Kokkos::Impl::TypeInfo<graph_composition_policy_t>::name() << ", the receiver is of type "
                  << Kokkos::Impl::TypeInfo<Rcvr>::name() << " and the inner operation state is of type "
                  << Kokkos::Impl::TypeInfo<inner_opstate_t>::name() << '.';
#endif
    }

    /**
     * If the graph composition policy is @ref GraphComposition::Create, return the root node of @ref state graph.
     * Otherwise, return the result of querying @ref inner_opstate for @ref get_node_t.
     */
    const predecessor_t& get_predecessor() const noexcept {
        if constexpr (after_root) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_INFO << "The predecessor is the root node of graph " << get_graph_impl_ptr(state.get_root_node())
                      << '.';
#endif
            return state.get_root_node();
        } else {
            if constexpr (stdexec::__queryable_with<inner_opstate_t, get_node_t>) {
                return inner_opstate.query(get_node);
            } else {
                return this->completion_signal.rcvr.query(get_node);
            }
        }
    }

    template <typename Error>
    void complete(stdexec::set_error_t, Error&& error) noexcept {
        stdexec::set_error(std::move(this->completion_signal.rcvr), std::forward<Error>(error));
    }

    void complete(stdexec::set_stopped_t) noexcept {
        stdexec::set_stopped(std::move(this->completion_signal.rcvr));
    }

    void submit() noexcept {
        if constexpr (after_root) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_INFO << "Submitting graph " << get_graph_impl_ptr(state.get_root_node()) << " on "
                      << Kokkos::Tools::Experimental::device_id(state.get_device_handle().m_exec) << '.';
#endif
            submit_graph(state.graph, state.get_device_handle().m_exec);
        }
        this->completion_signal.propagate(state.get_device_handle().m_exec);
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

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->completion_signal.rcvr)
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

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING) // NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#    define KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING(_type_, _node_, _predecessor_)                          \
        PLOG_INFO << "Adding '" _type_ "' node " << get_node_ptr(_node_) << " to graph " << get_graph_impl_ptr(_node_) \
                  << " after " << get_node_ptr(_predecessor_) << " on device "                                         \
                  << Kokkos::Tools::Experimental::device_id(get_node_ptr(_node_)->get_device_handle().m_exec) << '.';
#else
#    define KOKKOS_EXECUTION_IMPL_GRAPH_ADD_NODE_DEBUG_LOGGING(_type_, _node_, _predecessor_)
#endif
} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_OPERATION_STATE_HPP
