#ifndef KOKKOS_EXECUTION_GRAPH_WHEN_ALL_HPP
#define KOKKOS_EXECUTION_GRAPH_WHEN_ALL_HPP

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
#include "kokkos-execution/impl/diagnostics.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/queryable.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::GraphImpl {

/**
 * @brief Operation state for @c stdexec::when_all.
 *
 * It creates the graph, and passes it to its branches through
 * the receiver environment.
 */
template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr, stdexec::sender... Sndrs>
struct WhenAllOpState
    : public Impl::Immovable
    , public OpStateBase<Exec, Rcvr> {
    using operation_state_concept = stdexec::operation_state_tag;

    using base_t = OpStateBase<Exec, Rcvr>;
    using execution_space = Exec;

    using predecessor_t = GraphComposition::node_t<Exec, Rcvr>;

    //! Receiver for a child of @c stdexec::when_all.
    struct WhenAllChildReceiver : public Impl::Receiver<WhenAllOpState, stdexec::env_of_t<Rcvr>> {
        /**
         * @note This function is not declared @c constexpr because while the child operation-state types are being formed,
         *       the parent operation state may still be incomplete. With @c constexpr, @c clang may attempt constant evaluation
         *       through @c parent_op leading to an incomplete type compilation error.
         */
        [[nodiscard]]
        auto query(get_node_t) const & noexcept -> const predecessor_t& {
            return this->parent_op->get_predecessor();
        }
    };

    using children_op_states_t = stdexec::__tuple<stdexec::connect_result_t<Sndrs, WhenAllChildReceiver>...>;

#if defined(KOKKOS_ENABLE_DEBUG)
    //! If this assertion fails, it probably means that one of the branches finishes with an algorithm that is not customized yet.
    static_assert(
        stdexec::__mapply<
            stdexec::__mall_of<stdexec::__q<Impl::queryable_for<get_node_t>::type>>,
            children_op_states_t
        >::value,
        "Child senders of the 'when_all' must lead to 'get_node_t' queryable operation states.");
#endif

    //! Determine if all branches remain fully on the graph, if connected to @ref WhenAllChildReceiver.
    static constexpr bool as_one =
        (connect_result_remains_on_graph_for<execution_space, Sndrs, WhenAllChildReceiver> && ...);

    static constexpr bool adds_anything = stdexec::__mapply<
        stdexec::__many_of<stdexec::__q<Impl::queryable_for<get_node_t>::type>>,
        children_op_states_t
    >::value;

    //! Join the nodes of all operation states into one aggregate node.
    struct JoinOpStateNodes {
        template <typename ChildrenOpStates>
        constexpr auto operator()(const ChildrenOpStates& ops) const {
            // only add those ops.query when it is queryable...
            // auto agg = Kokkos::Experimental::when_all(ops.query(get_node)...);
            // graph_add_aggregate_node_event(agg, ops.query(get_node)...);
            // return agg;
            auto nodes = stdexec::__apply(
                []<typename... OpStates>(const OpStates&... op_states) {
                    return std::tuple_cat([&] {
                        if constexpr (requires { op_states.query(get_node); })
                            return std::make_tuple(op_states.query(get_node));
                        else
                            return std::tuple{};
                    }()...);
                },
                ops);


            return std::apply(
                [&]<typename... Nodes>(const Nodes&... ns) {
                    if constexpr (sizeof...(Nodes) == 1) {
                        return (ns, ...);
                    } else {
                        auto agg = Kokkos::Experimental::when_all(ns...);
                        graph_add_aggregate_node_event(agg, ns...);
                        return agg;
                    }
                },
                nodes);
        }
    };

    //! Counterpart for the "transparent" case: there is no node to produce.
    struct NoOpStateNode {
        template <typename... OpStates>
        constexpr auto operator()(const OpStates&...) const noexcept {
            return Impl::Empty{};
        }
    };

    using joiner_t = std::conditional_t<adds_anything, JoinOpStateNodes, NoOpStateNode>;
    // using node_t = decltype(stdexec::__apply(joiner_t{}, std::declval<const children_op_states_t&>()));
    using node_t = std::invoke_result_t<joiner_t, const children_op_states_t&>;

    /// If @ref adds_nodes is @c false, the composition policy is unconditionally @ref GraphComposition::Attach
    /// to avoid creating a useless @c Kokkos::Experimental::Graph.
    // using graph_composition_policy_t =
    //     std::conditional_t<adds_nodes, GraphComposition::policy_t<Rcvr>, GraphComposition::Attach>;
    using graph_composition_policy_t = GraphComposition::policy_t<Rcvr>;
    // using state_t = State<graph_composition_policy_t, execution_space>;
    using predecessor_t = GraphComposition::node_t<Exec, Rcvr>;
    using state_t = std::conditional_t<
        adds_anything,
        State<graph_composition_policy_t, execution_space>,
        State<GraphComposition::Attach, execution_space>
    >;

    [[no_unique_address]]
    state_t state;
    children_op_states_t children_op_states;
    [[no_unique_address]]
    node_t node;
    std::atomic<size_t> count = sizeof...(Sndrs);

    //! @warning Unconditionally **not** @c noexcept because both graph and node construction may throw.
    WhenAllOpState(stdexec::__tuple<Sndrs...>&& sndrs_, Rcvr&& rcvr_)
        : base_t(std::move(rcvr_))
        /**
         * @todo The graph will be created on the default device and submitted on the default execution space instance.
         *       The device selection for each node will still happen correctly according to each node properties.
         *       A possibility could be to ask the successor of @c stdexec::when_all for an execution space instance
         *       to submit the graph onto.
         */
        , state{Kokkos::Experimental::get_device_handle(execution_space{})}
        , children_op_states(
              stdexec::__apply(
                  [this](auto&&... children) -> children_op_states_t {
                      static_assert((std::same_as<decltype(children), Sndrs&&> && ...));
                      return children_op_states_t{
                          stdexec::connect(std::forward<Sndrs>(children), WhenAllChildReceiver{this})...};
                  },
                  std::move(sndrs_)))
        , node(joiner_t{}(children_op_states)) {
    }

    const predecessor_t& get_predecessor() const noexcept {
        if constexpr (std::same_as<graph_composition_policy_t, GraphComposition::Create>) {
            return state.get_root_node();
        } else {
            return this->completion_signal.rcvr.query(get_node);
        }
    }

    [[nodiscard]]
    auto query(get_node_t) const & noexcept -> const node_t& {
        return node;
    }

    [[nodiscard]]
    auto query(get_graph_t) const & noexcept -> const typename state_t::graph_t& {
        return state.graph;
    }

    /**
     * Each branch calls this method if @ref as_one is @c false.
     * Once the counter decreases to zero, the graph is submitted.
     * It is intended, because there is no strong check on the branch content if @ref as_one is @c false.
     * Indeed, one of the branches may start with operations that are not part of the graph,
     * implying that the graph cannot be submitted by one of the other branches.
     *
     * The following schematic illustrates such a situation.
     *
     * @verbatim
     * (branch 1) -> (thread) -> (then) -> (graph) -> (then) -\
     *                                                         -> (when_all)
     * (branch 0) -> (graph) -> (bulk)                       -/
     * @endverbatim
     *
     * If the branch 0 submits the graph before calling @ref WhenAllOpState::complete,
     * the graph will be submitted before the branch 1 is started, leading to operations of
     * the graph in branch 1 potentially running before the "thread".
     *
     * @todo Add a test for the above schematic.
     *
     * In the future, a solution could be to create one graph per branch in such a setting, but it would
     * require introspecting the sender type of each branch, recursively.
     */
    void submit() & noexcept requires(!as_one)
    {
        if (count.fetch_sub(1) == 1) {
            this->submit_graph();
        }
    }

    void submit() & noexcept requires(as_one)
    {
        this->submit_graph();
    }

    void complete(stdexec::set_value_t) & noexcept {
        this->submit();
    }

    template <typename Tag, typename... Args>
    requires(!std::same_as<Tag, stdexec::set_value_t>)
    void complete(Tag, Args&&... args) & noexcept {
        base_t::complete(Tag{}, std::forward<Args>(args)...);
    }

    void start() & noexcept requires(!as_one)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Starting all branches before submission.";
#endif
        stdexec::__apply([](auto&... ops) -> void { (stdexec::start(ops), ...); }, children_op_states);
    }

    //! If @ref as_one is @c true, there is no need to start the branches.
    void start() & noexcept requires(as_one)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Submit the graph directly without starting the branches.";
#endif
        this->submit();
    }

    void submit_graph() & noexcept requires(!adds_anything)
    {
        stdexec::set_value(std::move(this->completion_signal.rcvr));
    }

    void submit_graph() & noexcept
        requires(std::same_as<graph_composition_policy_t, GraphComposition::Create> && adds_anything)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Submitting graph " << get_graph_impl_ptr(state.graph.root_node()) << " on "
                  << Kokkos::Tools::Experimental::device_id(state.get_device_handle().m_exec) << '.';
#endif
        try {
            Kokkos::Execution::GraphImpl::submit_graph(state.graph, state.get_device_handle().m_exec);
        } catch (...) {
            stdexec::set_error(std::move(this->completion_signal.rcvr), std::current_exception());
            return;
        }
        this->completion_signal.propagate(state.get_device_handle().m_exec);
    }

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->completion_signal.rcvr)
};

//! Specialization for @ref Kokkos::Execution::GraphImpl::WhenAllOpState. @todo To be removed and done properly.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::WhenAllOpState>
    && std::same_as<typename OpState::execution_space, Exec>)
struct GraphOperationStateFor<OpState, Exec> : public std::true_type { };

//! Specialization for @ref Kokkos::Execution::GraphImpl::WhenAllOpState.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    graph_operation_state_for<OpState, Exec>
    && stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::WhenAllOpState>)
struct RemainsOnGraphFor<OpState, Exec> {
    template <stdexec::operation_state ChildOpState>
    using RemainsOnGraphForChild = RemainsOnGraphFor<ChildOpState, Exec>;

    static constexpr bool value = stdexec::__mapply<
        stdexec::__mall_of<stdexec::__q<RemainsOnGraphForChild>>,
        typename OpState::children_op_states_t
    >::value;

    static constexpr auto diagnose() noexcept {
        using children_t = stdexec::__mapply<
            stdexec::__mtransform<stdexec::__q<RemainsOnGraphForChild>, stdexec::__q<Impl::diagnose_t>>,
            typename OpState::children_op_states_t
        >;

        if constexpr (stdexec::__ok<Impl::first_error_t<children_t>>) {
            return stdexec::__msuccess{};
        } else {
            return stdexec::__mexception<
                stdexec::_WHAT_(OPERATION_STATE_DOES_NOT_REMAIN_ON_GRAPH),
                stdexec::_WHY_(BECAUSE_AT_LEAST_ONE_CHILD_DOES_NOT_REMAIN_ON_GRAPH),
                Impl::rebind_t<Impl::WITH_CHILDREN_ERRORS, Impl::keep_errors_t<children_t>>,
                Impl::WITH_OPSTATE<OpState>
            >{};
        }
    }
};

//! Sender for @c stdexec::when_all.
template <Kokkos::ExecutionSpace Exec, stdexec::sender... Sndrs>
struct WhenAllSender {
    using sender_concept = stdexec::sender_tag;

    using sndrs_t = stdexec::__tuple<Sndrs...>;

    struct attrs {
        template <typename... Env>
        [[nodiscard]]
        constexpr auto
            query(stdexec::get_completion_domain_t<stdexec::set_value_t>, const Env&...) const noexcept -> Domain {
            return {};
        }
    };

    //! @todo Make it much more robust as per https://github.com/NVIDIA/stdexec/blob/da7640aa1002cfcd665254f03434d3c949844361/include/stdexec/__detail/__when_all.hpp#L426.
    template <typename Self, typename... Env>
    static consteval auto get_completion_signatures() {
        return stdexec::completion_signatures<stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>{};
    }

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<WhenAllOpState<Exec, Rcvr, Sndrs...>, sndrs_t&&, Rcvr&&>) {
        return WhenAllOpState<Exec, Rcvr, Sndrs...>(std::move(sndrs), std::move(rcvr));
    }

    constexpr auto get_env() const noexcept -> attrs {
        return {};
    }

    sndrs_t sndrs;
};

struct BECAUSE_THE_EXECUTION_SPACE_TYPE_IS_NOT_HOMOGENEOUS;

template <size_t Index, typename Sndr>
struct WITH_SENDER_AT_INDEX { };

template <>
struct TransformSenderFor<stdexec::when_all_t> {
    template <typename Env, typename... Sndrs>
    using trnsfrmd_sndr_t = WhenAllSender<Impl::exec_of_t<stdexec::__m_at_c<0, Sndrs...>, Env>, Sndrs...>;

    template <typename Env, typename... Sndrs>
    auto operator()(const Env&, stdexec::when_all_t, stdexec::__ignore, Sndrs&&... sndrs) const
        noexcept(std::is_nothrow_constructible_v<typename trnsfrmd_sndr_t<Env, Sndrs...>::sndrs_t, Sndrs&&...>) {
        if constexpr ((graph_completing_sender<Sndrs, Env> && ...)) {
            using execution_space = Impl::exec_of_t<stdexec::__m_at_c<0, Sndrs...>, Env>;

            //! Ensure that all senders are on the same execution space type.
            if constexpr ((std::same_as<Impl::exec_of_t<Sndrs, Env>, execution_space> && ...)) {
                return trnsfrmd_sndr_t<Env, Sndrs...>{.sndrs = {std::forward<Sndrs>(sndrs)...}};
            } else {
                //! From https://github.com/NVIDIA/stdexec/blob/da7640aa1002cfcd665254f03434d3c949844361/include/nvexec/stream/when_all.cuh#L550-L554.
                STDEXEC_CONSTEXPR_LOCAL bool map[] = {
                    !std::same_as<Impl::exec_of_t<stdexec::__m_at_c<0, Sndrs>, Env>, execution_space>...};
                STDEXEC_CONSTEXPR_LOCAL std::size_t index = stdexec::__pos_of(map, map + sizeof...(Sndrs));
                using invalid_sndr_t = stdexec::__m_at_c<index, Sndrs...>;
                return stdexec::__not_a_sender<
                    stdexec::_WHAT_(CANNOT_DISPATCH_THIS_ALGORITHM_TO_THE_GRAPH_SCHEDULER),
                    stdexec::_WHY_(BECAUSE_THE_EXECUTION_SPACE_TYPE_IS_NOT_HOMOGENEOUS),
                    stdexec::_WHERE_(stdexec::_IN_ALGORITHM_, stdexec::when_all_t),
                    WITH_SENDER_AT_INDEX<index, invalid_sndr_t>,
                    stdexec::_WITH_SENDERS_<Sndrs...>,
                    stdexec::_WITH_ENVIRONMENT_(Env)
                >{};
            }
        } else {
            //! From https://github.com/NVIDIA/stdexec/blob/da7640aa1002cfcd665254f03434d3c949844361/include/nvexec/stream/when_all.cuh#L550-L554.
            STDEXEC_CONSTEXPR_LOCAL bool map[] = {!graph_completing_sender<Sndrs, Env>...};
            STDEXEC_CONSTEXPR_LOCAL std::size_t index = stdexec::__pos_of(map, map + sizeof...(Sndrs));
            using invalid_sndr_t = stdexec::__m_at_c<index, Sndrs...>;
            return no_graph_scheduler_in_env<stdexec::when_all_t, invalid_sndr_t, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_WHEN_ALL_HPP
