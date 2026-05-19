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
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
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

    using root_t = typename Kokkos::Experimental::Graph<execution_space>::root_t;

    //! Receiver for a child of @c stdexec::when_all.
    struct WhenAllChildReceiver : public Impl::Receiver<WhenAllOpState, stdexec::env_of_t<Rcvr>> {
        auto query(get_node_t) const & noexcept -> const root_t& {
            return this->parent_op->root;
        }
    };

    using state_t = State<GraphComposition::Create, execution_space>;
    using children_opstates_t = stdexec::__tuple<stdexec::connect_result_t<Sndrs, WhenAllChildReceiver>...>;

    static constexpr bool as_one = (Impl::remains_on<stdexec::set_value_t, Sndrs, Domain> && ...);

    using node_t = decltype(stdexec::__apply(
        [](const auto&... ops) { return Kokkos::Experimental::when_all(ops.query(get_node)...); },
        std::declval<const children_opstates_t&>()));

    state_t state;
    //! @todo The root node is stored to avoid reference counting incurred by https://github.com/kokkos/kokkos/blob/1945b637c3fab027fe90208753e8b2ec236302d4/core/src/Kokkos_Graph.hpp#L100.
    root_t root;
    children_opstates_t children_opstates;
    node_t node;
    std::atomic<size_t> count = sizeof...(Sndrs);

    //! @warning Unconditionally **not** @c noexcept because both graph and node construction may throw.
    WhenAllOpState(stdexec::__tuple<Sndrs...>&& sndrs_, Rcvr&& rcvr_)
        : base_t(std::move(rcvr_))
        /**
         * @todo The graph will be created on the default device and submitted on the default execution space instance.
         *       The device selection for each node will still happen correctly occording to each node properties.
         *       A possibility could be to ask the successor of @c stdexec::when_all for an execution space instance
         *       to submit the graph onto.
         */
        , state{Kokkos::Experimental::get_device_handle(execution_space{})}
        , root(state.graph.root_node())
        , children_opstates(
              stdexec::__apply(
                  [this]<typename... Children>(Children&&... children) -> children_opstates_t {
                      return children_opstates_t{
                          stdexec::connect(std::forward<Children>(children), WhenAllChildReceiver{this})...};
                  },
                  std::move(sndrs_)))
        , node(
              stdexec::__apply(
                  [](const auto&... child_op) {
                      auto agg = Kokkos::Experimental::when_all(child_op.query(get_node)...);
                      graph_add_aggregate_node_event(agg, child_op.query(get_node)...);
                      return agg;
                  },
                  children_opstates)) {
    }

    const auto& query(get_node_t) const & noexcept {
        return node;
    }

    const auto& query(get_graph_t) const & noexcept {
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
    void complete(stdexec::set_value_t) & noexcept requires(!as_one)
    {
        if (count.fetch_sub(1) == 1) {
            this->submit();
        }
    }

    void start() & noexcept requires(!as_one)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Starting all branches before submission.";
#endif
        stdexec::__apply([](auto&... ops) -> void { (stdexec::start(ops), ...); }, children_opstates);
    }

    //! If @ref as_one is @c true, there is no need to start the branches.
    void start() & noexcept requires(as_one)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Submit the graph directly without starting the branches.";
#endif
        this->submit();
    }

    void submit() & noexcept {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "Submitting graph " << get_graph_impl_ptr(state.graph.root_node()) << " on "
                  << Kokkos::Tools::Experimental::device_id(state.get_device_handle().m_exec) << '.';
#endif
        submit_graph(state.graph, state.get_device_handle().m_exec);
        this->completion_signal.propagate(state.get_device_handle().m_exec);
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> stdexec::env_of_t<Rcvr> {
        return stdexec::get_env(this->completion_signal.rcvr);
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

template <size_t Index, typename Sndr>
using WITH_PRETTY_SENDER_AT_INDEX = WITH_SENDER_AT_INDEX<Index, stdexec::__demangle_t<Sndr>>;

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
                    WITH_PRETTY_SENDER_AT_INDEX<index, invalid_sndr_t>,
                    stdexec::_WITH_PRETTY_SENDERS_<Sndrs...>,
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

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace stdexec::__detail {
template <typename... Sndrs>
extern __mtype<Kokkos::Execution::GraphImpl::WhenAllSender<__demangle_t<Sndrs>...>>
    __demangle_v<Kokkos::Execution::GraphImpl::WhenAllSender<Sndrs...>>;
} // namespace stdexec::__detail
// NOLINTEND(bugprone-reserved-identifier)

#endif // KOKKOS_EXECUTION_GRAPH_WHEN_ALL_HPP
