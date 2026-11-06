#ifndef KOKKOS_EXECUTION_GRAPH_FORK_JOIN_HPP
#define KOKKOS_EXECUTION_GRAPH_FORK_JOIN_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include <exec/fork_join.hpp>
PRAGMA_DIAGNOSTIC_POP

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "Kokkos_Macros.hpp"
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.graph;
#else
#    include "Kokkos_Graph.hpp"
#endif

#include "kokkos-execution/graph/graph_fwd.hpp"

#include "kokkos-execution/graph/events.hpp"
#include "kokkos-execution/graph/operation_state.hpp"
#include "kokkos-execution/graph/sender_concepts.hpp"
#include "kokkos-execution/graph/when_all.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/sender_concepts.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"

namespace Kokkos::Execution::GraphImpl {

/**
 * @brief Sender to be added to each of the @c exec::fork_join branches.
 *
 * Inspired by https://github.com/NVIDIA/stdexec/blob/da7640aa1002cfcd665254f03434d3c949844361/include/exec/fork_join.hpp#L65.
 *
 * Since it is only used for the graph, there is no need to store a variant containing the value or error completion.
 */
template <Kokkos::ExecutionSpace Exec>
struct CacheSender {
    using sender_concept = stdexec::sender_tag;

    template <stdexec::receiver Rcvr>
    struct CacheOpState {
        using operation_state_concept = stdexec::operation_state_tag;

        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
        }

        [[nodiscard]]
        constexpr auto query(Impl::get_exec_t) const noexcept -> Impl::ExecutionSpaceRef<Exec> {
            return Impl::ExecutionSpaceRef{env.state->exec};
        }

        Rcvr rcvr;
        Scheduler<Exec>::Sender::Attributes env;
    };

    template <typename Self, typename... Env>
    static consteval auto get_completion_signatures() {
        return stdexec::completion_signatures<stdexec::set_value_t()>{};
    }

    template <stdexec::receiver Rcvr>
        constexpr auto connect(Rcvr rcvr)
        && KOKKOS_EXECUTION_RETURNS_BRACED(CacheOpState<Rcvr>, std::move(rcvr), std::move(env))

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> const Scheduler<Exec>::Sender::Attributes& {
        return env;
    }

    Scheduler<Exec>::Sender::Attributes env;
};

//! Specialization for @ref Kokkos::Execution::GraphImpl::CacheSender::CacheOpState. @todo To be removed and done properly.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::CacheSender<Exec>::template CacheOpState>)
struct GraphOperationStateFor<OpState, Exec> : public std::true_type { };

//! Specialization for @ref Kokkos::Execution::GraphImpl::CacheSender::CacheOpState.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    graph_operation_state_for<OpState, Exec>
    && stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::CacheSender<Exec>::template CacheOpState>)
struct RemainsOnGraphFor<OpState, Exec> : public std::true_type { };

//! Construct a proper @c stdexec::when_all from closures.
struct make_when_all_fn {
    template <typename CacheSndr, typename... Closures>
    [[nodiscard]]
    constexpr auto operator()(CacheSndr cache_sndr, Closures&&... closures) const {
        return stdexec::when_all(std::forward<Closures>(closures)(cache_sndr)...);
    }
};

//! Result of applying @ref make_when_all_fn to a pack of closures.
template <Kokkos::ExecutionSpace Exec, typename PackedClosures>
using get_when_all_sndr_t = stdexec::__apply_result_t<make_when_all_fn, PackedClosures, CacheSender<Exec>>;

//! Operation state for @c exec::fork_join.
template <Kokkos::ExecutionSpace Exec, stdexec::sender Sndr, typename PackedClosures, stdexec::receiver Rcvr>
struct ForkJoinOpState
    : public Impl::Immovable
    , public OpStateBase<Exec, Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = OpStateBase<Exec, Rcvr>;
    using execution_space = Exec;

    using env_t = stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>;
    using when_all_sndr_t = get_when_all_sndr_t<Exec, PackedClosures>;
    using cache_sndr_t = CacheSender<Exec>;

    template <typename Tag, typename... Args>
    requires(!std::same_as<Tag, stdexec::set_value_t>)
    void complete(Tag, Args&&... args) & noexcept {
        base_t::complete(Tag{}, std::forward<Args>(args)...);
    }

    //! @name The @ref fork_opstate is the operation state of the sender before the fork point.
    ///@{
    using fork_rcvr_t = Impl::Receiver<ForkJoinOpState, env_t>;
    using fork_opstate_t = stdexec::connect_result_t<Sndr, fork_rcvr_t>;
    ///@}

    //! The root node type.
    using root_t = typename Kokkos::Experimental::Graph<Exec>::root_t;

    //! @name Check what the probe would do for composition.
    ///@{
    using probe_graph_composition_policy_t = GraphComposition::policy_t<fork_opstate_t, Rcvr>;
    using probe_node_t = GraphComposition::node_t<Exec, fork_opstate_t, Rcvr>;
    ///@}

    template <typename NodeType>
    struct ForkJoinReceiver {
        using receiver_concept = Impl::SubmittedReceiverTag;

        ForkJoinOpState* opstate;

        [[nodiscard]]
        constexpr auto query(get_node_t) const noexcept -> const NodeType& {
            return opstate->get_fork_node();
        }

        void set_value() && noexcept {
            opstate->submit_graph();
        }

        template <typename Error>
        void set_error(Error&& error) && noexcept {
            stdexec::set_error(std::move(opstate->completion_signal.rcvr), std::forward<Error>(error));
        }

        KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, *opstate)
    };

    /**
     * @name Probe if the resulting @c when_all operation state remains on a single graph.
     *
     * If @ref join_opstate doesn't remain on graph, it is still @c fork_join 's
     * responsibility to create the graph.
     *
     * Given @c probe_node_t:
     *  1. If @ref probe_join_opstate_t remains on graph, it is business as usual; either of the
     *     following happens:
     *      * @ref fork_opstate or the receiver is queryable for @ref get_node_t, such that the graph composition
     *        policy is @ref GraphComposition::Attach.
     *        The query result is passed to @ref join_opstate through @ref ForkJoinReceiver.
     *      * Neither is queryable for @ref get_node_t, such that the graph composition policy is
     *        @ref GraphComposition::Create.
     *        The root node is passed to @ref join_opstate through @ref ForkJoinReceiver.
     *  2. Otherwise, if @ref probe_join_opstate_t does not remain on graph,
     *     the graph composition policy is @ref GraphComposition::Create.
     *     The root node is passed to @ref join_opstate through @ref ForkJoinReceiver.
     *
     * @note If @ref probe_node_t is the same type as @ref fork_node_t, the probing has not created
     *       too much compilation pressure, since @ref join_opstate_t *is* the same type as
     *       @ref probe_join_opstate_t.
     */
    ///@{
    using probe_join_opstate_t = stdexec::connect_result_t<when_all_sndr_t, ForkJoinReceiver<probe_node_t>>;
    static constexpr bool join_opstate_as_one = probe_join_opstate_t::as_one;

    using fork_node_t = std::conditional_t<join_opstate_as_one, probe_node_t, root_t>;
    using join_opstate_t = stdexec::connect_result_t<when_all_sndr_t, ForkJoinReceiver<fork_node_t>>;

    using graph_composition_policy_t =
        std::conditional_t<join_opstate_as_one, probe_graph_composition_policy_t, GraphComposition::Create>;
    using state_t = State<graph_composition_policy_t, Exec>;
    ///@}

    Scheduler<Exec> schd;
    fork_opstate_t fork_opstate;
    [[no_unique_address]]
    state_t state;
    join_opstate_t join_opstate;

    ForkJoinOpState(
        Scheduler<Exec> schd_,
        Sndr&& sndr, // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        PackedClosures&& packed_closures,
        Rcvr rcvr) // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        : base_t(std::move(rcvr))
        , schd(std::move(schd_))
        , fork_opstate(stdexec::connect(std::forward<Sndr>(sndr), fork_rcvr_t{this}))
        , state(Kokkos::Experimental::get_device_handle(schd.state->exec))
        , join_opstate(
              stdexec::connect(
                  stdexec::__apply(
                      make_when_all_fn{},
                      std::move(packed_closures),
                      cache_sndr_t{.env = {.state = schd.state}}),
                  ForkJoinReceiver<fork_node_t>{this})) {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "The 'fork_join' operation state uses the graph composition policy "
                  << Impl::display_string_of<graph_composition_policy_t>();
#endif
    }

    [[nodiscard]]
    constexpr auto get_fork_node() const noexcept -> const fork_node_t& {
        if constexpr (std::same_as<graph_composition_policy_t, GraphComposition::Create>) {
            return state.get_root_node();
        } else if constexpr (stdexec::__queryable_with<fork_opstate_t, get_node_t>) {
            return fork_opstate.query(get_node);
        } else {
            return this->completion_signal.rcvr.query(get_node);
        }
    }

    const auto& query(get_node_t) const & noexcept {
        return join_opstate.query(get_node);
    }

    [[nodiscard]]
    constexpr auto query(Impl::get_exec_t) const noexcept -> Impl::ExecutionSpaceRef<Exec> {
        if constexpr (std::same_as<graph_composition_policy_t, GraphComposition::Create>) {
            return Impl::ExecutionSpaceRef<Exec>{state.get_device_handle().m_exec};
        } else {
            return Impl::get_exec(fork_opstate);
        }
    }

    void start() & noexcept {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "ForkJoinOpState::start";
#endif
        stdexec::start(fork_opstate);
    }

    void submit() & noexcept {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "ForkJoinOpState::submit";
#endif
        /// If @ref fork_opstate submitted operations, and @ref join_opstate does not remain on graph,
        /// the outstanding work must be synchronized.
        if constexpr (!join_opstate_t::as_one && Impl::signals_submitted<fork_opstate_t>) {
            try {
                Impl::get_exec(fork_opstate).get().fence("inner when_all does not remain on graph");
            } catch (...) {
                stdexec::set_error(std::move(this->completion_signal.rcvr), std::current_exception());
                return;
            }
        }
        stdexec::start(join_opstate);
    }

    void submit_graph() & noexcept {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "ForkJoinOpState::submit_graph";
#endif
        if constexpr (std::same_as<graph_composition_policy_t, GraphComposition::Create>) {
            try {
                Kokkos::Execution::GraphImpl::submit_graph(state.graph, state.get_device_handle().m_exec);
            } catch (...) {
                stdexec::set_error(std::move(this->completion_signal.rcvr), std::current_exception());
                return;
            }
            this->completion_signal.propagate(this->query(Impl::get_exec).get());
        } else {
            stdexec::set_value(std::move(this->completion_signal.rcvr));
        }
    }

    void complete(stdexec::set_value_t) noexcept {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_INFO << "ForkJoinOpState::complete";
#endif
        this->submit();
    }

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->completion_signal.rcvr)
};

//! Specialization for @ref Kokkos::Execution::GraphImpl::ForkJoinOpState. @todo To be removed and done properly.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::ForkJoinOpState>
    && std::same_as<typename OpState::execution_space, Exec>)
struct GraphOperationStateFor<OpState, Exec> : public std::true_type { };

//! Specialization for @ref Kokkos::Execution::GraphImpl::ForkJoinOpState.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    graph_operation_state_for<OpState, Exec>
    && stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::ForkJoinOpState>)
struct RemainsOnGraphFor<OpState, Exec> : public std::true_type { };

//! Sender for @c exec::fork_join.
template <Kokkos::ExecutionSpace Exec, typename Sndr, typename PackedClosures>
struct ForkJoinSender {
    using sender_concept = stdexec::sender_t;

    template <typename Self, typename... Env>
    static consteval auto get_completion_signatures() {
        return stdexec::completion_signatures_of_t<
            get_when_all_sndr_t<Exec, PackedClosures>,
            stdexec::__fwd_env_t<Env>...
        >{};
    }

    template <stdexec::receiver Rcvr>
        auto connect(Rcvr rcvr)
        && KOKKOS_EXECUTION_RETURNS_BRACED(
            (ForkJoinOpState<Exec, Sndr, PackedClosures, Rcvr>),
            std::move(schd),
            std::forward<Sndr>(sndr),
            std::move(packed_closures),
            std::move(rcvr))

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Scheduler<Exec> schd;
    Sndr sndr;
    PackedClosures packed_closures;
};

template <>
struct TransformSenderFor<exec::fork_join_t> {
    //! @todo Proper @c noexcept expression.
    template <typename Env, typename PackedClosures, typename Sndr>
    requires(stdexec::__sends<stdexec::set_value_t, Sndr, Env> && graph_completing_sender<Sndr, Env>)
    auto operator()(const Env& env, exec::fork_join_t, PackedClosures&& closures, Sndr&& sndr) const {
        auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

        return ForkJoinSender<Impl::exec_of_t<Sndr, Env>, Sndr, PackedClosures>{
            .schd = std::move(schd),
            .sndr = std::forward<Sndr>(sndr),
            .packed_closures = std::forward<PackedClosures>(closures)};
    }

    template <typename Env, typename PackedClosures, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env&, exec::fork_join_t, PackedClosures&&, Sndr&&) const noexcept {
        return no_graph_scheduler_in_env<exec::fork_join_t, Sndr, Env>();
    }
};

} // namespace Kokkos::Execution::GraphImpl

#endif // KOKKOS_EXECUTION_GRAPH_FORK_JOIN_HPP
