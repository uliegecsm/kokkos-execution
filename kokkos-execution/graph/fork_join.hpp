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

#include "Kokkos_Graph.hpp"

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
 * @brief Sender to be added to each of the @c exec::fork_join branch.
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

        Rcvr rcvr;
    };

    template <typename Self, typename... Env>
    static consteval auto get_completion_signatures() {
        return stdexec::completion_signatures<stdexec::set_value_t()>{};
    }

    template <stdexec::receiver Rcvr>
    constexpr auto connect(Rcvr rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) -> CacheOpState<Rcvr> {
        return {std::move(rcvr)};
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> const Scheduler<Exec>::Sender::Attributes& {
        return env;
    }

    Scheduler<Exec>::Sender::Attributes env;
};

//! Specialization for @ref Kokkos::Execution::GraphImpl::CacheSender::CacheOpState.
template <stdexec::operation_state OpState, Kokkos::ExecutionSpace Exec>
requires(
    graph_operation_state_for<OpState, Exec>
    && stdexec::__is_instance_of<OpState, Kokkos::Execution::GraphImpl::CacheSender<Exec>::template CacheOpState>)
struct RemainsOnGraphFor<OpState, Exec> : public std::true_type {
    static constexpr void diagnose() noexcept {
    }
};

//! Transform the closures to a proper @c stdexec::when_all.
struct make_when_all_fn {
    template <typename CacheSndr, typename... Closures>
    constexpr auto operator()(CacheSndr cache_sndr, Closures&&... closures) const {
        return stdexec::when_all(std::forward<Closures>(closures)(cache_sndr)...);
    }
};

template <Kokkos::ExecutionSpace Exec, typename PackedClosures>
using get_when_all_sndr_t = stdexec::__apply_result_t<make_when_all_fn, PackedClosures, CacheSender<Exec>>;

//! Operation state for @c exec::fork_join.
template <Kokkos::ExecutionSpace Exec, stdexec::sender Sndr, typename PackedClosures, stdexec::receiver Rcvr>
struct ForkJoinOpState
    : public Impl::Immovable
    , public OpStateBase<Exec, Rcvr> {
    using operation_state_concept = Impl::SubmittedOperationStateTag;

    using base_t = OpStateBase<Exec, Rcvr>;

    using env_t = stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>;
    using domain_t = stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, env_t>;
    using fork_completions_t = stdexec::completion_signatures_of_t<Sndr, env_t>;
    using when_all_sndr_t = get_when_all_sndr_t<Exec, PackedClosures>;
    using cache_sndr_t = CacheSender<Exec>;

    /**
     * The fork join node will be passed to the branches through the @c stdexec::when_all receiver environment.
     *
     * The following situation generates a single @c Kokkos graph.
     * @verbatim
     *                               /-> (then) ->\
     * (graph) -> (then) -> (fork) ->              -> (join) -> (then)
     *                               \-> (then) ->/
     * @endverbatim
     *
     * However, dealing with the following situation is less straightforward.
     * @verbatim
     *                               /-> (then) -> (thread) -> (then) ->\
     * (graph) -> (then) -> (fork) ->                                    -> (join) -> (then)
     *                               \-> (then)                       ->/
     * @endverbatim
     *
     * In the future, it could be accepted as a valid sender, and there would be four @c Kokkos graphs:
     * one before the fork, one per branch, and one after the fork.
     *
     * For now, it is not accepted.
     */
    // static_assert(Impl::remains_on<stdexec::set_value_t, when_all_sndr_t, Domain>);
    // static_assert(when_all_sndr_t::as_one);

    //! @name The @ref fork_opstate is the operation state of the sender before the fork point.
    ///@{
    using fork_rcvr_t = Impl::Receiver<ForkJoinOpState, env_t>;
    using fork_opstate_t = stdexec::connect_result_t<Sndr, fork_rcvr_t>;
    ///@}

    /**
     * @name A node is not necessarily created by the @ref fork_opstate, for instance if it's a schedule sender.
     *       If the @ref fork_opstate is queryable for @ref get_node_t, it will be used in @ref get_fork_node.
     *       Otherwise, the @ref get_fork_node serves the root node from @ref state.
     */
    ///@{
    using graph_composition_policy_t = GraphComposition::policy_t<fork_opstate_t>;
    using state_t = State<graph_composition_policy_t, Exec>;
    using fork_node_t = GraphComposition::node_t<Exec, fork_opstate_t>;
    ///@}

    struct ForkJoinReceiver {
        using receiver_concept = Impl::SubmittedReceiverTag;

        ForkJoinOpState* opstate;

        auto query(get_node_t) const noexcept -> const fork_node_t& {
            return opstate->get_fork_node();
        }

        void set_value() && noexcept {
            stdexec::set_value(std::move(opstate->completion_signal.rcvr));
        }

        template <typename Error>
        void set_error(Error&& error) && noexcept {
            stdexec::set_error(std::move(opstate->completion_signal.rcvr), std::forward<Error>(error));
        }

        void submitted() && noexcept {
            std::move(opstate->completion_signal.rcvr).submitted();
        }

        KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, *opstate)
    };

    using join_opstate_t = stdexec::connect_result_t<when_all_sndr_t, ForkJoinReceiver>;

    static_assert(diagnose_remains_on_graph_for<join_opstate_t, Exec>());

    Scheduler<Exec> schd;
    fork_opstate_t fork_opstate;
    state_t state;
    join_opstate_t join_opstate;

    ForkJoinOpState(Scheduler<Exec> schd_, Sndr&& sndr, PackedClosures&& packed_closures, Rcvr rcvr)
        : base_t(std::move(rcvr))
        , schd(std::move(schd_))
        , fork_opstate(stdexec::connect(std::forward<Sndr>(sndr), fork_rcvr_t{this}))
        , state(Kokkos::Experimental::get_device_handle(schd_.state->exec))
        , join_opstate(
              stdexec::connect(
                  stdexec::__apply(
                      make_when_all_fn{},
                      std::move(packed_closures),
                      cache_sndr_t{.env = {.state = schd_.state}}),
                  ForkJoinReceiver{this})) {
        // #if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        //         PLOG_INFO << "The 'fork_join' operation state uses the graph composition policy "
        //                   << Kokkos::Impl::TypeInfo<graph_composition_policy_t>::name() << ".\n"
        //                   << "The 'fork_opstate' member is of type " << Kokkos::Impl::TypeInfo<fork_opstate_t>::name()
        //                   << " and the 'join_opstate' is of type " << Kokkos::Impl::TypeInfo<join_opstate_t>::name() << '.';
        // #endif
    }

    const fork_node_t& get_fork_node() const noexcept {
        if constexpr (std::same_as<graph_composition_policy_t, GraphComposition::Create>) {
            return state.get_root_node();
        } else {
            return fork_opstate.query(get_node);
        }
    }

    const auto& query(get_node_t) const & noexcept {
        return join_opstate.query(get_node);
    }

    void start() & noexcept {
        stdexec::start(fork_opstate);
    }

    void submit() & noexcept {
        if constexpr (std::same_as<graph_composition_policy_t, GraphComposition::Create>) {
            submit_graph(state.graph, state.get_device_handle().m_exec);
        }
        stdexec::start(join_opstate);
    }

    KOKKOS_EXECUTION_GET_ENV(Rcvr, this->completion_signal.rcvr)
};

//! Sender for @c exec::fork_join.
template <Kokkos::ExecutionSpace Exec, typename Sndr, typename PackedClosures>
struct ForkJoinSender {
    using sender_concept = stdexec::sender_t;

    template <typename Self, typename... Env>
    static consteval auto get_completion_signatures() {
        // using sndr_t = stdexec::__copy_cvref_t<Self, Sndr>;
        // using sndr_completions_t = stdexec::completion_signatures_of_t<sndr_t, stdexec::__fwd_env_t<Env>...>;
        using when_all_sndr_t = get_when_all_sndr_t<Exec, PackedClosures>;
        using compl_t = stdexec::completion_signatures_of_t<when_all_sndr_t, stdexec::__fwd_env_t<Env>...>;
        // static_assert(std::same_as<compl_t, stdexec::completion_signatures_of_t<Sndr, stdexec::__fwd_env_t<Env>...>>);
        // return stdexec::completion_signatures_of_t<Sndr, stdexec::__fwd_env_t<Env>...>{};
        return compl_t{};
    }

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr rcvr) && /*noexcept(std::is_nothrow_move_constructible_v<Rcvr>)*/ {
        return ForkJoinOpState<Exec, Sndr, PackedClosures, Rcvr>(
            std::move(schd), std::forward<Sndr>(sndr), std::move(packed_closures), std::move(rcvr));
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Scheduler<Exec> schd;
    Sndr sndr;
    PackedClosures packed_closures;
};

template <>
struct TransformSenderFor<exec::fork_join_t> {
    template <typename Env, typename PackedClosures, typename Sndr>
    requires stdexec::__sends<stdexec::set_value_t, Sndr, Env>
    auto operator()(const Env& env, exec::fork_join_t, PackedClosures&& closures, Sndr&& sndr) const /*noexcept(

    )*/
    {
        if constexpr (graph_completing_sender<Sndr, Env>) {
            auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env);

            return ForkJoinSender<Impl::exec_of_t<Sndr, Env>, Sndr, PackedClosures>{
                .schd = std::move(schd),
                .sndr = std::forward<Sndr>(sndr),
                .packed_closures = std::forward<PackedClosures>(closures)};
        } else {
            return no_graph_scheduler_in_env<exec::fork_join_t, Sndr, Env>();
        }
    }
};

} // namespace Kokkos::Execution::GraphImpl

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace stdexec::__detail {
template <typename Exec, typename Sndr, typename PackedClosures>
extern __mtype<Kokkos::Execution::GraphImpl::ForkJoinSender<Exec, __demangle_t<Sndr>, PackedClosures>>
    __demangle_v<Kokkos::Execution::GraphImpl::ForkJoinSender<Exec, Sndr, PackedClosures>>;
} // namespace stdexec::__detail
// NOLINTEND(bugprone-reserved-identifier)

#endif // KOKKOS_EXECUTION_GRAPH_FORK_JOIN_HPP
