#ifndef KOKKOS_EXECUTION_GRAPH_HPP
#define KOKKOS_EXECUTION_GRAPH_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/graph/bulk.hpp"
#include "kokkos-execution/graph/continues_on.hpp"
#include "kokkos-execution/graph/domain.hpp"
#include "kokkos-execution/graph/get_node.hpp"
#include "kokkos-execution/graph/parallel_for.hpp"
#include "kokkos-execution/graph/schedule_from.hpp"
#include "kokkos-execution/graph/sync_wait.hpp"
#include "kokkos-execution/graph/then.hpp"
#include "kokkos-execution/graph/when_all.hpp"
#include "kokkos-execution/impl/state.hpp"

namespace Kokkos::Execution {

namespace GraphImpl {

//!  Scheduler for a @c Kokkos::Experimental::Graph.
template <Kokkos::ExecutionSpace Exec>
struct Scheduler {
    //! As per https://eel.is/c++draft/exec.sched#1.
    using scheduler_concept = stdexec::scheduler_t;

    using execution_space = Exec;

    template <stdexec::receiver Rcvr>
    struct OpState {
        using operation_state_concept = stdexec::operation_state_tag;

        Impl::State<execution_space>* state;
        Rcvr rcvr;

        [[nodiscard]]
        constexpr auto query(Impl::get_exec_t) const noexcept -> Impl::ExecutionSpaceRef<execution_space> {
            return Impl::ExecutionSpaceRef{state->exec};
        }

        //! @todo Check signature. And check whether we should move the receiver.
        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
        }

        [[nodiscard]]
        auto query(get_node_t) const & noexcept -> const
            typename Kokkos::Experimental::Graph<Exec>::root_t& requires stdexec::__queryable_with<Rcvr, get_node_t>
        {
            return this->rcvr.query(get_node);
        }
    };

    struct Sender {
        using sender_concept = stdexec::sender_tag;

        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

        //! See https://github.com/NVIDIA/stdexec/blob/5076be2b35de2e78330201b888d82c81b8cb428b/include/nvexec/stream_context.cuh#L110.
        struct Attributes {
            template <typename... Env>
            [[nodiscard]]
            constexpr auto
                query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>, Env...) const noexcept -> Scheduler {
                return {state};
            }

            template <typename... Env>
            [[nodiscard]]
            constexpr auto
                query(stdexec::get_completion_domain_t<stdexec::set_value_t>, Env...) const noexcept -> Domain {
                return {};
            }

            Impl::State<Exec>* state;
        };

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        [[nodiscard]]
        OpState<Rcvr> connect(Rcvr rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
            return {.state = env.state, .rcvr = std::move(rcvr)};
        }

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        [[nodiscard]]
        OpState<Rcvr> connect(Rcvr rcvr) const & noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
            return {.state = env.state, .rcvr = std::move(rcvr)};
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> const Attributes& {
            return env;
        }

        Attributes env;
    };

    [[nodiscard]]
    constexpr auto schedule() const noexcept -> Sender {
        return {state};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> Scheduler {
        return {state};
    }

    [[nodiscard]]
    friend bool operator==(const Scheduler&, const Scheduler&) noexcept = default;

    Impl::State<Exec>* state;
};

} // namespace GraphImpl

//! Execution context using @c Kokkos::Experimental::Graph under the hood.
template <Kokkos::ExecutionSpace Exec>
struct GraphContext {
    using state_t = Impl::State<Exec>;

    state_t m_state;

    explicit GraphContext(Exec exec) // NOLINT(performance-unnecessary-value-param)
        : m_state{std::move(exec)} {
    }

    auto get_scheduler() const noexcept -> GraphImpl::Scheduler<Exec> {
        return {const_cast<state_t*>(&m_state)};
    }
};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_GRAPH_HPP
