#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
#include "plog/Log.h"
#endif

#include "kokkos_ext/impl/execution_space/bulk.hpp"
#include "kokkos_ext/impl/execution_space/continues_on.hpp"
#include "kokkos_ext/impl/execution_space/get_exec.hpp"
#include "kokkos_ext/impl/execution_space/schedule_from.hpp"
#include "kokkos_ext/impl/execution_space/sync_wait.hpp"
#include "kokkos_ext/impl/execution_space/then.hpp"

namespace Kokkos::Experimental
{

namespace details::execution_space
{

struct Domain : public stdexec::default_domain
{
    template <typename Tag, stdexec::sender Sndr, typename... Args>
        requires stdexec::__callable<apply_sender_for<Tag>, Sndr, Args...>
    static auto apply_sender(Tag, Sndr&& sndr, Args&&... args) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": apply_sender for tag " << Kokkos::Impl::TypeInfo<Tag>::name();
#endif
        return apply_sender_for<Tag>{}(std::forward<Sndr>(sndr), std::forward<Args>(args)...);
    }

    template <stdexec::sender Sndr, typename Env>
        requires stdexec::__applicable<
            transform_sender_for<stdexec::tag_of_t<Sndr>, Env>,
            Sndr
        >
    static auto transform_sender(stdexec::set_value_t, Sndr&& sndr, const Env& env_) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": transform_sender for tag " << Kokkos::Impl::TypeInfo<stdexec::tag_of_t<Sndr>>::name();
#endif
        return stdexec::__apply(
            transform_sender_for<stdexec::tag_of_t<Sndr>, Env>{.env_ = env_},
            std::forward<Sndr>(sndr)
        );
    }
};

template <Kokkos::ExecutionSpace Exec>
struct State {
    Exec exec;
};

//! See https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L168-L195).
template <Kokkos::ExecutionSpace Exec>
struct SchedulerEnv
{
    using execution_space = Exec;

    [[nodiscard]] constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> Scheduler<Exec> {
        return {state};
    }

    [[nodiscard]] constexpr auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    [[nodiscard]] constexpr auto query(get_exec_t) const noexcept -> const Exec& {
        return state->exec;
    }

    State<Exec>* state;
};

/**
 * @brief Scheduler for a @c Kokkos execution space.
 *
 * Note that storing a @c Kokkos execution space instance and moving it around
 * generally implies a shared pointer copy, see https://github.com/kokkos/kokkos/pull/8807.
 */
template <Kokkos::ExecutionSpace Exec>
struct Scheduler
{
    //! As per https://eel.is/c++draft/exec.sched#1.
    using scheduler_concept = stdexec::scheduler_t;

    using execution_space = Exec;

    template <stdexec::receiver Rcvr>
    struct OpState
    {
        using operation_state_concept = stdexec::operation_state_t;

        Rcvr rcvr;

        //! @todo Check signature. And check whether we should move the receiver.
        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
        }
    };

    struct Sender
    {
        using sender_concept = stdexec::sender_t;

        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        [[nodiscard]] OpState<std::remove_cvref_t<Rcvr>> connect(Rcvr&& rcvr) noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<Rcvr>, Rcvr&&>) {
            return {std::forward<Rcvr>(rcvr)};
        }

        [[nodiscard]] constexpr auto get_env() const noexcept -> const SchedulerEnv<Exec>& { return env; }

        SchedulerEnv<Exec> env;
    };

    [[nodiscard]] Sender schedule() const noexcept { return {state}; }

    [[nodiscard]] constexpr auto
    query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    [[nodiscard]] constexpr auto
    query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> Scheduler {
        return {state};
    }

    [[nodiscard]] friend bool operator==(const Scheduler&, const Scheduler&) noexcept = default;

    State<Exec>* state;
};

} // namespace details::execution_space

/**
 * @brief Execution context using a @c Kokkos execution space under the hood.
 *
 * For instance, if @p Exec is @c Kokkos::Cuda, the following holds true:
 *  1. The execution context will be the @c Cuda stream stored by the @c Kokkos::Cuda instance in @ref Kokkos::Experimental::details::execution_space::State.
 *  2. The execution resource is the GPU the stream is attached to.
 */
template <Kokkos::ExecutionSpace Exec>
struct ExecutionSpaceContext
{
    using state_t = details::execution_space::State<Exec>;

    state_t m_state;

    explicit ExecutionSpaceContext(Exec exec) // NOLINT(performance-unnecessary-value-param)
        : m_state{std::move(exec)} {
    }

    auto get_scheduler() const noexcept -> details::execution_space::Scheduler<Exec> {
        return {const_cast<state_t*>(&m_state)};
    }
};

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
