#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP

#include <concepts>

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

#include "kokkos_ext/impl/ExecutionSpaceContext_fwd.hpp"

#include "kokkos_ext/impl/execution_space/bulk.hpp"
#include "kokkos_ext/impl/execution_space/continues_on.hpp"
#include "kokkos_ext/impl/execution_space/schedule_from.hpp"
#include "kokkos_ext/impl/execution_space/sync_wait.hpp"
#include "kokkos_ext/impl/execution_space/then.hpp"

namespace Kokkos::Experimental
{

namespace details::execution_space
{

//! See https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L168-L195).
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceSchedulerEnv
{
    [[nodiscard]] constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept {
        return ExecutionSpaceScheduler{exec};
    }

    bool operator==(const ExecutionSpaceSchedulerEnv&) const noexcept = default;

    Exec exec;
};

struct Domain : public stdexec::default_domain
{
    template <typename Tag, ::stdexec::sender Sndr, typename... Args>
        requires stdexec::__callable<apply_sender_for<Tag>, Sndr, Args...>
    static auto apply_sender(Tag, Sndr&& sndr, Args&&... args) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": apply_sender for tag " << Kokkos::Impl::TypeInfo<Tag>::name();
#endif
        return apply_sender_for<Tag>{}(std::forward<Sndr>(sndr), std::forward<Args>(args)...);
    }

    template <stdexec::sender Sndr, typename Env>
        requires stdexec::__callable<
            stdexec::__sexpr_apply_t,
            Sndr,
            transform_sender_for<stdexec::tag_of_t<Sndr>, Env>
        >
    static auto transform_sender(::stdexec::set_value_t, Sndr&& sndr, const Env& env_) {
#if defined(GRAPH_DISPATCHING_KOKKOS_EXT_DEBUG)
        PLOG_DEBUG << Kokkos::Impl::TypeInfo<Domain>::name() << ": transform_sender for tag " << Kokkos::Impl::TypeInfo<stdexec::tag_of_t<Sndr>>::name();
#endif
        return stdexec::__sexpr_apply(
            std::forward<Sndr>(sndr),
            transform_sender_for<stdexec::tag_of_t<Sndr>, Env>{.env_ = env_}
        );
    }
};

//! Scheduler for a @c Kokkos execution space.
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceScheduler
{
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

        using completion_signatures = ::stdexec::completion_signatures<::stdexec::set_value_t()>;

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        OpState<std::remove_cvref_t<Rcvr>> connect(Rcvr&& rcvr) noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<Rcvr>, Rcvr&&>) {
            return {std::forward<Rcvr>(rcvr)};
        }

        auto& get_env() const noexcept { return env; }

        ExecutionSpaceSchedulerEnv<Exec> env;
    };

    template <typename T>
    explicit ExecutionSpaceScheduler(T&& exec) : env{std::forward<T>(exec)} {}

    ::stdexec::sender auto schedule() const noexcept { return Sender{.env = env}; }

    auto query(stdexec::get_domain_t) const noexcept { return Domain{}; }

    [[nodiscard]] constexpr auto
    query(stdexec::get_completion_domain_t<::stdexec::set_value_t>) const noexcept -> Domain {
        return {};
    }

    [[nodiscard]] constexpr auto
    query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> ExecutionSpaceScheduler {
        return ExecutionSpaceScheduler{env};
    }

    bool operator==(const ExecutionSpaceScheduler&) const noexcept = default;

    ExecutionSpaceSchedulerEnv<Exec> env;
};

//! Deduction guide for @ref ExecutionSpaceScheduler.
template <typename Exec>
ExecutionSpaceScheduler(Exec&&) -> ExecutionSpaceScheduler<std::remove_cvref_t<Exec>>;

} // namespace details::execution_space

/**
 * @brief Execution context using a @c Kokkos execution space under the hood.
 *
 * For instance, if @p Exec is @c Kokkos::Cuda, the following holds true:
 *  1. The execution context will be the @c Cuda stream stored by the @c Kokkos::Cuda instance @ref exec.
 *  2. The execution resource is the GPU the stream is attached to.
 */
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceContext
{
    Exec exec;

    auto get_scheduler() const noexcept { return details::execution_space::ExecutionSpaceScheduler{exec}; }
};

} // namespace Kokkos::Experimental

#endif // GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
