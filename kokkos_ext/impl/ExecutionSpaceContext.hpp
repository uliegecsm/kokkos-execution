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

    struct Domain
    {
        /**
         * @brief Customize @c sync_wait.
         *
         * References:
         *  - https://github.com/NVIDIA/stdexec/blob/e8a6a7b25fbc2463e1dfe0ee20973b1fe622bfcf/include/nvexec/stream_context.cuh#L247-L251
         *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait
         *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-dispatch
         *
         * @todo Make the @c noexcept specifier depend on the completion signatures of @p sndr.
         */
        template <execution_space_completing_sender Sndr>
        auto apply_sender(stdexec::sync_wait_t, Sndr&& sndr) const noexcept(false)
        {
            auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), stdexec::env{});
            return SyncWait{}(std::move(schd), std::forward<Sndr>(sndr));
        }

        //! For customization of @c then or @c bulk or @c continues_on.
        template <stdexec::sender Sndr, typename Env> requires (
            (
                std::same_as<stdexec::tag_of_t<Sndr>, stdexec::then_t>
             || std::same_as<stdexec::tag_of_t<Sndr>, stdexec::bulk_t>
             || std::same_as<stdexec::tag_of_t<Sndr>, stdexec::continues_on_t>
            )
            && execution_space_completing_sender<Sndr, Env>)
        static auto transform_sender(::stdexec::set_value_t, Sndr&& sndr, const Env& env_) noexcept
        {
            return sndr.apply(
                std::forward<Sndr>(sndr),
                transform_sender_for<stdexec::tag_of_t<Sndr>, Env>{.env_ = env_}
            );
        }

        //! For customization of @c schedule_from.
        template <stdexec::sender_expr_for<stdexec::schedule_from_t> Sndr, typename Env>
            requires execution_space_completing_sender<Sndr, Env>
        auto transform_sender(stdexec::set_value_t, Sndr&& sndr, const Env& env_) const noexcept
        {
            auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);

            auto [tag, from, inner] = std::forward<Sndr>(sndr);

            const bool skip = [&](){
                if constexpr (std::same_as<std::remove_cvref_t<Env>, ExecutionSpaceSchedulerEnv<Exec>>) {
                    return schd.env.exec == env_.exec;
                }
                return false;
            }();
            return ScheduleFromSender{
                .env = std::move(schd.env),
                .sndr = std::move(inner),
                .skip = skip
            };
        }
    };

    auto query(stdexec::get_domain_t) const noexcept { return Domain{}; }

    [[nodiscard]] constexpr auto query(stdexec::get_completion_domain_t<::stdexec::set_value_t>) const noexcept -> Domain {
        return {};
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
