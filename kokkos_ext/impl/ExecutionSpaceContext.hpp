#ifndef GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP
#define GRAPH_DISPATCHING_KOKKOS_EXT_IMPL_EXECUTIONSPACECONTEXT_HPP

#include <concepts>

#include "tests/IgnoreWarnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wunused-parameter")
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#include "kokkos_ext/impl/execution_space/sync_wait.hpp"

namespace Kokkos::Experimental
{

namespace details::execution_space
{
//! Scheduler for a @c Kokkos execution space.
template <typename Exec> requires Kokkos::is_execution_space_v<Exec>
struct ExecutionSpaceScheduler
{
    //! See https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L168-L195).
    struct Env
    {
        //! The accepted completion tag types must agree with @c Sender::completion_signatures.
        template <::stdexec::__one_of<::stdexec::set_value_t> CompletionTag>
        auto query(stdexec::get_completion_scheduler_t<CompletionTag>) const noexcept { return ExecutionSpaceScheduler{exec}; }

        bool operator==(const Env&) const noexcept = default;

        Exec exec;
    };

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

        Env env;
    };

    template <typename T>
    explicit ExecutionSpaceScheduler(T&& exec) : env{std::forward<T>(exec)} {}

    ::stdexec::sender auto schedule() const noexcept { return Sender{.env = env}; }

    /**
     * @name Customization points.
     *
     * Sender algorithms are customizable. We follow the approach developped in
     * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-customization.
     *
     * See also:
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/exec/static_thread_pool.hpp#L236-L245
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/exec/static_thread_pool.hpp#L264-L278
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/exec/static_thread_pool.hpp#L427-L429
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/stdexec/__detail/__sender_introspection.hpp#L23-L31
     */
    ///@{
    struct Domain
    {
        /**
         * @brief Customize @c sync_wait.
         *
         * References:
         *  - https://github.com/NVIDIA/stdexec/blob/e8a6a7b25fbc2463e1dfe0ee20973b1fe622bfcf/include/nvexec/stream_context.cuh#L247-L251
         *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#spec-execution.senders.consumers.sync_wait
         *  - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-dispatch
         */
        template <stdexec::sender Sndr>
        auto apply_sender(stdexec::sync_wait_t, Sndr&& sndr) const noexcept
        {
            if constexpr (::stdexec::__completes_on<Sndr, ExecutionSpaceScheduler>) {
                auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr));
                return SyncWait{}(std::move(schd), std::forward<Sndr>(sndr));
            } else {
                static_assert(false, "No 'ExecutionSpaceScheduler' can be found on which to sync wait.");
            }
        }
    };

    auto query(stdexec::get_domain_t) const noexcept { return Domain{}; }
    ///@}

    bool operator==(const ExecutionSpaceScheduler&) const noexcept = default;

    Env env;
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
