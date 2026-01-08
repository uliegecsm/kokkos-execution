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

    /**
     * @name Customization points.
     *
     * Sender algorithms are customizable. We follow the approach developed in
     * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-customization.
     *
     * See also:
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/exec/static_thread_pool.hpp#L236-L245
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/exec/static_thread_pool.hpp#L264-L278
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/exec/static_thread_pool.hpp#L427-L429
     *  - https://github.com/NVIDIA/stdexec/blob/3302dda347491a6818861d418d37d930c7ef088e/include/stdexec/__detail/__sender_introspection.hpp#L23-L31
     */
    ///@{
    //! For @c then and @c bulk.
    struct TransformDispatch
    {
        ExecutionSpaceScheduler schd;

        template <typename Functor, stdexec::sender Sndr>
        auto operator()(stdexec::then_t, Functor&& functor, Sndr&& sndr) && noexcept {
            return ThenSender<Sndr, Functor, ExecutionSpaceScheduler>{
                .sndr    = std::forward<Sndr>(sndr),
                .functor = std::forward<Functor>(functor),
                .schd    = std::move(schd)
            };
        }

        template <typename Data, stdexec::sender Sndr>
        auto operator()(stdexec::bulk_t, Data&& data, Sndr&& sndr) && noexcept {
            auto [policy, shape, functor] = std::forward<Data>(data);
            return BulkSender<Sndr, decltype(policy), decltype(shape), decltype(functor), ExecutionSpaceScheduler>{
                .sndr    = std::forward<Sndr>(sndr),
                .policy  = std::move(policy),
                .shape   = std::move(shape),
                .functor = std::move(functor),
                .schd    = std::move(schd)
            };
        }
    };

    struct TransformContinuesOn
    {
        template <stdexec::scheduler Schd, ::stdexec::sender Sndr>
        auto operator()(stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) && noexcept {
            static_assert(std::same_as<Schd, ExecutionSpaceScheduler>);
            return ContinuesOnSender{.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
        }
    };

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
        template <stdexec::sender Sndr>
        auto apply_sender(stdexec::sync_wait_t, Sndr&& sndr) const noexcept(false)
        {
            constexpr bool completes_on = std::same_as<
                std::invoke_result_t<::stdexec::get_completion_scheduler_t<::stdexec::set_value_t>, ::stdexec::env_of_t<Sndr>, stdexec::env<>>,
                ExecutionSpaceScheduler
            >;
            if constexpr (completes_on) {
                auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), stdexec::env{});
                return SyncWait{}(std::move(schd), std::forward<Sndr>(sndr));
            } else {
                static_assert(false, "No 'ExecutionSpaceScheduler' can be found on which to sync wait.");
            }
        }

        //! For customization of @c then or @c bulk.
        template <stdexec::sender Sndr, typename Env> requires (std::same_as<stdexec::tag_of_t<Sndr>, stdexec::then_t> || std::same_as<stdexec::tag_of_t<Sndr>, stdexec::bulk_t>)
        auto transform_sender(::stdexec::set_value_t, Sndr&& sndr, const Env& env_) const noexcept
        {
            if constexpr (::stdexec::__completes_on<Sndr, ExecutionSpaceScheduler, Env>) {
                auto schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env_);
                return sndr.apply(std::forward<Sndr>(sndr), TransformDispatch{.schd = std::move(schd)});
            } else {
                static_assert(::stdexec::__completes_on<Sndr, ExecutionSpaceScheduler, Env>);
            }
        }

        //! For customization of @c continues_on.
        template <stdexec::sender_expr_for<stdexec::continues_on_t> Sndr, typename Env>
        auto transform_sender(stdexec::set_value_t, Sndr&& sndr, const Env&) const noexcept
        {
            if constexpr (::stdexec::__completes_on<Sndr, ExecutionSpaceScheduler, Env>) {
                return sndr.apply(std::forward<Sndr>(sndr), TransformContinuesOn{});
            } else {
                static_assert(::stdexec::__completes_on<Sndr, ExecutionSpaceScheduler, Env>);
            }
        }

        //! For customization of @c schedule_from.
        template <stdexec::sender_expr_for<stdexec::schedule_from_t> Sndr, typename Env>
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
    ///@}

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
