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

    struct Sender
    {
        using sender_concept = stdexec::sender_t;

        auto& get_env() const noexcept { return env; }

        Env env;
    };

    template <typename T>
    explicit ExecutionSpaceScheduler(T&& exec) : env{std::forward<T>(exec)} {}

    ::stdexec::sender auto schedule() const noexcept { return Sender{.env = env}; }

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
