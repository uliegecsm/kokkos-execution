#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_HPP

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
PRAGMA_DIAGNOSTIC_IGNORED("-Wdeprecated-copy")
PRAGMA_DIAGNOSTIC_IGNORED("-Wempty-body")
PRAGMA_DIAGNOSTIC_IGNORED("-Wshadow")
PRAGMA_DIAGNOSTIC_IGNORED("-Wsuggest-override")
PRAGMA_DIAGNOSTIC_IGNORED("-Wswitch-default")
#include <stdexec/execution.hpp>
PRAGMA_DIAGNOSTIC_POP

#include "Kokkos_Core.hpp"

#if defined(KOKKOS_EXECUTION_DEBUG)
#    include "plog/Log.h"
#endif

#include "kokkos-execution/execution_space/bulk.hpp"
#include "kokkos-execution/execution_space/continues_on.hpp"
#include "kokkos-execution/execution_space/domain.hpp"
#include "kokkos-execution/execution_space/get_exec.hpp"
#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/schedule_from.hpp"
#include "kokkos-execution/execution_space/sync_wait.hpp"
#include "kokkos-execution/execution_space/then.hpp"

namespace Kokkos::Execution {

namespace ExecutionSpaceImpl {

template <Kokkos::ExecutionSpace Exec>
struct State {
    Exec exec;
};

//! See https://github.com/NVIDIA/stdexec/blob/9514e7bdf4b5d16d8ee4b5ad0e9c8733c3539f37/include/nvexec/stream/common.cuh#L168-L195).
template <Kokkos::ExecutionSpace Exec>
struct SchedulerEnv {
    using execution_space = Exec;

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_scheduler_t<stdexec::set_value_t>) const noexcept -> Scheduler<Exec> {
        return {state};
    }

    [[nodiscard]]
    constexpr auto query(stdexec::get_completion_domain_t<stdexec::set_value_t>) const noexcept -> Domain {
        return {};
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
struct Scheduler {
    //! As per https://eel.is/c++draft/exec.sched#1.
    using scheduler_concept = stdexec::scheduler_t;

    using execution_space = Exec;

    template <stdexec::receiver Rcvr>
    struct OpState {
        using operation_state_concept = stdexec::operation_state_t;

        Rcvr rcvr;

        //! @todo Check signature. And check whether we should move the receiver.
        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
        }
    };

    struct Sender {
        using sender_concept = stdexec::sender_t;

        using completion_signatures = stdexec::completion_signatures<stdexec::set_value_t()>;

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        [[nodiscard]]
        OpState<Rcvr> connect(Rcvr rcvr) noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
            return {std::move(rcvr)};
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> const SchedulerEnv<Exec>& {
            return env;
        }

        SchedulerEnv<Exec> env;
    };

    [[nodiscard]]
    Sender schedule() const noexcept {
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

    State<Exec>* state;
};

} // namespace ExecutionSpaceImpl

/**
 * @brief Execution context using a @c Kokkos execution space under the hood.
 *
 * For instance, if @p Exec is @c Kokkos::Cuda, the following holds true:
 *  1. The execution context will be the @c Cuda stream stored by the @c Kokkos::Cuda instance in @ref Kokkos::Execution::ExecutionSpaceImpl::State.
 *  2. The execution resource is the GPU the stream is attached to.
 */
template <Kokkos::ExecutionSpace Exec>
struct ExecutionSpaceContext {
    using state_t = State<Exec>;

    state_t m_state;

    explicit ExecutionSpaceContext(Exec exec) // NOLINT(performance-unnecessary-value-param)
        : m_state{std::move(exec)} {
    }

    auto get_scheduler() const noexcept -> Scheduler<Exec> {
        return {const_cast<state_t*>(&m_state)};
    }
};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_HPP
