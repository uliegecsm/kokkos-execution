#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_HPP

#include "kokkos-execution/stdexec.hpp"

#include "Kokkos_Core.hpp"

#include "kokkos-execution/execution_space/bulk.hpp"
#include "kokkos-execution/execution_space/continues_on.hpp"
#include "kokkos-execution/execution_space/domain.hpp"
#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/execution_space/parallel_for.hpp"
#include "kokkos-execution/execution_space/schedule_from.hpp"
#include "kokkos-execution/execution_space/scoped_region.hpp"
#include "kokkos-execution/execution_space/sync_wait.hpp"
#include "kokkos-execution/execution_space/then.hpp"
#include "kokkos-execution/execution_space/when_all.hpp"
#include "kokkos-execution/impl/forward_progress_guarantee.hpp"
#include "kokkos-execution/impl/state.hpp"

namespace Kokkos::Execution {

namespace ExecutionSpaceImpl {

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
        using operation_state_concept = stdexec::operation_state_tag;

        Rcvr rcvr;

        //! @todo Check signature. And check whether we should move the receiver.
        void start() & noexcept {
            stdexec::set_value(std::move(rcvr));
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
            return {std::move(rcvr)};
        }

        template <stdexec::receiver_of<completion_signatures> Rcvr>
        [[nodiscard]]
        OpState<Rcvr> connect(Rcvr rcvr) const & noexcept(std::is_nothrow_move_constructible_v<Rcvr>) {
            return {std::move(rcvr)};
        }

        [[nodiscard]]
        constexpr auto get_env() const noexcept -> const Attributes& {
            return env;
        }

        Attributes env;
    };

    [[nodiscard]]
    constexpr auto
        query(stdexec::get_forward_progress_guarantee_t) const noexcept -> stdexec::forward_progress_guarantee {
        return Impl::forward_progress_guarantee_of_v<Exec>;
    }

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

} // namespace ExecutionSpaceImpl

/**
 * @brief Execution context using a @c Kokkos execution space under the hood.
 *
 * For instance, if @p Exec is @c Kokkos::Cuda, the following holds true:
 *  1. The execution context will be the @c Cuda stream stored by the @c Kokkos::Cuda instance in @ref Kokkos::Execution::Impl::State.
 *  2. The execution resource is the GPU the stream is attached to.
 */
template <Kokkos::ExecutionSpace Exec>
struct ExecutionSpaceContext {
    using state_t = Impl::State<Exec>;

    state_t m_state;

    explicit ExecutionSpaceContext(Exec exec) // NOLINT(performance-unnecessary-value-param)
        : m_state{std::move(exec)} {
    }

    auto get_scheduler() const noexcept -> ExecutionSpaceImpl::Scheduler<Exec> {
        return {const_cast<state_t*>(&m_state)};
    }
};

} // namespace Kokkos::Execution

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_HPP
