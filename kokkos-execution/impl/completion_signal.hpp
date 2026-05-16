#ifndef KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNAL_HPP
#define KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNAL_HPP

#include "kokkos-execution/stdexec.hpp"

#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
#    include "plog/Log.h"
#endif

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/impl/dispatch_label.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/event.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/impl/optional_storage.hpp"
#include "kokkos-execution/impl/schedulers.hpp"

namespace Kokkos::Execution::Impl {

/**
 * If the receiver environment can be queried for @ref Kokkos::Execution::Impl::get_exec_t
 * and if the successor enqueues work on the same execution space instance, the implementation is allowed
 * to skip the synchronization of asynchronous work.
 *
 * This situation may arise, for example, when the execution space scheduler is used in a @c stdexec::when_all branch.
 */
template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct RequiresSync {
    //! @todo Simplify by moving and using @ref Tests::Utils::are_same_instances.
    bool operator()(const Exec& exec, const Rcvr& rcvr) const noexcept
        requires(stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>)
    {
        if constexpr (
            std::same_as<
                std::remove_cvref_t<stdexec::__query_result_t<stdexec::env_of_t<Rcvr>, get_exec_t>>,
                ExecutionSpaceRef<Exec>
            >) {
            const auto& src = exec;
            const auto& dst = get_exec(stdexec::get_env(rcvr)).get();
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
            PLOG_DEBUG << "The synchronization happens if " << Kokkos::Tools::Experimental::device_id(src)
                       << " is not equal to " << Kokkos::Tools::Experimental::device_id(dst) << '.';
#endif
            return src != dst;
        }
        return true;
    }

    //! @todo This is compile-time only and should go away.
    constexpr bool operator()(const Exec&, const Rcvr&) const noexcept
        requires(!stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>)
    {
#if defined(KOKKOS_EXECUTION_ENABLE_DEBUG_LOGGING)
        PLOG_DEBUG << "Synchronization always required.";
#endif
        return true;
    }
};

template <typename Policy, Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionSignal;

struct SyncPolicyTag { };

/**
 * @brief Under a sync policy, a terminal completion is propagated: in-flight operations must complete.
 *
 * The receiver is invoked after a synchronization through:
 *   - a fence under @ref InlineFenceExec
 *   - waiting for an event under @ref ScheduleWaitEvent
 */
struct SyncPolicy {
    struct InlineFenceExec : SyncPolicyTag { };
    struct ScheduleWaitEvent : SyncPolicyTag { };
};

template <typename Policy>
concept sync_policy = std::derived_from<Policy, SyncPolicyTag>;

/**
 * @brief Fence the execution space instance to complete the operation and then invoke the receiver.
 *
 * Skip the fence if synchronization is not required.
 */
template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionSignal<SyncPolicy::InlineFenceExec, Exec, Rcvr> {
    static constexpr auto label = Impl::dispatch_label<Exec, ": after dispatch">();

    Rcvr rcvr;

    void propagate(const Exec& exec) & noexcept {
        if (!RequiresSync<Exec, Rcvr>{}(exec, rcvr)) {
            stdexec::set_value(std::move(rcvr));
        } else {
            try {
                exec.fence(std::string(label));
                stdexec::set_value(std::move(rcvr));
            } catch (...) {
                stdexec::set_error(std::move(rcvr), std::current_exception());
            }
        }
    }
};

template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct ScheduleWaitEventReceiver {
    using receiver_concept = stdexec::receiver_tag;

    CompletionSignal<SyncPolicy::ScheduleWaitEvent, Exec, Rcvr>* completion_signal;

    void set_value() && noexcept {
        try {
            Impl::wait(*completion_signal->event);
            stdexec::set_value(std::move(completion_signal->rcvr));
        } catch (...) {
            stdexec::set_error(std::move(completion_signal->rcvr), std::current_exception());
        }
    }
};

/**
 * @brief Create an event in the execution space instance, and schedule on the delegation scheduler a task that waits
 *        on this event and then invoke the receiver.
 *
 * Skip the scheduling of the task if synchronization is not required.
 */
template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionSignal<SyncPolicy::ScheduleWaitEvent, Exec, Rcvr> {
    using event_storage_t = Impl::event_storage_t<Exec>;
    using delegation_scheduler_t = Impl::delegation_scheduler_of_t<stdexec::env_of_t<Rcvr>>;
    using inner_opstate_t = stdexec::connect_result_t<
        stdexec::schedule_result_t<delegation_scheduler_t>,
        ScheduleWaitEventReceiver<Exec, Rcvr>
    >;

    Rcvr rcvr;
    event_storage_t event = std::nullopt;
    OptionalStorage<inner_opstate_t> inner_opstate{};

    void propagate(const Exec& exec) & noexcept {
        if (!RequiresSync<Exec, Rcvr>{}(exec, rcvr)) {
            stdexec::set_value(std::move(rcvr));
        } else {
            try {
                event.emplace();
                record(*event, exec);
                inner_opstate.emplace_from(
                    stdexec::connect,
                    stdexec::schedule(stdexec::get_delegation_scheduler(stdexec::get_env(rcvr))),
                    ScheduleWaitEventReceiver<Exec, Rcvr>{this});
                stdexec::start(inner_opstate.get());
            } catch (...) {
                stdexec::set_error(std::move(rcvr), std::current_exception());
            }
        }
    }
};

struct SubmittedPolicyTag { };

/**
 * Under a submitted policy, an intermediate completion is propagated, signaling
 * to the receiver that the operation has been submitted to the execution space instance,
 * but has not necessarily completed yet:
 *   - @ref OrderOnExec informs that the operation was submitted on the underlying execution space instance
 *   - @ref DependOnEvent provides an event recorded on the underlying execution space instance after submission
 */
struct SubmittedPolicy {
    struct OrderOnExec : SubmittedPolicyTag { };
    struct DependOnEvent : SubmittedPolicyTag { };
};

template <typename Policy>
concept submitted_policy = std::derived_from<Policy, SubmittedPolicyTag>;

template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionSignal<SubmittedPolicy::OrderOnExec, Exec, Rcvr> {
    Rcvr rcvr;

    void propagate(const Exec&) & noexcept {
        std::move(rcvr).submitted();
    }
};

/**
 * @brief Create an event in the execution space instance, and pass a handle to the event to the receiver,
 *        thus deferring to the successor the responsibility to wait on the event.
 *
 * Skip the creation of the event if synchronization is not required.
 */
template <Kokkos::ExecutionSpace Exec, stdexec::receiver Rcvr>
struct CompletionSignal<SubmittedPolicy::DependOnEvent, Exec, Rcvr> {
    using event_storage_t = Impl::event_storage_t<Exec>;

    Rcvr rcvr;
    event_storage_t event_storage = std::nullopt;

    void propagate(const Exec& exec) & noexcept {
        if (RequiresSync<Exec, Rcvr>{}(exec, rcvr)) {
            auto& event = event_storage.emplace();
            Impl::record(event, exec);
            std::move(rcvr).submitted(OptionalConstEventRef{event});
        } else {
            std::move(rcvr).submitted(OptionalConstEventRef<Exec>{});
        }
    }
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_COMPLETION_SIGNAL_HPP
