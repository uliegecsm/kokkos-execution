#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP

#include "kokkos-execution/stdexec.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"
#include "kokkos-execution/impl/get_exec.hpp"
#include "kokkos-execution/impl/sender_introspection.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct WithDelegatedSyncPolicy { };
struct WithoutDelegatedSyncPolicy { };

//! Receiver for @c schedule_from.
template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ScheduleFromReceiver<WithDelegatedSyncPolicy, WithoutExecEnvPolicy, Schd, Rcvr> {
    using receiver_concept = Impl::SubmittedReceiverTag;

    Schd schd;
    Rcvr rcvr;

    void set_value() && noexcept {
        stdexec::set_value(std::move(rcvr));
    }

    void submitted(Impl::OrderOn<typename Schd::execution_space>) & noexcept {
        /// If the downstream receiver environment is queryable for @ref Kokkos::Execution::Impl::get_exec
        /// and it shares the same execution space instance, skip the fence.
        const bool requires_synchronization = [&]() {
            if constexpr (stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, Impl::get_exec_t>) {
                if constexpr (
                    std::same_as<
                        typename std::remove_cvref_t<
                            stdexec::__query_result_t<stdexec::env_of_t<Rcvr>, Impl::get_exec_t>
                        >::execution_space,
                        typename Schd::execution_space
                    >) {
                    return Impl::get_exec(stdexec::get_env(rcvr)).get() != schd.state->exec;
                }
            }
            return true;
        }();
        if (requires_synchronization) {
            schd.state->exec.fence(
                std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<typename Schd::execution_space>::name()));
        }
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        stdexec::set_stopped(std::move(rcvr));
    }

    KOKKOS_EXECUTION_FORWARDING_GET_ENV(Rcvr, rcvr)
};

template <typename ExecEnvPolicy, typename Rcvr>
struct ScheduleFromReceiver<WithoutDelegatedSyncPolicy, ExecEnvPolicy, Rcvr> {
    using receiver_concept = stdexec::receiver_tag;

    Rcvr rcvr;

    void set_value() && noexcept {
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        stdexec::set_stopped(std::move(rcvr));
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> extend_env_t<ExecEnvPolicy, stdexec::env_of_t<Rcvr>> {
        return extend_env<ExecEnvPolicy>(stdexec::get_env(this->rcvr));
    }
};

template <typename...>
struct ScheduleFromSender;

//! Sender for @c schedule_from.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ScheduleFromSender<WithDelegatedSyncPolicy, Schd, Sndr> {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender)

    template <typename Rcvr>
    using rcvr_t = ScheduleFromReceiver<WithDelegatedSyncPolicy, WithoutExecEnvPolicy, Schd, Rcvr>;

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<rcvr_t<Rcvr>, Schd&&, Rcvr&&>
        && stdexec::__nothrow_connectable<Sndr&&, rcvr_t<Rcvr>>) {
        return stdexec::connect(
            std::forward<Sndr>(sndr), rcvr_t<Rcvr>{.schd = std::move(schd), .rcvr = std::move(rcvr)});
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Schd schd;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <stdexec::sender Sndr>
struct ScheduleFromSender<WithoutDelegatedSyncPolicy, Sndr> {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender)

    template <typename Rcvr>
    using rcvr_t = ScheduleFromReceiver<WithoutDelegatedSyncPolicy, exec_env_policy_t<stdexec::env_of_t<Rcvr>>, Rcvr>;

    template <stdexec::receiver Rcvr>
    stdexec::operation_state auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<rcvr_t<Rcvr>, Rcvr&&> && stdexec::__nothrow_connectable<Sndr&&, rcvr_t<Rcvr>>) {
        return stdexec::connect(std::forward<Sndr>(sndr), rcvr_t<Rcvr>{std::move(rcvr)});
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

template <>
struct TransformSenderFor<stdexec::schedule_from_t> {
    template <typename Sndr, typename Env>
    using schd_t = Impl::completion_scheduler_of_t<stdexec::set_value_t, Sndr, Env>;

    template <typename Env, execution_space_completing_sender<Env> Sndr>
    auto operator()(const Env& env, stdexec::schedule_from_t, stdexec::__ignore, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<
                 ScheduleFromSender<WithDelegatedSyncPolicy, schd_t<Sndr, Env>, Sndr>,
                 schd_t<Sndr, Env>&&,
                 Sndr&&
        >) {

        return ScheduleFromSender<WithDelegatedSyncPolicy, schd_t<Sndr, Env>, Sndr>{
            .schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env),
            .sndr = std::forward<Sndr>(sndr)};
    }

    template <typename Env, typename Sndr>
    requires(!stdexec::__has_completion_scheduler_for<stdexec::set_value_t, Sndr, Env>)
    auto operator()(const Env&, stdexec::schedule_from_t, stdexec::__ignore, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ScheduleFromSender<WithoutDelegatedSyncPolicy, Sndr>, Sndr&&>) {
        return ScheduleFromSender<WithoutDelegatedSyncPolicy, Sndr>{.sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
