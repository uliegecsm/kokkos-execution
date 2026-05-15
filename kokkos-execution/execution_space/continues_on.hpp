#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "kokkos-execution/execution_space/env.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Receiver for @c continues_on.
template <typename ExecEnvPolicy, stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ContinuesOnReceiver {
    using receiver_concept = Impl::SubmittedReceiverTag;

    Schd schd;
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

    template <Kokkos::ExecutionSpace Exec>
    void submitted(Impl::OrderOn<Exec> order_on) & noexcept {
        if constexpr (std::same_as<Exec, typename Schd::execution_space>) {
            // Continue on same instance.
            if (order_on.exec() == schd.state->exec) {
                // No wait.
            }
            // Transition to another instance of the same execution space type.
            else {
                // @todo Make an event and stream-wait-event. This requires an opstate to store the event.
                order_on.exec().fence(std::format("{}: continues_on", Kokkos::Impl::TypeInfo<Exec>::name()));
            }
        }
        //! Transition to exec of different type.
        else {
            order_on.exec().fence(std::format("{}: continues_on", Kokkos::Impl::TypeInfo<Exec>::name()));
        }
        stdexec::set_value(std::move(rcvr));
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept
        -> join_env_with_exec_t<ExecEnvPolicy, stdexec::env_of_t<Rcvr>, typename Schd::execution_space> {
        return join_env_with_exec<ExecEnvPolicy>(stdexec::get_env(rcvr), schd.state->exec);
    }
};

//! Sender for @c continues_on.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_tag;

    Schd schd;
    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <typename Rcvr>
    using exec_env_policy_t = std::conditional_t<
        std::same_as<
            stdexec::__completion_domain_of_t<stdexec::set_value_t, Sndr, stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>>,
            Domain
        >,
        WithExecEnvPolicy,
        WithoutExecEnvPolicy
    >;

    template <typename Rcvr>
    using rcvr_t = ContinuesOnReceiver<exec_env_policy_t<Rcvr>, Schd, Rcvr>;

    template <stdexec::receiver Rcvr>
    auto connect(Rcvr rcvr) && noexcept(
        std::is_nothrow_constructible_v<rcvr_t<Rcvr>, Schd&&, Rcvr&&>
        && stdexec::__nothrow_connectable<Sndr&&, rcvr_t<Rcvr>>) -> stdexec::connect_result_t<Sndr, rcvr_t<Rcvr>> {
        return stdexec::connect(
            std::forward<Sndr>(sndr), rcvr_t<Rcvr>{.schd = std::forward<Schd>(schd), .rcvr = std::move(rcvr)});
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <>
struct TransformSenderFor<stdexec::continues_on_t> {
    template <typename Env, stdexec::__is_instance_of<Scheduler> Schd, stdexec::sender Sndr>
    auto operator()(const Env&, stdexec::continues_on_t, Schd&& schd, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ContinuesOnSender<Schd, Sndr>, Schd&&, Sndr&&>) {
        return ContinuesOnSender<Schd, Sndr>{.schd = std::forward<Schd>(schd), .sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
