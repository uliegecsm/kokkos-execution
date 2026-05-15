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

template <typename ExecEnvPolicy, stdexec::receiver Rcvr>
struct ScheduleFromReceiver {
    using receiver_concept = Impl::SubmittedReceiverTag;

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
        //! Stay in the @ref Kokkos::Execution::ExecutionSpaceImpl::Domain.
        if constexpr (Impl::supports_submitted<Rcvr>) {
            rcvr.submitted(order_on);
        }
        //! Transition to another domain.
        else {
            try {
                order_on.synchronize(std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<Exec>::name()));
                stdexec::set_value(std::move(rcvr));
            } catch (...) {
                stdexec::set_error(std::move(rcvr), std::current_exception());
            }
        }
    }

    [[nodiscard]]
    constexpr auto get_env() const noexcept -> extend_env_t<ExecEnvPolicy, stdexec::env_of_t<Rcvr>> {
        return extend_env<ExecEnvPolicy>(stdexec::get_env(this->rcvr));
    }
};

template <stdexec::sender Sndr>
struct ScheduleFromSender {
    using sender_concept = stdexec::sender_tag;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender)

    template <typename Rcvr>
    using rcvr_t = ScheduleFromReceiver<exec_env_policy_t<stdexec::env_of_t<Rcvr>>, Rcvr>;

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
    template <typename Env, typename Sndr>
    auto operator()(const Env&, stdexec::schedule_from_t, stdexec::__ignore, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ScheduleFromSender<Sndr>, Sndr&&>) {
        return ScheduleFromSender<Sndr>{.sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
