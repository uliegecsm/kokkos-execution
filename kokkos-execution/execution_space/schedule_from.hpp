#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP

#include "stdexec/execution.hpp"

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"

#include "kokkos-execution/execution_space/sender_concepts.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/env.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

//! Receiver for @c schedule_from.
template <stdexec::scheduler Schd, stdexec::receiver Rcvr>
struct ScheduleFromReceiver {
    using receiver_concept = stdexec::receiver_t;

    Schd schd;
    Rcvr rcvr;

    void set_value() && noexcept {
        //! If the downstream receiver is our customization of @c continues_on and it shares the same execution space instance, skip the fence.
        const bool skip = [&]() {
            if constexpr (stdexec::__is_instance_of<Rcvr, ContinuesOnReceiver>) {
                if constexpr (stdexec::__queryable_with<stdexec::env_of_t<Rcvr>, get_exec_t>) {
                    if constexpr (
                        std::same_as<
                            std::remove_cvref_t<decltype(get_exec(stdexec::get_env(rcvr)).get())>,
                            typename Schd::execution_space
                        >) {
                        return schd.state->exec == get_exec(stdexec::get_env(rcvr)).get();
                    }
                }
            }
            return false;
        }();
        if (!skip)
            schd.state->exec.fence(
                std::format("{}: schedule_from", Kokkos::Impl::TypeInfo<typename Schd::execution_space>::name()));
        stdexec::set_value(std::move(rcvr));
    }

    template <typename Error>
    void set_error(Error&& err) && noexcept {
        stdexec::set_error(std::move(rcvr), std::forward<Error>(err));
    }

    void set_stopped() && noexcept {
        stdexec::set_stopped(std::move(rcvr));
    }

    //! Make others aware of which execution space instance it may synchronize.
    KOKKOS_EXECUTION_UPSERT_EXEC(typename Schd::execution_space, schd.state->exec, Rcvr, rcvr)
};

//! Sender for @c schedule_from.
template <stdexec::scheduler Schd, stdexec::sender Sndr>
struct ScheduleFromSender {
    using sender_concept = stdexec::sender_t;

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ScheduleFromSender)

    template <typename Rcvr>
    using rcvr_t = ScheduleFromReceiver<Schd, Rcvr>;

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

template <>
struct TransformSenderFor<stdexec::schedule_from_t> {
    template <typename Sndr, typename Env>
    using schd_t = stdexec::__completion_scheduler_of_t<stdexec::set_value_t, Sndr, Env>;

    template <typename Sndr, typename Env>
    using sndr_t = ScheduleFromSender<schd_t<Sndr, Env>, Sndr>;

    template <typename Env, execution_space_completing_sender<Env> Sndr>
    auto operator()(const Env& env, stdexec::schedule_from_t, stdexec::__ignore, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<sndr_t<Sndr, Env>, schd_t<Sndr, Env>&&, Sndr&&>) {

        return sndr_t<Sndr, Env>{
            .schd = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(sndr), env),
            .sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_SCHEDULE_FROM_HPP
