#ifndef KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP

#include "kokkos-execution/utils/ignore_warnings.hpp"
PRAGMA_DIAGNOSTIC_PUSH
KOKKOS_EXECUTION_STDEXEC_PRAGMA_DIAGNOSTIC_IGNORED
#include "exec/env.hpp"
PRAGMA_DIAGNOSTIC_POP

#include "kokkos-execution/execution_space/execution_space_fwd.hpp"
#include "kokkos-execution/execution_space/get_exec.hpp"
#include "kokkos-execution/impl/attributes.hpp"
#include "kokkos-execution/impl/completion_signatures.hpp"
#include "kokkos-execution/impl/continues_on.hpp"
#include "kokkos-execution/impl/env.hpp"

namespace Kokkos::Execution::ExecutionSpaceImpl {

struct FwdWithExec { };
struct FwdWithoutExec { };

//! Receiver for @c continues_on.
template <stdexec::receiver Rcvr, typename FwdPolicy = FwdWithExec>
struct ContinuesOnReceiver {
    using receiver_concept = stdexec::receiver_t;

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
    constexpr auto get_env() const noexcept -> std::conditional_t<
        std::same_as<FwdPolicy, FwdWithoutExec>,
        stdexec::__call_result_t<
            experimental::execution::__envs::__without_t,
            stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>,
            get_exec_t
        >,
        stdexec::__fwd_env_t<stdexec::env_of_t<Rcvr>>
    > {
        if constexpr (std::same_as<FwdPolicy, FwdWithoutExec>) {
            return experimental::execution::without(stdexec::__fwd_env(stdexec::get_env(rcvr)), get_exec);
        } else {
            return stdexec::__fwd_env(stdexec::get_env(rcvr));
        }
    }
};

//! Sender for @c continues_on.
template <stdexec::sender Sndr>
struct ContinuesOnSender {
    using sender_concept = stdexec::sender_t;

    Sndr sndr; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

    KOKKOS_EXECUTION_COMPL_SIGS_KEEP(ContinuesOnSender)

    template <
        stdexec::receiver Rcvr,
        typename FwdPolicy = std::conditional_t<
            std::same_as<
                stdexec::__detail::__completing_domain_t<stdexec::set_value_t, Sndr, stdexec::env_of_t<Rcvr>>,
                Domain
            >
                || has_when_all_child_with_at_least_one_child_completing_on_v<
                    stdexec::set_value_t,
                    Domain,
                    Sndr,
                    stdexec::env_of_t<Rcvr>
                >
                || has_fork_join_child_with_at_least_one_child_completing_on_v<
                    stdexec::set_value_t,
                    Domain,
                    Sndr,
                    stdexec::env_of_t<Rcvr>
                >,
            FwdWithExec,
            FwdWithoutExec
        >
    >
    auto connect(Rcvr rcvr) && noexcept(std::is_nothrow_move_constructible_v<Rcvr>)
        -> stdexec::connect_result_t<Sndr, ContinuesOnReceiver<Rcvr, FwdPolicy>> {
        using recv_t = ContinuesOnReceiver<Rcvr, FwdPolicy>;

        return stdexec::connect(std::forward<Sndr>(sndr), recv_t{.rcvr = std::move(rcvr)});
    }

    KOKKOS_EXECUTION_IMPL_FORWARDING_ATTRIBUTES_GET_ENV(Sndr, sndr)
};

template <>
struct TransformSenderFor<stdexec::continues_on_t> {
    template <typename Env, stdexec::__is_instance_of<Scheduler> Schd, stdexec::sender Sndr>
    auto operator()(const Env&, stdexec::continues_on_t, Schd&&, Sndr&& sndr) const
        noexcept(std::is_nothrow_constructible_v<ContinuesOnSender<Sndr>, Sndr&&>) {
        return ContinuesOnSender<Sndr>{.sndr = std::forward<Sndr>(sndr)};
    }
};

} // namespace Kokkos::Execution::ExecutionSpaceImpl

#endif // KOKKOS_EXECUTION_EXECUTION_SPACE_CONTINUES_ON_HPP
