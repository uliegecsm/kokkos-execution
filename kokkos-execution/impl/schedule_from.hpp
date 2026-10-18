#ifndef KOKKOS_EXECUTION_IMPL_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_IMPL_SCHEDULE_FROM_HPP

#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::Impl::ScheduleFrom {

template <typename InnerOp, typename Rcvr>
concept signals_submitted_order_on = Impl::signals_submitted_order_on<InnerOp> && Impl::supports_submitted<Rcvr>;

template <typename InnerOp, typename Rcvr>
concept signals_submitted_depend_on = Impl::signals_submitted_depend_on<InnerOp> && Impl::supports_submitted<Rcvr>;

template <typename InnerOp, typename Rcvr>
using completion_signal_policy_t = std::conditional_t<
    ScheduleFrom::signals_submitted_order_on<InnerOp, Rcvr>,
    Impl::SubmittedPolicy::OrderOnExec,
    std::conditional_t<
        ScheduleFrom::signals_submitted_depend_on<InnerOp, Rcvr>,
        Impl::SubmittedPolicy::DependOnEvent,
        Impl::SyncPolicy::InlineFenceExec
    >
>;

} // namespace Kokkos::Execution::Impl::ScheduleFrom

#endif // KOKKOS_EXECUTION_IMPL_SCHEDULE_FROM_HPP
