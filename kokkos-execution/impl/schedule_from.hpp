#ifndef KOKKOS_EXECUTION_IMPL_SCHEDULE_FROM_HPP
#define KOKKOS_EXECUTION_IMPL_SCHEDULE_FROM_HPP

#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::Impl::ScheduleFrom {

template <typename InnerOp, typename Rcvr>
concept signals_submitted = Impl::signals_submitted<InnerOp> && Impl::supports_submitted_order_on<Rcvr>;

template <typename InnerOp, typename Rcvr>
using completion_signal_policy_t = std::conditional_t<
    ScheduleFrom::signals_submitted<InnerOp, Rcvr>,
    Impl::SubmittedPolicy::OrderOnExec,
    Impl::SyncPolicy::InlineFenceExec
>;

} // namespace Kokkos::Execution::Impl::ScheduleFrom

#endif // KOKKOS_EXECUTION_IMPL_SCHEDULE_FROM_HPP
