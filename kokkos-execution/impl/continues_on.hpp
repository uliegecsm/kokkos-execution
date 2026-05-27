#ifndef KOKKOS_EXECUTION_IMPL_CONTINUES_ON_HPP
#define KOKKOS_EXECUTION_IMPL_CONTINUES_ON_HPP

#include "kokkos-execution/impl/dependency.hpp"
#include "kokkos-execution/impl/empty.hpp"
#include "kokkos-execution/impl/submitted.hpp"

namespace Kokkos::Execution::Impl::ContinuesOn {

template <typename InnerOp, typename ExecTo>
struct DependencyFor {
    using type = Impl::Empty;
};

template <typename InnerOp, typename ExecTo>
requires Impl::signals_submitted<InnerOp>
struct DependencyFor<InnerOp, ExecTo> {
    using type = Impl::Dependency<ExecTo, Impl::exec_of_t<InnerOp>>;
};

template <typename InnerOp, typename ExecTo>
using dependency_for_t = typename DependencyFor<InnerOp, ExecTo>::type;

template <typename InnerOp, typename ExecTo, typename Rcvr>
concept signals_submitted = Impl::signals_submitted<InnerOp> && std::same_as<ExecTo, Impl::exec_of_t<InnerOp>>
                         && Impl::supports_submitted_order_on<Rcvr>;

template <typename InnerOp, typename ExecTo, typename Rcvr>
using completion_signal_policy_t = std::conditional_t<
    ContinuesOn::signals_submitted<InnerOp, ExecTo, Rcvr>,
    Impl::SubmittedPolicy::OrderOnExec,
    Impl::SyncPolicy::InlineFenceExec
>;

} // namespace Kokkos::Execution::Impl::ContinuesOn

#endif // KOKKOS_EXECUTION_IMPL_CONTINUES_ON_HPP
