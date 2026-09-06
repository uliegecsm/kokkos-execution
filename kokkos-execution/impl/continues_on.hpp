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
requires Impl::signals_submitted_order_on<InnerOp>
struct DependencyFor<InnerOp, ExecTo> {
    using type = Impl::Dependency<ExecTo, Impl::exec_of_t<InnerOp>>;
};

template <typename InnerOp, typename ExecTo>
using dependency_for_t = typename DependencyFor<InnerOp, ExecTo>::type;

template <typename InnerOp, typename ExecTo, typename Rcvr>
concept signals_submitted_order_on = ((Impl::signals_submitted_order_on<InnerOp>
                                       && std::same_as<ExecTo, Impl::exec_of_t<InnerOp>>)
                                      || (Impl::signals_submitted_depend_on<InnerOp>
                                          && stdexec::__mapply<
                                              stdexec::__many_of<stdexec::__mbind_front_q<stdexec::__msame_as, ExecTo>>,
                                              typename InnerOp::inner_opstate_t::child_execs_t
                                          >::value))
                                  && Impl::supports_submitted_order_on<Rcvr>;

template <typename InnerOp, typename ExecTo, typename Rcvr>
using completion_signal_policy_t = std::conditional_t<
    ContinuesOn::signals_submitted_order_on<InnerOp, ExecTo, Rcvr>,
    Impl::SubmittedPolicy::OrderOnExec,
    Impl::SyncPolicy::InlineFenceExec
>;

} // namespace Kokkos::Execution::Impl::ContinuesOn

#endif // KOKKOS_EXECUTION_IMPL_CONTINUES_ON_HPP
