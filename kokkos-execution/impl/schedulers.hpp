#ifndef KOKKOS_EXECUTION_IMPL_SCHEDULERS_HPP
#define KOKKOS_EXECUTION_IMPL_SCHEDULERS_HPP

#include "kokkos-execution/stdexec.hpp"

namespace Kokkos::Execution::Impl {

template <typename Env>
using delegation_scheduler_of_t = stdexec::__query_result_t<const Env&, stdexec::get_delegation_scheduler_t>;

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_SCHEDULERS_HPP
