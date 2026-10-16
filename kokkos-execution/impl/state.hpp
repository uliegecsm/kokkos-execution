#ifndef KOKKOS_EXECUTION_IMPL_STATE_HPP
#define KOKKOS_EXECUTION_IMPL_STATE_HPP

#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#    include "Kokkos_Core.hpp"
#endif

namespace Kokkos::Execution::Impl {

template <Kokkos::ExecutionSpace Exec>
struct State {
    Exec exec;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_STATE_HPP
