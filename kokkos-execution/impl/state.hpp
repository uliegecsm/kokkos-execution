#ifndef KOKKOS_EXECUTION_IMPL_STATE_HPP
#define KOKKOS_EXECUTION_IMPL_STATE_HPP

#include "Kokkos_Core.hpp"

namespace Kokkos::Execution::Impl {

template <Kokkos::ExecutionSpace Exec>
struct State {
    Exec exec;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_STATE_HPP
