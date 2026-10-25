#ifndef KOKKOS_EXECUTION_IMPL_STATE_HPP
#define KOKKOS_EXECUTION_IMPL_STATE_HPP

#include "Kokkos_Core.hpp"
#include "Kokkos_Graph.hpp"

namespace Kokkos::Execution::Impl {

template <Kokkos::ExecutionSpace Exec>
struct State {
    [[nodiscard]]
    auto get_device_handle() const noexcept -> Kokkos::Impl::DeviceHandle<Exec> {
        return Kokkos::Experimental::get_device_handle(exec);
    }

    Exec exec;
};

} // namespace Kokkos::Execution::Impl

#endif // KOKKOS_EXECUTION_IMPL_STATE_HPP
