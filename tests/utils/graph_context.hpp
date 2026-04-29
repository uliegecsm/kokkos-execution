#ifndef KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP
#define KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP

#include "Kokkos_Graph_fwd.hpp"
#include "impl/Kokkos_DeviceHandle.hpp"

#include "kokkos-execution/graph/graph_fwd.hpp"

#include "tests/utils/context.hpp"

namespace Tests::Utils {

template <Kokkos::ExecutionSpace Exec>
struct GraphContextTest : public ContextTest<Kokkos::Execution::GraphContext, Exec> {
    using device_handle_t = Kokkos::Impl::DeviceHandle<Exec>;
    using graph_t = Kokkos::Experimental::Graph<Exec>;

    device_handle_t device_handle{this->exec};
};

} // namespace Tests::Utils

#endif // KOKKOS_EXECUTION_TESTS_UTILS_GRAPH_CONTEXT_HPP
